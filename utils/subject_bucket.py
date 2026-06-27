"""Subject bucketing + diminishing-returns repeat counts for parquet datasets.

NOT part of the upstream diffusion-pipe PR — this is the subject-specific layer
(ported from AbstractEyes/anima-trainer's subject_buckets.py /
build_multiconcept_dataset.py) that PRODUCES a bucket manifest. The generic hook
that CONSUMES it lives in utils/bucket_manifest.py and is already wired into
ParquetDirectoryDataset; this module just writes the manifest, fully offline.

What's ported (the useful methodology):
- subject extraction + normalization (head-noun, singularize),
- subject -> bucket planning (lexical difflib by default; optional semantic
  agglomerative clustering when numpy+sklearn are available),
- oversized-bucket splitting (rarest attribute -> secondary subject -> chunk),
- dampened_repeats: the "adjusted contribution effect" repeat-count policy.

What's intentionally NOT ported: image-folder export, hardlinks, the reconstruct
index / SHA plumbing, HF shard materialization. We keep the images in parquet and
only emit a manifest {id -> num_repeats}.

Dependency-light: stdlib + (optionally) numpy/sklearn for semantic clustering, and
utils.parquet_source for reading. No torch/deepspeed.

CLI:
    python -m utils.subject_bucket --dataset AbstractPhil/diffusion-pretrain-set-ft1 \
        --config sdxl_qwen_phase0 --caption-column caption_vlm_json \
        --out manifest_vlm.json [--alpha 0.5 --max-repeats 8 --semantic]
Then in your dataset toml's [[directory]]:
    bucket_manifest = 'manifest_vlm.json'
    manifest_key_column = 'id'
"""

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import get_close_matches


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class SubjectBucketConfig:
    head_noun: bool = True                 # "fire truck" -> "truck"
    min_bucket_size: int = 10              # below this a subject is small (merged/grouped)
    fuzzy_cutoff: float = 0.62             # difflib ratio to merge a small subject into a big one
    keep_small: bool = True                # leftovers -> weighted misc_* (never dropped)
    # repeat policy (the "adjusted contribution effect")
    balance_alpha: float = 0.5             # 0=equalize(overtrain), 1=no balance, 0.5=sqrt damping
    cap_mult: float = 1.25                 # cap a bucket's effective samples at cap_mult*top
    max_repeats: int = 8                   # per-image exposure ceiling
    target_effective: int | None = None    # top reference (default = largest bucket's images)
    default_repeats: int = 1               # for rows with no parseable subject
    # oversized split
    split_oversized: bool = True
    max_bucket_size: int | None = None     # None -> data-dependent
    attr_min_split: int = 2
    split_separator: str = "."
    # semantic grouping (optional; falls back to lexical difflib if deps absent)
    use_semantic: bool = False
    human_min_size: int = 4
    sim_threshold: float = 0.50
    min_final_group_size: int = 8


# =============================================================================
# SUBJECT / ATTRIBUTE NORMALIZATION  (ported)
# =============================================================================
_ARTICLES = {"a", "an", "the"}
_IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child",
              "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose"}
_ATTR_STOP = {"1girl", "1boy", "2girls", "2boys", "solo", "general", "sensitive",
              "simple_background", "white_background", "looking_at_viewer"}
_SLUG = re.compile(r"[^a-zA-Z0-9_.-]+")


def _singularize(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalize_subject(name, *, head_noun: bool = True):
    """'Fire Truck' -> 'truck', 'the Police Officers' -> 'officer'."""
    if not name:
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower()).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    if not toks:
        return None
    key = toks[-1] if head_noun else " ".join(toks)
    return _singularize(key) or None


def normalize_attr(a):
    """'Blonde Hair' -> 'blonde_hair'; '1girl' -> None (count/meta tags dropped)."""
    if not a:
        return None
    s = re.sub(r"[^a-z0-9 _]+", " ", str(a).lower()).strip()
    s = re.sub(r"[ _]+", "_", s).strip("_")
    if not s or s in _ATTR_STOP:
        return None
    return s


def slug(s) -> str:
    return _SLUG.sub("_", str(s).strip()).strip("_") or "unknown"


def _subject_name(s):
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        return s.get("name")
    return None


def caption_features(caption_json, *, head_noun: bool = True):
    """(dominant_subject, attrs of subjects[0], secondary subject) from a caption JSON
    string of the form {"subjects":[{"name","attributes":[...]}, ...], ...}."""
    if not caption_json or (isinstance(caption_json, str) and caption_json.startswith("__")):
        return None, (), None
    try:
        obj = json.loads(caption_json)
    except (json.JSONDecodeError, TypeError):
        return None, (), None
    if not isinstance(obj, dict):
        return None, (), None
    subs = obj.get("subjects") or []
    if not subs:
        return None, (), None
    subject = normalize_subject(_subject_name(subs[0]), head_noun=head_noun)
    attrs = tuple(x for x in (normalize_attr(a) for a in (
        subs[0].get("attributes") if isinstance(subs[0], dict) else []) or []) if x)
    secondary = normalize_subject(_subject_name(subs[1]), head_noun=head_noun) if len(subs) > 1 else None
    return subject, attrs, secondary


# =============================================================================
# REPEAT COUNTS  —  the "adjusted contribution effect"  (ported, faithful)
# =============================================================================
def dampened_repeats(images: int, top: int, *, alpha: float = 0.5,
                     max_repeats: int = 8, cap_mult: float = 1.25) -> int:
    """Bounded, diminishing-returns num_repeats for one bucket.

    repeats = round((top/images) ** (1 - alpha)), then clamped:
      alpha=0 -> equalize (small buckets scaled up to `top`; OVERTRAINS sparse concepts),
      alpha=1 -> no balancing (all 1x),
      alpha=0.5 -> sqrt damping (default): big buckets ~1x, sparse ones a bounded lift.
    Caps per-image exposure at `max_repeats` and effective samples at `cap_mult*top`.
    """
    if images <= 0:
        return 1
    rep = max(1, round((top / images) ** (1.0 - alpha)))
    rep = min(rep, max_repeats)
    if images * rep > cap_mult * top:               # effective-samples ceiling
        rep = max(1, int(cap_mult * top // images))
    return rep


def bucket_repeats(bucket_sizes: dict, cfg: SubjectBucketConfig) -> dict:
    """bucket -> num_repeats using dampened_repeats; top = target or largest bucket."""
    if not bucket_sizes:
        return {}
    top = cfg.target_effective or max(bucket_sizes.values())
    return {b: dampened_repeats(n, top, alpha=cfg.balance_alpha,
                                max_repeats=cfg.max_repeats, cap_mult=cfg.cap_mult)
            for b, n in bucket_sizes.items()}


# =============================================================================
# BUCKET PLANNING
# =============================================================================
@dataclass
class BucketPlan:
    mapping: dict                                   # normalized subject -> bucket name
    raw_counts: Counter = field(default_factory=Counter)


def plan_buckets_lexical(subjects, cfg: SubjectBucketConfig) -> BucketPlan:
    """Big subjects are buckets; merge small ones into the nearest big via difflib;
    leftovers -> 'misc' (kept) or dropped. Stdlib only."""
    counts = Counter(s for s in subjects if s)
    big = {k for k, c in counts.items() if c >= cfg.min_bucket_size}
    mapping = {k: k for k in big}
    big_list = sorted(big, key=lambda k: -counts[k])
    for k, c in counts.most_common():
        if k in big:
            continue
        match = get_close_matches(k, big_list, n=1, cutoff=cfg.fuzzy_cutoff)
        if match:
            mapping[k] = match[0]
        elif cfg.keep_small:
            mapping[k] = "misc"
        else:
            mapping[k] = None
    return BucketPlan(mapping=mapping, raw_counts=counts)


def plan_buckets_semantic(subjects, cfg: SubjectBucketConfig):
    """Optional: protect big+human subjects, cluster the sparse tail by char-trigram
    cosine similarity (agglomerative avg-linkage), fold tiny leftovers into misc_*.
    Returns None if numpy/sklearn unavailable (caller falls back to lexical)."""
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except Exception:
        return None
    counts = Counter(s for s in subjects if s)
    subs = [s for s in counts]
    if not subs:
        return BucketPlan(mapping={}, raw_counts=counts)

    def trigram_embed(phrases, dim=1024):
        mat = np.zeros((len(phrases), dim), "float32")
        for r, p in enumerate(phrases):
            s = "^^" + re.sub(r"[^a-z0-9 ]+", "", str(p).lower()) + "$$"
            for i in range(len(s) - 2):
                mat[r, hash(s[i:i + 3]) % dim] += 1.0
        n = np.linalg.norm(mat, axis=1, keepdims=True); n[n == 0] = 1.0
        return mat / n

    emb = trigram_embed(subs)
    idx = {s: i for i, s in enumerate(subs)}
    human_seed = ("person", "man", "woman", "child", "boy", "girl", "player",
                  "worker", "performer", "crowd", "figure", "human")
    seed_emb = trigram_embed(list(human_seed))
    agentive = re.compile(r"(ist|er|man|woman|person|girl|boy)$")
    human = {}
    for s in subs:
        sim = float((emb[idx[s]] @ seed_emb.T).max()) if len(human_seed) else 0.0
        human[s] = sim >= cfg.sim_threshold or bool(agentive.search(s))

    big = {s for s in subs if counts[s] >= cfg.min_bucket_size}
    protected = big | {s for s in subs if human[s] and counts[s] >= cfg.human_min_size}
    mapping = {s: s for s in protected}

    def cluster(side):
        if len(side) <= 1:
            return [list(side)] if side else []
        sidx = np.fromiter((idx[s] for s in side), dtype=np.intp, count=len(side))
        sub = emb[sidx] @ emb[sidx].T
        dist = np.clip(1.0 - sub, 0.0, 2.0); np.fill_diagonal(dist, 0.0)
        labels = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                         linkage="average",
                                         distance_threshold=1.0 - cfg.sim_threshold).fit(dist).labels_
        comps = defaultdict(list)
        for s, lab in zip(side, labels):
            comps[int(lab)].append(s)
        return list(comps.values())

    small = [s for s in subs if s not in protected]
    for prefix, misc, side in [("grp_h_", "misc_human", [s for s in small if human[s]]),
                               ("grp_", "misc_other", [s for s in small if not human[s]])]:
        for members in cluster(side):
            total = sum(counts[m] for m in members)
            if len(members) > 1 and total >= cfg.min_final_group_size:
                name = prefix + slug(max(members, key=lambda m: counts[m]))
                for m in members:
                    mapping[m] = name
            else:
                for m in members:
                    mapping[m] = misc if cfg.keep_small else None
    return BucketPlan(mapping=mapping, raw_counts=counts)


# =============================================================================
# OVERSIZED SPLIT  (ported, simplified to operate on row ids)
# =============================================================================
def _max_bucket_size(total_images, override=None):
    if override is not None:
        return override
    if total_images > 10_000:
        return 1_000
    if total_images >= 1_000:
        return 500
    return 250


def _even_chunks(seq, n):
    n = max(1, n); k, m = divmod(len(seq), n)
    out, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(seq[start:start + size]); start += size
    return [c for c in out if c]


def _split_bucket(ids, subj, M, attrs_of, secondary_of, cfg):
    """Partition one oversized bucket's row ids: rarest attribute -> secondary -> chunk."""
    sep = cfg.split_separator
    attr_freq = Counter(a for i in ids for a in attrs_of.get(i, ()))
    groups = defaultdict(list); no_attr = []
    for i in ids:
        cand = [a for a in attrs_of.get(i, ()) if a]
        if cand:
            key = min(cand, key=lambda a: (attr_freq[a], a))
            groups[f"{subj}{sep}{key}"].append(i)
        else:
            no_attr.append(i)
    for name in [n for n, m in list(groups.items()) if len(m) < cfg.attr_min_split]:
        no_attr.extend(groups.pop(name))
    for i in no_attr:
        sec = secondary_of.get(i)
        groups[f"{subj}{sep}with_{sec}" if sec else f"{subj}{sep}plain"].append(i)
    final = {}
    for name, members in groups.items():
        if len(members) <= M:
            final[name] = members
        else:
            for c, part in enumerate(_even_chunks(sorted(members), math.ceil(len(members) / M))):
                final[f"{name}_{c:02d}"] = part
    return final


# =============================================================================
# MANIFEST BUILD
# =============================================================================
def build_manifest(rows, cfg: SubjectBucketConfig) -> dict:
    """rows: iterable of (row_key, caption_json_string).
    Returns a bucket manifest dict consumable by utils.bucket_manifest.load_bucket_manifest:
        {"default_repeats": N, "rows": {key: {"num_repeats": r, "bucket": b}}}
    """
    keys, subjects = [], []
    attrs_of, secondary_of = {}, {}
    for key, cap in rows:
        subj, attrs, sec = caption_features(cap, head_noun=cfg.head_noun)
        keys.append(key); subjects.append(subj)
        attrs_of[key] = attrs; secondary_of[key] = sec

    plan = None
    if cfg.use_semantic:
        plan = plan_buckets_semantic([s for s in subjects if s], cfg)
    if plan is None:
        plan = plan_buckets_lexical([s for s in subjects if s], cfg)

    # row -> bucket (rows with no subject / dropped -> a kept "misc_no_subject")
    NO_SUBJ = "misc_no_subject"
    row_bucket = {}
    for key, subj in zip(keys, subjects):
        b = plan.mapping.get(subj) if subj else None
        row_bucket[key] = b if b is not None else NO_SUBJ

    # oversized split (operates on the per-row bucket assignment)
    if cfg.split_oversized:
        M = _max_bucket_size(len(keys), cfg.max_bucket_size)
        by_bucket = defaultdict(list)
        for key, b in row_bucket.items():
            by_bucket[b].append(key)
        for b, ids in list(by_bucket.items()):
            if b == NO_SUBJ or len(ids) <= M:
                continue
            for name, members in _split_bucket(ids, b, M, attrs_of, secondary_of, cfg).items():
                for key in members:
                    row_bucket[key] = name

    bucket_sizes = Counter(row_bucket.values())
    reps = bucket_repeats(dict(bucket_sizes), cfg)
    reps[NO_SUBJ] = min(reps.get(NO_SUBJ, cfg.default_repeats), cfg.default_repeats)

    out_rows = {}
    for key in keys:
        b = row_bucket[key]
        out_rows[key] = {"num_repeats": int(reps.get(b, cfg.default_repeats)), "bucket": b}
    return {
        "default_repeats": cfg.default_repeats,
        "num_buckets": len(bucket_sizes),
        "rows": out_rows,
    }


def iter_caption_rows(directory_config, caption_column, key_column="id", num_proc=4):
    """Yield (key, raw_caption_json) from a parquet/HF source, reading only the key +
    caption columns (no image bytes)."""
    import pyarrow.parquet as pq
    from utils.parquet_source import resolve_parquet_source
    source = resolve_parquet_source(directory_config, num_proc=num_proc)
    for shard in source.shard_paths:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[key_column, caption_column])
            ks = tbl.column(key_column).to_pylist()
            cs = tbl.column(caption_column).to_pylist()
            for k, c in zip(ks, cs):
                yield k, c


def generate_manifest(directory_config, caption_column, cfg, key_column="id"):
    rows = list(iter_caption_rows(directory_config, caption_column, key_column))
    manifest = build_manifest(rows, cfg)
    return manifest


def _main():
    ap = argparse.ArgumentParser(description="Build a subject-bucketed repeat manifest from parquet.")
    ap.add_argument("--dataset", help="HF dataset repo id (type=huggingface)")
    ap.add_argument("--config", help="HF config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--parquet-files", help="glob/path for local parquet (type=parquet)")
    ap.add_argument("--caption-column", default="caption_vlm_json")
    ap.add_argument("--key-column", default="id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap-mult", type=float, default=1.25)
    ap.add_argument("--max-repeats", type=int, default=8)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--min-bucket-size", type=int, default=10)
    ap.add_argument("--semantic", action="store_true", help="use semantic clustering (needs numpy+sklearn)")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args()

    if args.parquet_files:
        directory_config = {"type": "parquet", "parquet_files": args.parquet_files}
    else:
        directory_config = {"type": "huggingface", "dataset": args.dataset,
                            "config": args.config, "split": args.split}
    cfg = SubjectBucketConfig(balance_alpha=args.alpha, cap_mult=args.cap_mult,
                              max_repeats=args.max_repeats, target_effective=args.target,
                              min_bucket_size=args.min_bucket_size, use_semantic=args.semantic,
                              split_oversized=not args.no_split)
    manifest = generate_manifest(directory_config, args.caption_column, cfg, args.key_column)
    with open(args.out, "w") as f:
        json.dump(manifest, f)

    # report
    sizes = Counter(r["bucket"] for r in manifest["rows"].values())
    reps = {r["bucket"]: r["num_repeats"] for r in manifest["rows"].values()}
    print(f"rows={len(manifest['rows'])} buckets={manifest['num_buckets']} -> {args.out}")
    print(f"{'bucket':<32}{'images':>8}{'repeats':>9}{'effective':>11}")
    for b, n in sizes.most_common(25):
        print(f"{b:<32}{n:>8}{reps[b]:>9}{n*reps[b]:>11}")


if __name__ == "__main__":
    _main()
