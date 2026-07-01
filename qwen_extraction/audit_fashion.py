#!/usr/bin/env python3
"""Audit the qwen-deepfashion parquet shards: counts, balance, dedup, quality.

Runs on the pod from the repo root (so the qwen_extraction.* imports resolve):

    python3 -m qwen_extraction.audit_fashion --root /workspace/qwen_df_out \
        --sample-images 400 --contact-sheet /workspace/audit_contact_sheet.png \
        --report /workspace/audit_fashion_report.json

Import-light: pyarrow + PIL + numpy + stdlib only (no torch / no datasets), matching the
discipline of utils/parquet_source.py so it runs on the CPU venv too. Reads ONLY the small
metadata columns over the full set; the `image` column is touched solely for a bounded image
sample (saturation check + contact sheet), and only for a handful of whole shards to avoid
1-row-per-shard I/O amplification.

The studio-vs-real and garment-category breakdowns are *text-derived* from the prompt using the
dataset's own vocab (fashion_vocab); studio/real is near-exact (the policy appends one phrase
verbatim) while garment matching is reliable only for the synthetic rows (real DeepFashion
captions are free-text and mostly fall in "unmatched").
"""
import argparse
import glob
import io
import json
import os
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

# ---- optional repo imports (vocab + guards); degrade gracefully if unavailable ----
try:
    from qwen_extraction import fashion_vocab as V
except Exception:  # pragma: no cover - pod import fallback
    V = None
try:
    from qwen_extraction.prompt_policy import BW_STRIP
except Exception:  # pragma: no cover
    BW_STRIP = ["black and white", "black-and-white", "monochrome", "monochromatic",
                "greyscale", "grayscale", "sepia", "b&w", "in b and w"]
try:
    from qwen_extraction.prompt_policy import DiversityGuard
except Exception:  # pragma: no cover
    DiversityGuard = None

META_COLS = ["id", "prompt", "source_prompt", "race", "race_injected", "is_tail",
             "gender", "age_band", "is_amateur", "image_width", "image_height",
             "width_ratio", "policy_version"]

_ARTICLE = re.compile(r"^(a |an |the )")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def id_prefix(rid):
    if rid.startswith("deepfashion_"):
        return "deepfashion"
    if rid.startswith("fsyn_female_"):
        return "fsyn_female"
    if rid.startswith("fsyn_male_"):
        return "fsyn_male"
    return rid.split("_", 1)[0] if "_" in rid else "other"


def list_shards(root, ranks):
    shards = []
    for r in ranks:
        shards += sorted(glob.glob(os.path.join(root, f"rank{r}", "*.parquet")))
    return shards


def index_shard_files(root, ranks):
    """Return {rank: {"shards": N, "uploaded": M, "rows": R}} from each rank's _index.json."""
    out = {}
    for r in ranks:
        p = os.path.join(root, f"rank{r}", "_index.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
            shards = d.get("shards", [])
            out[r] = {
                "shards": len(shards),
                "uploaded": sum(1 for s in shards if s.get("uploaded")),
                "rows": sum(int(s.get("rows", 0)) for s in shards),
            }
        except Exception as e:  # pragma: no cover
            out[r] = {"error": str(e)}
    return out


def build_garment_matchers():
    """List of (needle, gender, category) sorted longest-first; needle = article-stripped phrase."""
    matchers = []
    if V is None:
        return matchers
    for src, gender in [(getattr(V, "FEMALE_GARMENTS", {}), "female"),
                        (getattr(V, "MALE_GARMENTS", {}), "male")]:
        for cat, items in src.items():
            for it in items:
                needle = _ARTICLE.sub("", it.strip().lower())
                if needle:
                    matchers.append((needle, gender, cat))
    matchers.sort(key=lambda t: len(t[0]), reverse=True)
    return matchers


def match_garment(text, matchers):
    for needle, gender, cat in matchers:
        if needle in text:
            return gender, cat
    return None, None


def normalize_prompt(p):
    return _WS.sub(" ", _PUNCT.sub("", p.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/qwen_df_out")
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="smoke test: cap number of shards scanned (0 = all)")
    ap.add_argument("--sample-images", type=int, default=400)
    ap.add_argument("--sample-shards", type=int, default=8,
                    help="how many whole shards to decode for the image sample")
    ap.add_argument("--contact-sheet", default="/workspace/audit_contact_sheet.png")
    ap.add_argument("--contact-n", type=int, default=24)
    ap.add_argument("--report", default="/workspace/audit_fashion_report.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-diversityguard", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip() != ""]
    shards = list_shards(args.root, ranks)
    if not shards:
        raise SystemExit(f"no shards under {args.root} ranks={ranks}")
    if args.max_shards:
        shards = shards[:args.max_shards]

    idx = index_shard_files(args.root, ranks)
    studio_set = [s.lower() for s in getattr(V, "STUDIO_BACKGROUNDS", [])] if V else []
    real_set = [s.lower() for s in getattr(V, "REAL_SETTINGS", [])] if V else []
    garment_matchers = build_garment_matchers()
    bw_terms = [t.lower() for t in BW_STRIP]

    # ---- aggregates ----
    total = 0
    meta_rows_by_shard = {}
    gender_c, race_c, polver_c, wr_c, prefix_c = Counter(), Counter(), Counter(), Counter(), Counter()
    n_tail = n_race_inj = n_amateur = 0
    studio_c = Counter()                              # studio / real / neither
    garment_by_provenance = defaultdict(Counter)      # "synth"/"real" -> "gender:cat"
    garment_matched = defaultdict(lambda: [0, 0])     # prov -> [matched, total]
    bw_text_ids = []
    id_to_shard = {}
    dup_ids = []                                       # (id, shard_a, shard_b)
    exact_c = Counter()
    norm_c = Counter()
    dg_sample_prompts = []

    print(f"[audit] {len(shards)} shards under {args.root}; pass A (metadata only)...", flush=True)
    for si, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        meta_rows_by_shard[shard] = pf.metadata.num_rows
        cols = [c for c in META_COLS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=4096):
            d = batch.to_pydict()
            n = len(d["id"])
            total += n
            for i in range(n):
                rid = d["id"][i]
                pre = id_prefix(rid)
                prefix_c[pre] += 1
                if rid in id_to_shard:
                    if id_to_shard[rid] != shard:
                        dup_ids.append((rid, id_to_shard[rid], shard))
                else:
                    id_to_shard[rid] = shard
                gender_c[d.get("gender", [None] * n)[i]] += 1
                race_c[d.get("race", [None] * n)[i]] += 1
                polver_c[d.get("policy_version", [None] * n)[i]] += 1
                wr_c[d.get("width_ratio", [None] * n)[i]] += 1
                if d.get("is_tail", [False] * n)[i]:
                    n_tail += 1
                if d.get("race_injected", [False] * n)[i]:
                    n_race_inj += 1
                if d.get("is_amateur", [False] * n)[i]:
                    n_amateur += 1

                prompt = (d.get("prompt", [""] * n)[i] or "")
                src = (d.get("source_prompt", [""] * n)[i] or "")
                pl = prompt.lower()

                # studio vs real (verbatim phrase match)
                if any(s in pl for s in studio_set):
                    studio_c["studio"] += 1
                elif any(s in pl for s in real_set):
                    studio_c["real"] += 1
                else:
                    studio_c["neither"] += 1

                # garment category (synth reliable; report by provenance)
                prov = "synth" if pre.startswith("fsyn") else "real"
                g_gender, g_cat = match_garment(src.lower(), garment_matchers)
                if g_cat is None:
                    g_gender, g_cat = match_garment(pl, garment_matchers)
                garment_matched[prov][1] += 1
                if g_cat is not None:
                    garment_matched[prov][0] += 1
                    garment_by_provenance[prov][f"{g_gender}:{g_cat}"] += 1

                # greyscale (text)
                if any(t in pl or t in src.lower() for t in bw_terms):
                    if len(bw_text_ids) < 50:
                        bw_text_ids.append(rid)

                # dedup
                exact_c[prompt] += 1
                norm_c[normalize_prompt(prompt)] += 1
                if len(dg_sample_prompts) < 20000 and rng.random() < 0.15:
                    dg_sample_prompts.append(prompt)
        if (si + 1) % 50 == 0:
            print(f"[audit]   {si + 1}/{len(shards)} shards, {total} rows", flush=True)

    # ---- dedup rates ----
    exact_dups = sum(c - 1 for c in exact_c.values() if c > 1)
    norm_dups = sum(c - 1 for c in norm_c.values() if c > 1)
    dedup = {
        "unique_exact": len(exact_c),
        "exact_dup_rows": exact_dups,
        "exact_dup_rate": round(exact_dups / total, 5) if total else 0,
        "unique_normalized": len(norm_c),
        "normalized_dup_rows": norm_dups,
        "normalized_dup_rate": round(norm_dups / total, 5) if total else 0,
        "top_repeated_prompt_count": exact_c.most_common(1)[0][1] if exact_c else 0,
    }
    # DiversityGuard near-dup probe on a sample (best-effort; reuses production guard)
    if DiversityGuard is not None and not args.no_diversityguard and dg_sample_prompts:
        try:
            guard = DiversityGuard(threshold=0.82)
            hits = 0
            for p in dg_sample_prompts:
                is_dup = False
                for meth in ("is_duplicate", "is_near_duplicate", "check"):
                    fn = getattr(guard, meth, None)
                    if callable(fn):
                        is_dup = bool(fn(p))
                        break
                add = getattr(guard, "add", None)
                if callable(add):
                    add(p)
                hits += int(is_dup)
            dedup["diversityguard_sample"] = len(dg_sample_prompts)
            dedup["diversityguard_neardup_rate"] = round(hits / len(dg_sample_prompts), 5)
        except Exception as e:
            dedup["diversityguard_error"] = str(e)

    # ---- image sample: saturation + contact sheet ----
    img_report = {"sampled": 0, "near_greyscale": 0, "min_saturation_examples": []}
    contact_imgs = []
    try:
        import numpy as np
        from PIL import Image
        try:
            from utils.image_resize import composite_to_rgb
        except Exception:
            def composite_to_rgb(im):
                return im.convert("RGB")

        step = max(1, len(shards) // max(1, args.sample_shards))
        sample_shards = shards[::step][:args.sample_shards]
        per = max(1, args.sample_images // max(1, len(sample_shards)))
        sats = []
        for shard in sample_shards:
            pf = pq.ParquetFile(shard)
            cols = [c for c in ("id", "gender", "is_amateur", "image") if c in pf.schema_arrow.names]
            tbl = pf.read(columns=cols)
            imgcol = tbl.column("image").to_pylist()
            ids = tbl.column("id").to_pylist() if "id" in cols else [None] * len(imgcol)
            gens = tbl.column("gender").to_pylist() if "gender" in cols else [None] * len(imgcol)
            amt = tbl.column("is_amateur").to_pylist() if "is_amateur" in cols else [None] * len(imgcol)
            picks = rng.sample(range(len(imgcol)), min(per, len(imgcol)))
            for i in picks:
                cell = imgcol[i]
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                if not raw:
                    continue
                try:
                    im = composite_to_rgb(Image.open(io.BytesIO(raw)))
                except Exception:
                    continue
                hsv = np.asarray(im.convert("HSV"))
                sat = float(hsv[..., 1].mean())
                sats.append(sat)
                img_report["sampled"] += 1
                if sat < 10.0:
                    img_report["near_greyscale"] += 1
                    if len(img_report["min_saturation_examples"]) < 20:
                        img_report["min_saturation_examples"].append(
                            {"id": ids[i], "saturation": round(sat, 2)})
                if len(contact_imgs) < args.contact_n:
                    contact_imgs.append((im, gens[i], amt[i]))
            del imgcol, tbl
        if sats:
            img_report["mean_saturation"] = round(sum(sats) / len(sats), 2)
            img_report["min_saturation"] = round(min(sats), 2)

        # contact sheet 6-wide grid
        if contact_imgs:
            from PIL import ImageDraw
            cols_n = 6
            thumb = 224
            rows_n = (len(contact_imgs) + cols_n - 1) // cols_n
            sheet = Image.new("RGB", (cols_n * thumb, rows_n * thumb), (24, 24, 24))
            draw = ImageDraw.Draw(sheet)
            for k, (im, g, a) in enumerate(contact_imgs):
                t = im.copy()
                t.thumbnail((thumb, thumb))
                x = (k % cols_n) * thumb
                y = (k // cols_n) * thumb
                sheet.paste(t, (x + (thumb - t.width) // 2, y + (thumb - t.height) // 2))
                label = f"{g or '?'}{' A' if a else ''}"
                draw.text((x + 4, y + 4), label, fill=(255, 230, 0))
            sheet.save(args.contact_sheet)
            img_report["contact_sheet"] = args.contact_sheet
    except Exception as e:
        img_report["error"] = repr(e)

    # ---- assemble report ----
    def frac(c):
        return {k: [v, round(v / total, 4)] for k, v in c.most_common()}

    report = {
        "root": args.root,
        "ranks": ranks,
        "n_shards_on_disk": len(shards),
        "index_json": idx,
        "total_rows": total,
        "id_provenance": frac(prefix_c),
        "unique_ids": len(id_to_shard),
        "cross_shard_dup_ids": len(dup_ids),
        "cross_shard_dup_examples": dup_ids[:20],
        "gender": frac(gender_c),
        "race": frac(race_c),
        "is_tail_frac": round(n_tail / total, 4) if total else 0,
        "race_injected_frac": round(n_race_inj / total, 4) if total else 0,
        "is_amateur_frac": round(n_amateur / total, 4) if total else 0,
        "width_ratio": frac(wr_c),
        "policy_version": frac(polver_c),
        "background_studio_vs_real": {k: [v, round(v / total, 4)] for k, v in studio_c.most_common()},
        "garment_match_coverage": {p: {"matched": m[0], "total": m[1],
                                       "coverage": round(m[0] / m[1], 4) if m[1] else 0}
                                   for p, m in garment_matched.items()},
        "garment_category_synth": dict(garment_by_provenance.get("synth", Counter()).most_common()),
        "garment_category_real": dict(garment_by_provenance.get("real", Counter()).most_common()),
        "greyscale_text_hits": {"count": len(bw_text_ids), "ids": bw_text_ids},
        "image_quality": img_report,
        "dedup": dedup,
    }

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    # ---- human summary ----
    print("\n========== qwen-deepfashion audit ==========")
    idx_rows = sum(v.get("rows", 0) for v in idx.values() if isinstance(v, dict))
    idx_shards = sum(v.get("shards", 0) for v in idx.values() if isinstance(v, dict))
    idx_up = sum(v.get("uploaded", 0) for v in idx.values() if isinstance(v, dict))
    print(f"shards on disk     : {len(shards)}  index: {idx_shards} shards, "
          f"{idx_up} uploaded, {idx_rows} rows")
    print(f"total rows         : {total}")
    print(f"unique ids         : {len(id_to_shard)}   cross-shard dup ids: {len(dup_ids)}")
    print(f"provenance         : {dict(prefix_c.most_common())}")
    print(f"gender             : {dict(gender_c.most_common())}")
    print(f"race (top)         : {dict(race_c.most_common(8))}  ... tail_frac={report['is_tail_frac']}")
    print(f"race_injected_frac : {report['race_injected_frac']}")
    print(f"is_amateur_frac    : {report['is_amateur_frac']}")
    print(f"width_ratio        : {dict(wr_c.most_common())}")
    print(f"policy_version     : {dict(polver_c.most_common())}")
    print(f"background          : {dict(studio_c.most_common())}")
    print(f"garment coverage    : {report['garment_match_coverage']}")
    print(f"garment synth top   : {dict(list(report['garment_category_synth'].items())[:10])}")
    print(f"greyscale text hits : {len(bw_text_ids)}")
    print(f"image sample        : {img_report.get('sampled')} sampled, "
          f"near-greyscale={img_report.get('near_greyscale')}, "
          f"mean_sat={img_report.get('mean_saturation')}")
    print(f"dedup               : exact_dup_rate={dedup['exact_dup_rate']} "
          f"normalized_dup_rate={dedup['normalized_dup_rate']} "
          f"dg_neardup={dedup.get('diversityguard_neardup_rate')}")
    print(f"report              : {args.report}")
    print(f"contact sheet       : {img_report.get('contact_sheet')}")
    print("============================================\n")


if __name__ == "__main__":
    main()
