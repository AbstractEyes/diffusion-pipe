"""Synthesize character images with Qwen-Image Lightning (4-step) and write an HF-ingestible
parquet dataset, with race-bias mitigation + adult age constraint (see prompt_policy.py).

Reads ONLY the `prompt` column of a source dataset (default the FFHQ llava captions), rewrites
each prompt for a fair race mix / age 25-35, generates an image with Qwen-Image + the lightx2v
4-step Lightning LoRA, and writes shards whose `image` column is an HF Image feature
(struct<bytes,path>) so the result is both Hub-viewable AND directly trainable by diffusion-pipe's
parquet input source. Data-parallel across GPUs (one process per GPU); resumable; deterministic.

Smoke test (1 GPU, 64 imgs, no upload):
  CUDA_VISIBLE_DEVICES=0 python -m qwen_extraction.qwen_lightning_extraction \
    --rank 0 --world-size 1 --limit 64 --batch-size 4 --no-upload --local-dir /workspace/smoke

Full run (4 GPUs):
  export HF_TOKEN=hf_xxx HF_HUB_ENABLE_HF_TRANSFER=1
  for R in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$R python -m qwen_extraction.qwen_lightning_extraction \
      --rank $R --world-size 4 --out-repo AbstractPhil/qwen-synth-characters \
      --local-dir /workspace/qwen_synth_out --batch-size 8 > /workspace/log_rank$R.txt 2>&1 & done; wait
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback

# Make `qwen_extraction.prompt_policy` importable whether run as a module or a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from qwen_extraction import prompt_policy
except ImportError:
    import prompt_policy  # when run from inside the qwen_extraction dir

RATIOS = [(1024, 1024), (832, 1216), (1216, 832)]
SHARD_FMT = "shard_r{rank}_{n:05d}.parquet"
SOURCE_DATASET = "AbstractPhil/ffhq_with_llava_shorter_captions_flux_latents"
BASE_MODEL = "Qwen/Qwen-Image"
LORA_REPO = "lightx2v/Qwen-Image-Lightning"
LORA_WEIGHT = "Qwen-Image-Lightning-4steps-V1.0.safetensors"


# --------------------------------------------------------------------------------------
# Output schema (HF Image feature -> Hub-viewable + diffusion-pipe ingestible)
# --------------------------------------------------------------------------------------
def build_arrow_schema():
    """Arrow schema with HF `Features` metadata so `image` renders as a thumbnail on the Hub
    and is read as struct<bytes,path> by utils/parquet_source.ParquetImageReader."""
    from datasets import Features, Image, Value
    features = Features({
        "id": Value("string"),
        "image": Image(),                  # arrow storage = struct<bytes: binary, path: string>
        "image_width": Value("int32"),
        "image_height": Value("int32"),
        "prompt": Value("string"),         # the AUGMENTED prompt actually generated with
        "source_prompt": Value("string"),
        "race": Value("string"),
        "race_injected": Value("bool"),
        "is_tail": Value("bool"),
        "gender": Value("string"),
        "age_band": Value("string"),
        "hair": Value("string"),
        "eye": Value("string"),
        "expression": Value("string"),
        "makeup": Value("string"),
        "jewelry": Value("string"),
        "is_amateur": Value("bool"),
        "seed": Value("int64"),
        "width_ratio": Value("string"),
        "policy_version": Value("string"),
    })
    return features, features.arrow_schema


# --------------------------------------------------------------------------------------
# Source prompt loader (columnar; never touches the big `latent` column)
# --------------------------------------------------------------------------------------
def load_prompts(dataset: str, column: str, limit: int = 0, config: str = "", json_path: str = ""):
    """Return list[(id:str, prompt:str)]. Reads ONLY `column` from the source parquet via HTTP range
    reads (HfFileSystem), so big columns (latents/images) are never downloaded.
    - config: restrict to the dataset config subdir, e.g. 'deepfashion' -> data/deepfashion/*.parquet.
    - json_path: if set, each cell is a JSON object/string and this key is extracted (e.g.
      column='captions_source_json', json_path='deepfashion_caption').
    `id` is the stable global row index (prefixed by config) across shards in sorted-file order."""
    import json as _json
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    rels = sorted(f for f in api.list_repo_files(dataset, repo_type="dataset") if f.endswith(".parquet"))
    if config:
        rels = [f for f in rels if config in f.replace("\\", "/").split("/")]
    if not rels:
        raise RuntimeError(f"No parquet files found in {dataset} (config={config!r})")

    def extract(v):
        if json_path:
            if v is None:
                return ""
            d = v if isinstance(v, dict) else _json.loads(v)
            val = d.get(json_path) if isinstance(d, dict) else None
            return val if isinstance(val, str) else ("" if val is None else str(val))
        return v if isinstance(v, str) else ("" if v is None else str(v))

    fs = HfFileSystem()
    prefix = f"{config}_" if config else ""
    out = []
    gidx = 0
    for rel in rels:
        with fs.open(f"datasets/{dataset}/{rel}") as fh:     # range-read; footer + one column only
            pf = pq.ParquetFile(fh)
            if column not in pf.schema_arrow.names:
                raise RuntimeError(f"Column {column!r} not in {rel}; have {pf.schema_arrow.names}")
            tbl = pf.read(columns=[column])
        for v in tbl.column(column).to_pylist():
            cap = extract(v)
            out.append((f"{prefix}{gidx}", cap))
            gidx += 1
            if limit and len(out) >= limit:
                return out
    return out


def load_prompts_file(path: str, limit: int = 0):
    """Return list[(id, caption)] from a 'id<TAB>caption' TSV (synthetic-caption batches)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rid, _, cap = line.partition("\t")
            out.append((rid, cap))
            if limit and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------------------
# Incremental shard writer (PNG bytes in an HF Image struct; ~shard_size_mb shards; resumable)
# --------------------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, local_dir, rank, out_repo=None, upload=True,
                 shard_size_mb=350, image_format="png"):
        self.rank = rank
        self.dir = os.path.join(local_dir, f"rank{rank}")
        os.makedirs(self.dir, exist_ok=True)
        self.out_repo = out_repo
        self.upload = upload and bool(out_repo)
        self.shard_size = int(shard_size_mb * 1024 * 1024)
        self.image_format = image_format
        self.features, self.schema = build_arrow_schema()
        self.index_path = os.path.join(self.dir, "_index.json")
        self.index = self._load_index()
        self.done_ids = set()
        for sh in self.index["shards"]:
            self.done_ids.update(sh.get("ids", []))
        self._buf = []
        self._buf_bytes = 0
        self._prune_orphans()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"rank": self.rank, "next_shard": 0, "shards": []}

    def _write_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.index, f)
        os.replace(tmp, self.index_path)

    def _prune_orphans(self):
        keep = {sh["file"] for sh in self.index["shards"]}
        for name in os.listdir(self.dir):
            if name.endswith(".tmp") or (name.endswith(".parquet") and name not in keep):
                try:
                    os.remove(os.path.join(self.dir, name))
                except OSError:
                    pass

    def add(self, row: dict, image_bytes: bytes):
        row = dict(row)
        row["image"] = {"bytes": image_bytes, "path": None}
        self._buf.append(row)
        self._buf_bytes += len(image_bytes)
        if self._buf_bytes >= self.shard_size:
            self.finalize_current_shard()

    def finalize_current_shard(self):
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        n = self.index["next_shard"]
        fname = SHARD_FMT.format(rank=self.rank, n=n)
        fpath = os.path.join(self.dir, fname)
        tmp = fpath + ".tmp"
        table = pa.Table.from_pylist(self._buf, schema=self.schema)
        pq.write_table(table, tmp, compression="none", use_dictionary=False)
        os.replace(tmp, fpath)
        ids = [r["id"] for r in self._buf]
        self.index["shards"].append({"file": fname, "rows": len(ids), "ids": ids, "uploaded": False})
        self.index["next_shard"] = n + 1
        self.done_ids.update(ids)
        self._write_index()
        if self.upload:
            self._upload_shard(fname)
        self._buf, self._buf_bytes = [], 0

    def _upload_shard(self, fname):
        from huggingface_hub import upload_file
        local = os.path.join(self.dir, fname)
        repo_path = f"data/rank{self.rank}/{fname}"
        for attempt in range(3):
            try:
                upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                            repo_id=self.out_repo, repo_type="dataset")
                for sh in self.index["shards"]:
                    if sh["file"] == fname:
                        sh["uploaded"] = True
                self._write_index()
                # also publish the index so a fresh machine can resume
                upload_file(path_or_fileobj=self.index_path,
                            path_in_repo=f"data/rank{self.rank}/_index.json",
                            repo_id=self.out_repo, repo_type="dataset")
                return
            except Exception as e:
                wait = 2 * (4 ** attempt)
                print(f"[rank{self.rank}] upload {fname} attempt {attempt+1} failed: {repr(e)[:140]}; "
                      f"retry in {wait}s", flush=True)
                time.sleep(wait)
        print(f"[rank{self.rank}] upload {fname} FAILED after retries; kept local, marked pending", flush=True)


# --------------------------------------------------------------------------------------
# Generator (lazy torch/diffusers import so the rest of the module is GPU-free testable)
# --------------------------------------------------------------------------------------
class QwenLightningGenerator:
    def __init__(self, steps=4, true_cfg_scale=1.0, negative=" ", offload="auto",
                 base=BASE_MODEL, lora_repo=LORA_REPO, lora_weight=LORA_WEIGHT, max_seq_len=512):
        import torch
        from diffusers import DiffusionPipeline
        self.torch = torch
        self.steps = steps
        self.true_cfg_scale = true_cfg_scale
        self.negative = negative
        self.max_seq_len = max_seq_len
        print(f"loading {base} (bf16) + LoRA {lora_repo}/{lora_weight}", flush=True)
        self.pipe = DiffusionPipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
        self.pipe.load_lora_weights(lora_repo, weight_name=lora_weight)
        try:
            self.pipe.fuse_lora()
        except Exception as e:
            print(f"fuse_lora skipped: {repr(e)[:100]}", flush=True)
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(self.pipe.vae, fn)()
            except Exception:
                pass
        self._place_pipeline(offload)

    def _place_pipeline(self, offload):
        # Qwen-Image is a ~20B model (~44GB bf16 pipeline).
        #   'off'  = resident on one GPU (FASTEST; needs the whole pipeline to fit, e.g. >=80GB cards).
        #   'on'   = cpu offload (fits a 48GB A40 but ~10x slower: re-stages the DiT every batch).
        #   'auto' = try resident, fall back to offload on OOM (default; fast on big cards, works on small).
        torch = self.torch
        if offload == "on":
            self.pipe.enable_model_cpu_offload()
            self.mode = "offload"
        else:
            try:
                self.pipe.to("cuda")
                self.mode = "resident"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if offload == "off":
                    raise
                print("resident load OOM -> enable_model_cpu_offload()", flush=True)
                self.pipe.enable_model_cpu_offload()
                self.mode = "offload"
        print(f"pipeline placement: {self.mode}", flush=True)

    def generate(self, prompts, width, height, seeds):
        gens = [self.torch.Generator("cuda").manual_seed(int(s) % (2 ** 63 - 1)) for s in seeds]
        kwargs = dict(prompt=list(prompts), negative_prompt=self.negative,
                      num_inference_steps=self.steps, width=width, height=height, generator=gens)
        try:
            return self.pipe(true_cfg_scale=self.true_cfg_scale, **kwargs).images
        except TypeError:
            # older diffusers exposes guidance_scale instead of true_cfg_scale
            return self.pipe(guidance_scale=self.true_cfg_scale, **kwargs).images


def encode_image(img, fmt="png"):
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", optimize=False)
    elif fmt in ("jpg", "jpeg"):
        img.convert("RGB").save(buf, format="JPEG", quality=95)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=95)
    else:
        raise ValueError(fmt)
    return buf.getvalue()


def is_black(img):
    import numpy as np
    return float(np.asarray(img.convert("L")).mean()) < 2.0


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def main():
    import random

    ap = argparse.ArgumentParser(description="Qwen-Image Lightning synthetic-character extraction.")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--out-repo", default="AbstractPhil/qwen-synth-characters")
    ap.add_argument("--local-dir", default="/workspace/qwen_synth_out")
    ap.add_argument("--source-dataset", default=SOURCE_DATASET)
    ap.add_argument("--source-column", default="prompt")
    ap.add_argument("--source-config", default="", help="restrict to a dataset config subdir, e.g. 'deepfashion'")
    ap.add_argument("--source-json-path", default="", help="extract this key from a JSON column, e.g. 'deepfashion_caption'")
    ap.add_argument("--domain", default="face", choices=["face", "fashion"],
                    help="augmentation policy: face (portraits) or fashion (full-body outfits)")
    ap.add_argument("--prompts-file", default="",
                    help="TSV of 'id<TAB>caption' lines; overrides the HF source (synthetic batches)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="cap total prompts (smoke test)")
    ap.add_argument("--shard-size-mb", type=int, default=350)
    ap.add_argument("--image-format", default="png", choices=["png", "jpeg", "webp"])
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--negative-prompt", default=" ")
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--offload", default="auto", choices=["auto", "on", "off"],
                    help="GPU placement: off=resident (fastest, needs >=80GB); on=cpu-offload "
                         "(fits 48GB but ~10x slower); auto=resident then offload on OOM")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable the sentence-similarity anti-flooding resampler")
    ap.add_argument("--dedup-threshold", type=float, default=0.82,
                    help="token-shingle Jaccard above which a prompt counts as a near-duplicate")
    ap.add_argument("--dry-run", action="store_true", help="augment + write a tiny solid image, no GPU")
    args = ap.parse_args()

    cfg = prompt_policy.PromptAugmentConfig(domain=args.domain)
    if args.prompts_file:
        print(f"[rank{args.rank}/{args.world_size}] loading prompts from {args.prompts_file} (domain={args.domain})", flush=True)
        prompts = load_prompts_file(args.prompts_file, args.limit)
    else:
        print(f"[rank{args.rank}/{args.world_size}] loading prompts from {args.source_dataset} "
              f"config={args.source_config!r} (domain={args.domain})", flush=True)
        prompts = load_prompts(args.source_dataset, args.source_column, args.limit,
                               config=args.source_config, json_path=args.source_json_path)
    mine = prompts[args.rank::args.world_size]
    print(f"[rank{args.rank}] {len(prompts)} total, {len(mine)} for this rank", flush=True)

    writer = ShardWriter(args.local_dir, args.rank, out_repo=args.out_repo,
                         upload=not args.no_upload, shard_size_mb=args.shard_size_mb,
                         image_format=args.image_format)
    if not args.no_resume and writer.done_ids:
        before = len(mine)
        mine = [(i, p) for (i, p) in mine if i not in writer.done_ids]
        print(f"[rank{args.rank}] resume: {before - len(mine)} already done, {len(mine)} remaining", flush=True)

    gen = None
    if not args.dry_run:
        gen = QwenLightningGenerator(steps=args.steps, true_cfg_scale=args.true_cfg_scale,
                                     negative=args.negative_prompt, offload=args.offload,
                                     max_seq_len=args.max_seq_len)

    # Anti-flooding: per-rank sentence-similarity guard (resamples attributes on near-duplicates).
    guard = None if args.no_dedup else prompt_policy.DiversityGuard(threshold=args.dedup_threshold)

    t0 = time.time()
    done = 0
    resampled = 0
    batch_idx = 0
    for start in range(0, len(mine), args.batch_size):
        batch = mine[start:start + args.batch_size]
        # deterministic per-batch ratio
        rrng = random.Random(prompt_policy.stable_seed(f"{args.seed_base}:{args.rank}:{batch_idx}"))
        width, height = RATIOS[rrng.randrange(len(RATIOS))]
        batch_idx += 1

        metas = [prompt_policy.augment_diverse(p, rid, cfg, guard=guard) for (rid, p) in batch]
        resampled += sum(m.get("dedup_resamples", 0) > 0 for m in metas)
        # skip empties / non-person prompts that produced no text
        rows = [(rid, m) for (rid, _), m in zip(batch, metas) if m["final_prompt"]]
        if not rows:
            continue
        aug_prompts = [m["final_prompt"] for _, m in rows]
        seeds = [args.seed_base ^ m["seed"] for _, m in rows]

        images = _generate_with_backoff(gen, aug_prompts, width, height, seeds, args)

        for (rid, m), img in zip(rows, images):
            png = encode_image(img, args.image_format)
            writer.add({
                "id": rid,
                "image_width": int(img.width),
                "image_height": int(img.height),
                "prompt": m["final_prompt"],
                "source_prompt": m["source_prompt"] or "",
                "race": m["race"] or "",
                "race_injected": bool(m["race_injected"]),
                "is_tail": bool(m["is_tail"]),
                "gender": m["gender"] or "",
                "age_band": m["age_band"] or "",
                "hair": m["hair"] or "",
                "eye": m.get("eye") or "",
                "expression": m.get("expression") or "",
                "makeup": m.get("makeup") or "",
                "jewelry": m.get("jewelry") or "",
                "is_amateur": bool(m.get("is_amateur")),
                "seed": int(m["seed"]),
                "width_ratio": f"{width}x{height}",
                "policy_version": m["policy_version"],
            }, png)
            done += 1

        if (start // args.batch_size) % 10 == 0 and done:
            rate = done / max(1e-6, time.time() - t0)
            print(f"[rank{args.rank}] {done}/{len(mine)} ({rate:.2f} img/s) ratio={width}x{height}", flush=True)

    writer.finalize_current_shard()
    dt = time.time() - t0
    print(f"[rank{args.rank}] DONE {done} images in {dt:.0f}s ({done/max(1e-6,dt):.2f} img/s); "
          f"dedup-resampled {resampled}", flush=True)


def _generate_with_backoff(gen, prompts, width, height, seeds, args):
    """Generate a batch with OOM backoff. In --dry-run, return tiny solid images (no GPU)."""
    if gen is None:
        from PIL import Image
        return [Image.new("RGB", (width, height), (i * 7 % 255, 80, 160)) for i in range(len(prompts))]
    import torch
    bs = len(prompts)
    while True:
        try:
            imgs = []
            for s in range(0, len(prompts), bs):
                chunk = gen.generate(prompts[s:s + bs], width, height, seeds[s:s + bs])
                for j, im in enumerate(chunk):           # black-image guard: reseed once
                    if is_black(im):
                        chunk[j] = gen.generate([prompts[s + j]], width, height,
                                                [seeds[s + j] ^ 0x9E3779B9])[0]
                imgs.extend(chunk)
            return imgs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2)
                print(f"[rank{args.rank}] OOM at {width}x{height}; retrying sub-batch={bs}", flush=True)
                continue
            if (width, height) != (1024, 1024):
                width, height = 1024, 1024
                print(f"[rank{args.rank}] OOM at bs=1; falling back to 1024x1024", flush=True)
                continue
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
