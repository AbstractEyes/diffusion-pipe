# CLAUDE.md — diffusion-pipe (AbstractEyes fork)

Guidance for working in this repo. diffusion-pipe is a multi-GPU (DeepSpeed) trainer for
diffusion image/video models (LoRA, LoKr, or full fine-tune). This fork
(`feat/parquet-hf-dataset-backend`) adds an **opt-in parquet / HuggingFace dataset input
source** and a **multi-column parquet cache backend** (HF-uploadable, resumable). The legacy
folder-of-images input + binary cache path is unchanged and remains the default — all new
behavior is opt-in via config.

## Environment reality (read this first)

- **Training runs remote** (Colab / RunPod Linux, usually multi-GPU). This Windows dev box
  has **no working full stack** (no torch+deepspeed+comfy together). Do GPU/end-to-end runs
  remotely; hand the user exact commands.
- **Local unit testing**: the dependency-light utils (`utils/parquet_cache.py`,
  `utils/parquet_source.py`, `utils/image_resize.py`, `utils/bucket_manifest.py`,
  `utils/subject_bucket.py`) import without torch-heavy deps and are tested in an isolated
  **CPU venv** (torch-cpu + pyarrow + datasets + pillow). Each `test/test_*.py` has a
  `__main__` runner — no pytest needed: `python test/test_parquet_cache.py`.
- Keep new core utils import-light so they stay locally testable.

## Running training

```bash
# from the repo root, on the remote multi-GPU box
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=N train.py --deepspeed --config examples/main_example.toml
```

- Resume: `--resume_from_checkpoint [folder_name]` (omit the value to resume the most recent).
- Cache only (no training), e.g. to pre-build/upload the cache: add `--cache_only`.

### `train.py` CLI flags

| flag | effect |
|---|---|
| `--config <path>` | TOML config (required) |
| `--cache_only` | Build the latent/text-embedding cache, then exit (no training) |
| `--trust_cache` | Load metadata/cache without re-checking fingerprints (fast for 100k+ images). **Caveat below.** |
| `--regenerate_cache` | Force-clear and rebuild the cache |
| `--resume_from_checkpoint [folder]` | Resume training; folder optional (default = most recent) |
| `--reset_dataloader` | On resume, restart the dataloader (keep optimizer state) |
| `--reset_optimizer` / `--reset_optimizer_params` | On resume, reset optimizer state/params |
| `--dump_dataset <dir>` | Decode cached latents back to images for inspection |
| `--test_sample` | Generate one `example.png` and quit |
| `--i_know_what_i_am_doing` | Skip some safety checks/overrides |
| `--master_port <int>` | Distributed master port (default 29500) |

## Config: training TOML

Top-level keys (see `examples/main_example.toml`, `examples/wan_14b_min_vram.toml`):

- `output_dir`, `dataset` (path to a dataset TOML), `epochs` **or** `max_steps`
- `micro_batch_size_per_gpu` (int, or per-resolution `[[512,4],[1024,1]]`), `gradient_accumulation_steps`
- `save_every_n_epochs` / `save_every_n_steps`, `checkpoint_every_n_minutes`
- `activation_checkpointing` (bool or `'unsloth'`), `pipeline_stages`, `blocks_to_swap` (RAM offload)
- `gradient_clipping`, `warmup_steps`, `lr_scheduler`
- `caching_batch_size`, `map_num_proc` (caching parallelism), `compile`
- eval: `eval_every_n_epochs/steps/examples`, `eval_datasets = [{name=..., config=...}]`
- `[model]` — `type` (see below), `dtype` (`'bfloat16'`), optional `transformer_dtype`/`guidance`, model-specific paths (`transformer_path`, `vae_path`, `llm_path`, …)
- `[adapter]` — `type='lora'|'lokr'`, `rank`, `alpha` (the trainer enforces `alpha=rank`), `dtype`, `dropout`, `init_from_existing` (path to a prior LoRA to continue from)
- `[optimizer]` — `type` (e.g. `'adamw'`, `'adamw_optimi'`, `'AdamW8bitKahan'`), `lr`, optional `betas`/`weight_decay`/`eps`
- `[monitoring]` — `enable_wandb`, `wandb_*`

### Model types (`[model].type`)

Dispatched in `train.py` (~line 312). Supported: `flux`, `ltx-video`, `hunyuan-video`, `sdxl`,
`cosmos`, `lumina_2`, `wan`, `chroma`, `hidream`, `sd3`, `cosmos_predict2` **or** `anima`
(both → `CosmosPredict2Pipeline`), `omnigen2`, `qwen_image`, `hunyuan_image`, `auraflow`,
`z_image`, `hunyuan_video_15`, `flux2`, `ernie_image`, `ltx2`, `ideogram4`, `krea2`.

- **Anima** = `type='anima'` → `models/cosmos_predict2.py`; uses a Qwen3-0.6B text encoder +
  `qwen_image_vae`. Public weights: `circlestone-labs/Anima` (`split_files/diffusion_models/anima-base-v1.0.safetensors`, `split_files/vae/qwen_image_vae.safetensors`, `split_files/text_encoders/qwen_3_06b_base.safetensors`). Anima text embeds are fixed-length 512; latents/embeds are bf16 (dtype preservation matters).

## Config: dataset TOML

See `examples/dataset.toml` (folder), `examples/anima_hf_parquet_dataset.toml` (parquet/HF).

Top-level: `resolutions` (`[512]` or `[[1280,720]]`), `enable_ar_bucket`, `min_ar`/`max_ar`/`num_ar_buckets`
or explicit `ar_buckets`, `frame_buckets` (video), `skip_empty_caption`.

`[[directory]]` (one or more), `type` is one of:

- **`directory`** (default): `path` (images + `.txt` captions, or a `captions.json`), optional `mask_path`, `num_repeats`.
- **`huggingface`**: `dataset` (repo id), `config`, `split`, `image_column`, `width_column`,
  `height_column`, `caption_column` (str or list), `caption_type` (`'text'`|`'json'`),
  `caption_json_path`, `num_repeats`, `skip_empty_caption`, optional `path` (cache-root override).
- **`parquet`**: `parquet_files` (glob or list) + the same image/width/height/caption keys.

AR bucketing reads `width_column`/`height_column` vectorized (no image decode); image bytes are
decoded lazily only during latent caching. Image column = HF Image feature (Arrow
`struct<bytes, path>`) or raw encoded bytes.

## Parquet / HF cache backend (this fork)

Dataset-level (top of the dataset TOML), all opt-in:

- `cache_backend = 'parquet'` (default `'legacy'` = sqlite + torch.save)
- `cache_shard_size_mb = 350`, `cache_row_group_size = 1`, `cache_compression = 'none'`
  (latents are incompressible bf16; row_group_size=1 + no compression gives fast random reads —
  do **not** raise these for big-blob caches)
- `cache_hf_repo = 'user/my-cache'`, `cache_hf_upload = true` (back up each finalized shard to a HF dataset repo; a fresh machine re-downloads instead of recomputing)
- `resize_on_gpu = true` (batched GPU resize during caching; uses a separate fingerprint so CPU caches stay valid)
- `parquet_shard_lru = 8` (whole source-shard image columns held per decode worker; set `>= num_shards` for single-pass caching)

Cache layout: `<cache_root>/cache/<model>/...` →
`cache_<bucket>/latents`, `cache_<bucket>/iteration_order[_<key>]`,
`ar_frames_.../text_embeddings_N`, `metadata/grouped_metadata_<bucket>`, `uncond_text_embeddings_N`.

### Cache reuse (encode once, reuse) — important gotchas

The **latents cache is caption/repeat-independent** (keyed by an image-identity fingerprint),
while `iteration_order` and `text_embeddings` are caption-dependent. To reuse latents across two
runs that differ only in caption set or a repeat manifest (**Option A**, verified):

1. Point **both** runs' `[[directory]].path` at the **same** cache_root.
2. The run that **builds** latents may use `--trust_cache`.
3. The run that **reuses** latents must pass **neither `--trust_cache` nor `--regenerate_cache`**
   — otherwise stale `grouped_metadata` (keyed only by size-bucket) leaks the *previous* run's
   captions/repeats. Without `--trust_cache`, metadata + iteration_order + text-embeddings
   rebuild correctly while latents are reused by fingerprint match.
4. Reuse only materializes when the **surviving row set is identical** across runs (the identity
   fingerprint covers the whole set; e.g. differing JSON-parse failures between two caption
   columns defeat reuse — then latents re-encode, which is correct).

The flag is `--regenerate_cache` (underscore).

### Caching throughput (big shards / many images)

- `TORCHINDUCTOR_COMPILE_THREADS=1` (prevents a torch._inductor compile-worker storm that starves CPU feeders)
- keep `map_num_proc` modest (4–12 for ~1.8GB shards), set `parquet_shard_lru >= num_shards`
- `resize_on_gpu = true` makes CPU workers decode-only

## Key utils

- `utils/dataset.py` — dataset loading + caching orchestration (`make_cache` factory,
  `DatasetManager`, `SizeBucketDataset`, `ParquetDirectoryDataset`, `_map_and_cache`).
- `utils/parquet_cache.py` — `ParquetCache`: multi-column parquet cache (one column per dict key;
  tensors as raw bytes + `__shape`/`__dtype`, bf16/fp8 preserved), atomic shard publish,
  `_index.json`, orphan-prune, HF upload/download.
- `utils/parquet_source.py` — `resolve_parquet_source`, `iter_parquet_rows` (metadata-only, fast
  AR bucketing), `ParquetImageReader` (lazy image-bytes decode with shard LRU). Defines the image
  schema the trainer ingests.
- `utils/image_resize.py` — CPU composite-to-RGB + GPU batched resize (kept dependency-light).
- `utils/bucket_manifest.py` — generic per-row `{num_repeats, bucket}` manifest hook (`bucket_manifest` dataset key).
- `utils/subject_bucket.py` — subject/repeat-balancing (`normalize_subject`, `dampened_repeats`,
  `build_manifest`, CLI `python -m utils.subject_bucket`). **Private — keep out of any upstream PR.**

## Tests

```bash
# in the CPU venv
python test/test_parquet_cache.py
python test/test_parquet_source.py
python test/test_image_resize.py
python test/test_bucket_manifest.py
python test/test_subject_bucket.py
python test/test_latents_decouple.py
```

## Conventions

- Don't push the private `subject_bucket` work to a public fork/PR.
- When adding cache behavior, preserve the legacy path as the default and gate new behavior on config.
- Commit messages end with the Co-Authored-By trailer; branch off `main` for PRs (current work branch: `feat/parquet-hf-dataset-backend`).
