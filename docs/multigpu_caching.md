# Multi-GPU (data-parallel) latents caching

Cache the latents of a large parquet dataset across **all** your GPUs, instead of the one GPU
the built-in `--cache_only` actually uses. On a 4× A40 box this turns a ~15 h latents cache
(83 k images) into ~30–90 min. This is an **opt-in tool** — the normal single-GPU cache path is
unchanged and remains the default.

---

## TL;DR — the streamlined command

```bash
# 1. Build the latents cache across N GPUs (one command; hides all the complexity below):
python cache_multigpu.py --config path/to/train.toml --num_gpus 4

# 2. Train exactly as normal, but add --trust_cache so the trainer adopts the merged cache:
deepspeed --num_gpus=4 train.py --deepspeed --config path/to/train.toml --trust_cache
```

`cache_multigpu.py` prints that exact step-2 command when it finishes. That is the whole
workflow. Everything below is *why* it works and the two gotchas that bite if you hand-roll it.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--config <toml>` | (required) | the training TOML you will train with |
| `--num_gpus <N>` | (required) | how many GPUs to shard caching across |
| `--shards_per_job <k>` | `1` | shards per single-GPU job. **Leave at 1** unless you know your `parquet_shard_lru` covers it (the tool auto-bumps the per-job `parquet_shard_lru` to `k+1` to stay deadlock-safe) |
| `--master_port <p>` | `29500` | base port; the job on GPU *g* uses `p+g`. Raise it if 29500–29500+N are taken or you run two cache jobs on one box |
| `--work_dir <dir>` | `<cache_root>/.mgpu_work` | scratch dir for per-job caches, generated configs, and `cache_job####.log` logs |
| `--keep_jobs` | off | keep the scratch `work_dir` after a successful merge (for debugging); normally it is deleted |

On any job that cannot cache latents after retries, the tool **aborts without merging** (a partial
cache would crash training), leaving the per-job logs in `--work_dir` for inspection.

---

## Why this exists

`train.py --cache_only` is a **single producer**: rank 0 runs one CPU decode pool that feeds a
*serialized* GPU consumer, so on an N-GPU box only ~1 GPU does useful VAE-encode work.
`map_num_proc` does not fix it (the decode pool deadlocks above ~16 workers). Latent VAE-encode
is the bottleneck.

`cache_multigpu.py` shards the **latents** caching across all N GPUs, then merges the per-shard
caches into the single cache root your training config already points at. (Each per-job
`--cache_only` also encodes text-embeddings into its scratch cache, but the merge copies **only the
`latents/`** dirs and discards the rest; the normal training run rebuilds text-embeddings — which
are caption-keyed and always fingerprint-checked — for the exact caption column you train on.)

---

## How it works

Given your normal training TOML, the tool:

1. Reads the referenced dataset TOML, finds the one `parquet`/`huggingface` `[[directory]]`, and
   resolves it to a list of local parquet shards (via `utils.parquet_source.resolve_parquet_source`).
2. Splits the shards into **one-shard jobs** and schedules them **N-at-a-time in waves**. Each job
   is an independent single-GPU `train.py --cache_only` over a copy of your config pinned to that
   one shard and a private cache dir. GPU pinning uses `deepspeed --include localhost:<gpu>` with a
   distinct `MASTER_PORT` per GPU.
3. **Merges** every job's per-size-bucket `latents/` shards into your real cache root and writes a
   fresh `_index.json` per bucket (placeholder fingerprint `merged_multigpu`), then deletes the
   scratch `work_dir`.

**Disk usage:** the merge *moves* (renames, same filesystem) rather than copies, and the scratch
dir is deleted afterward, so peak disk is ~1× the final cache size, not 2×. Budget roughly the size
of the latents cache you're building (plus your source shards, which the jobs read in place).

Your subsequent `train.py --trust_cache` run then adopts that merged cache and only has to build
text-embeddings (fast) before training.

```
shards ── wave 0 ─┬─ GPU0: job0 (shard0) ─► cache_j0000/…/latents
                  ├─ GPU1: job1 (shard1) ─► cache_j0001/…/latents
                  ├─ GPU2: job2 (shard2) ─► cache_j0002/…/latents
                  └─ GPU3: job3 (shard3) ─► cache_j0003/…/latents
        ── wave 1 ─┬─ GPU0: job4 (shard4) …                      merge
                    …                       ─────────────────────►  <cache_root>/…/latents
```

---

## The two rules the tool encodes for you

### 1. One shard per job (deadlock avoidance)

Do **not** hand a single caching job many shards with `parquet_shard_lru < num_shards`.
AR-bucketing interleaves images across all of a job's shards, but each decode worker can only hold
`parquet_shard_lru` shard image-columns. If `lru < num_shards`, the workers thrash-evict-reload
forever and the producer/consumer pipeline **deadlocks at batch 0** (looks frozen: GPU 0 %, tqdm
stuck at `0/N`; the consumer waits on `unix_stream_data_wait`, the decode workers on
`futex_wait_queue`). The CLAUDE.md rule is *"set `parquet_shard_lru >= num_shards`"* — but for many
big shards that OOMs (e.g. 14 shards × 1.3 GB × 8 workers × 4 ranks ≈ 580 GB).

**One shard per job** makes `lru ≥ num_shards` trivially true with low memory, and matches the
proven working condition. The tool defaults `--shards_per_job 1`; raise it only if you also raise
`parquet_shard_lru` to cover it (and have the RAM).

### 2. The merge is order-independent (safe by construction)

The per-shard caches are concatenated in arbitrary order. This is safe because the trainer's latent
read is **image_spec-keyed**: `utils.dataset` builds `latents_idx` from the latent cache's *own*
`image_spec` column (`ParquetCache.image_specs()`), so each dataset row maps to wherever its latent
actually landed. A merge bug can therefore only cause a **safe re-cache**, never a silently
misaligned latent. (For the legacy sqlite cache, which has no `image_specs()`, the code falls back
to metadata order — the multi-GPU tool only targets the parquet backend.)

---

## Gotcha: per-phase caches for `before/after` (different caption columns)

`skip_empty_caption = true` drops rows whose caption is empty / parse-fail — **per caption column**.
Two caption columns (e.g. a VLM column and a tagger column) therefore have **different surviving row
sets**. A latents cache built for column A **cannot** be reused by a run reading column B: the
trainer expects `cache-set == dataset-set` (its `_map_and_cache` asserts `cache_size <= dataset_size`
then re-encodes a *positional* tail, and the image_spec read `KeyError`s on rows unique to B). The
result is a hard crash **hours in**, not silent corruption — but it wastes the run.

Measured example (83,020-row Anima set): `caption_vlm_json` → 82,328 survivors, `caption_animetimm_json`
→ 79,324 survivors, with **642 rows unique to animetimm**. Sharing one cache across those two phases
crashes phase 2.

**Rule:** for a 2-phase before/after run, give each phase its own cache root and cache each
separately:

```bash
python cache_multigpu.py --config train_phaseA.toml --num_gpus 4   # caption col A -> cache_A
# ... train phase A ...
python cache_multigpu.py --config train_phaseB.toml --num_gpus 4   # caption col B -> cache_B
# ... train phase B (init_from_existing = phase-A adapter) ...
```

(Each phase's dataset TOML sets a different `path`.) Only share one cache across phases if you have
**verified** both columns have identical survivor sets. To check quickly, count survivors per column
with `utils.parquet_source.extract_caption(cell, caption_type, json_path) is not None`.

---

## Enabling code changes (already in this fork)

Two small, gated changes in the core make the merge safe and adoptable (legacy path unchanged):

- **`utils/parquet_cache.py`**
  - `ParquetCache(..., trust_cache=False)` + `image_specs()`. When `trust_cache` is set and a stored
    `_index.json` fingerprint doesn't match, the cache is **adopted as-is** (orphans pruned) instead
    of cleared — this is what lets a merged cache (placeholder fingerprint) be read.
- **`utils/dataset.py`**
  - `make_cache` / `_map_and_cache` / `SizeBucketDataset` thread `trust_cache` through to the latents
    cache only (text-embeddings are always fingerprint-checked, so a phase always builds its *own*
    captions' embeddings — verified).
  - The iteration-order build keys `latents_idx` on the cache's own `image_specs()` (order-independent)
    with a `len(map) == len(latent_cache)` assertion catching any duplicate/missing spec, falling back
    to metadata order for caches without `image_specs()`.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| tqdm frozen at `0/N`, GPU 0 %, not I/O-bound (`ps -eo stat \| grep ^D` is 0) | decode-pool deadlock from `parquet_shard_lru < shards/job` | use `--shards_per_job 1` (default) |
| phase-2 `KeyError` / `assert cache_size <= dataset_size` hours in | shared cache across two different caption columns | per-phase caches (above) |
| training re-encodes everything despite the cache | forgot `--trust_cache` (merged fingerprint won't match) | add `--trust_cache` to the train command |
| a job crashes / hangs (`BrokenPipeError`, OOM, teardown hang holding GPU memory) | intermittent decode-pipe break under N-way concurrency | the tool auto-kills the hung process and **retries** the job (up to 2×); nothing to do |
| `[mgpu] ABORT: N job(s) could not cache latents` | a job failed every retry (usually a genuinely bad shard or persistent OOM) | the tool did **not** merge (no partial cache). Inspect that job's log in `--work_dir` (`cache_job####.log`, default `<cache_root>/.mgpu_work/`); lower `map_num_proc` or `--num_gpus` if OOM, then re-run |

**Diagnosing a suspected freeze:** compare tqdm progress across two snapshots (identical = frozen),
`cat /proc/<pid>/wchan` on the stuck main + worker procs (`unix_stream_data_wait` + `futex_wait_queue`
= producer/consumer deadlock), and `df -h /dev/shm` (rule out a tiny shm).

---

## Limitations

- **Machine-local cache.** Each latent is keyed by its image's `image_spec`, which embeds the
  **absolute** source parquet path. A merged cache is therefore valid on the machine where it was
  built; moving `cache_root` to a box where `resolve_parquet_source` yields different absolute shard
  paths won't match (the training run would re-cache). Durability across a pod cull is handled by the
  cache's own HF upload/download (`cache_hf_repo`/`cache_hf_upload`), which re-resolves paths — not by
  copying `cache_root` between boxes.
- **One parquet source.** The tool pre-caches a single `parquet`/`huggingface` `[[directory]]`. Any
  folder (`type='directory'`) sources in the same dataset TOML are left for the normal training run to
  cache single-GPU (the tool prints a note when it sees them).

---

## Testing

Pure logic (job planning, merge, config copy-override) is unit-tested without a GPU:

```bash
python test/test_cache_multigpu.py
```

The GPU launch path (`run_waves`) is exercised by real multi-GPU cache runs.
