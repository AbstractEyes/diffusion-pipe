#!/usr/bin/env python3
"""Data-parallel latents caching across N GPUs, driven by a normal training config.

WHY THIS EXISTS
---------------
The built-in `train.py --cache_only` path is a *single producer*: rank 0 runs one decode
pool that feeds a serialized GPU consumer, so on a multi-GPU box only ~1 GPU does useful
work and a 100k-image latents cache can take 10-15h. Latent VAE-encode is the bottleneck;
text-embedding encode is cheap and is left to the normal training run.

This tool shards the *latents* caching across all N GPUs and then merges the per-shard
caches into the one cache root your training config already points at. On a 4x A40 box this
turns a ~15h latents cache into ~30-90 min.

STREAMLINED ACTIVATION (one command)
------------------------------------
    python cache_multigpu.py --config path/to/train.toml --num_gpus 4

Then train exactly as normal, but add `--trust_cache` so the trainer adopts the merged
cache (its fingerprint is a placeholder; the read is image_spec-keyed so this is safe):

    deepspeed --num_gpus=4 train.py --deepspeed --config path/to/train.toml --trust_cache

The tool prints that exact follow-up command when it finishes.

WHAT IT DOES (and the two hard-won rules it encodes so you don't have to)
-------------------------------------------------------------------------
1. ONE SHARD PER JOB. Each job caches exactly one source parquet shard into its own cache
   dir, run as an independent single-GPU `train.py --cache_only`. Jobs run NGPU-at-a-time
   in waves. This is deliberate: giving one job many shards with `parquet_shard_lru <
   num_shards` makes the AR-bucketed iteration order interleave images across shards while
   each decode worker can only hold `lru` of them -> the workers thrash-evict-reload and the
   producer/consumer pipeline DEADLOCKS at batch 0. One shard per job => lru>=num_shards
   trivially => no thrash, low memory. (Raising lru to cover many big shards OOMs instead.)
2. ORDER-INDEPENDENT MERGE. The per-shard caches are concatenated in whatever order; the
   trainer's latent read is image_spec-keyed (utils.dataset builds latents_idx from the
   cache's own image_spec column via ParquetCache.image_specs()), so a merged/reordered
   cache maps every row to the correct latent. A merge bug can therefore only cause a safe
   re-cache, never a silently misaligned latent.

PER-PHASE CACHES FOR before/after (DIFFERENT CAPTION COLUMNS) -- IMPORTANT
-------------------------------------------------------------------------
`skip_empty_caption=true` drops rows whose caption is empty/parse-fail, PER caption column.
Two caption columns (e.g. a VLM column vs a tagger column) therefore have DIFFERENT surviving
row sets. A latents cache built for column A cannot be reused by a run reading column B: the
trainer expects cache-set == dataset-set and will crash (assert/KeyError) on the rows unique
to B. So for a 2-phase before/after run, cache EACH phase's config into its OWN cache root:

    python cache_multigpu.py --config train_phaseA.toml --num_gpus 4   # caption col A -> cache_A
    python cache_multigpu.py --config train_phaseB.toml --num_gpus 4   # caption col B -> cache_B

(Set a different `path` in each phase's dataset toml.) Only share one cache across phases if
you have verified both columns have identical survivor sets.

SCOPE
-----
Supports a single parquet/huggingface `[[directory]]` with a `path` (cache root) -- the shape
used by big single-source datasets this optimization targets. Multiple directories or a
missing `path` raise a clear error.
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

# repo root on sys.path so `utils.*` imports work when run as `python cache_multigpu.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toml


# --------------------------------------------------------------------------- planning --
def plan_jobs(shard_paths, num_gpus, shards_per_job=1):
    """Split shard_paths into jobs of `shards_per_job` shards, scheduled in waves of
    `num_gpus`. Returns a list of dicts: {idx, gpu, wave, shards}. Pure/deterministic
    so it is unit-testable without any GPU."""
    jobs = []
    groups = [shard_paths[i:i + shards_per_job] for i in range(0, len(shard_paths), shards_per_job)]
    for idx, shards in enumerate(groups):
        jobs.append({'idx': idx, 'gpu': idx % num_gpus, 'wave': idx // num_gpus, 'shards': shards})
    return jobs


# --------------------------------------------------------------------------- config io --
def _load_configs(train_config_path):
    """Load the train toml and its referenced dataset toml. Returns
    (train_cfg, dataset_cfg, dataset_path)."""
    train_cfg = toml.load(train_config_path)
    ds_path = train_cfg.get('dataset')
    if not ds_path:
        raise ValueError(f"train config {train_config_path!r} has no 'dataset' key")
    if not os.path.isabs(ds_path):
        # dataset path is resolved relative to CWD by train.py; mirror that.
        ds_path = os.path.abspath(ds_path)
    dataset_cfg = toml.load(ds_path)
    return train_cfg, dataset_cfg, ds_path


def _single_parquet_directory(dataset_cfg):
    """Return (dir_index, directory_config) for the one parquet/hf directory, or raise."""
    dirs = dataset_cfg.get('directory', [])
    if isinstance(dirs, dict):
        dirs = [dirs]
    pq = [(i, d) for i, d in enumerate(dirs) if d.get('type') in ('parquet', 'huggingface')]
    if len(pq) != 1:
        raise ValueError(
            f"multi-GPU caching supports exactly one parquet/huggingface [[directory]]; "
            f"found {len(pq)} (of {len(dirs)} total). Cache such configs with the normal "
            f"single-GPU `train.py --cache_only`.")
    i, d = pq[0]
    if not d.get('path'):
        raise ValueError(
            "the parquet [[directory]] must set 'path' (the cache root) so the merged cache "
            "has a home. Add path='/some/cache_root' to the dataset toml.")
    return i, d


def _write_job_configs(train_cfg, dataset_cfg, dir_index, shards, job_cache_dir,
                       job_dataset_path, job_train_path, job_out_dir):
    """Write a per-job dataset+train toml that is a COPY of the originals with the target
    parquet directory pinned to `shards` (as type='parquet') and its cache root pointed at
    `job_cache_dir`. All other settings (resolutions, AR buckets, caption config, model,
    adapter, optimizer) are preserved verbatim so the cached latents match the real run."""
    ds = json.loads(json.dumps(dataset_cfg))  # deep copy
    dirs = ds.get('directory', [])
    if isinstance(dirs, dict):
        dirs = [dirs]
        ds['directory'] = dirs
    d = dirs[dir_index]
    # Force a plain parquet source over the resolved local shard paths (works whether the
    # original was 'parquet' or 'huggingface'). Drop hf-only keys that would re-resolve.
    d['type'] = 'parquet'
    d['parquet_files'] = list(shards)
    d['path'] = job_cache_dir
    # Ensure each decode worker can hold ALL of this job's shards (lru >= num_shards) so the
    # AR-bucketed iteration order never thrash-evicts -> the decode-pool deadlock. Matches the
    # proven reference (run_cache_4gpu.py: lru = max(2, num_shards+1)).
    d['parquet_shard_lru'] = max(int(d.get('parquet_shard_lru') or 0), len(shards) + 1, 2)
    for k in ('dataset', 'config', 'split', 'data_files', 'parquet_path'):
        d.pop(k, None)
    with open(job_dataset_path, 'w') as f:
        toml.dump(ds, f)

    tr = json.loads(json.dumps(train_cfg))  # deep copy
    tr['dataset'] = job_dataset_path
    tr['output_dir'] = job_out_dir
    with open(job_train_path, 'w') as f:
        toml.dump(tr, f)


# --------------------------------------------------------------------------- launching --
# crash signatures: a job printing one of these has failed. Under N-way concurrency the decode
# pipe can break (BrokenPipe) or a worker can OOM and the process then HANGS on teardown holding
# GPU memory -- so a plain p.wait() would block the whole wave forever. We poll the log and killpg.
_CRASH_SIGS = ('BrokenPipeError', 'Traceback (most recent call last)', 'CUDA out of memory',
               'torch.cuda.OutOfMemoryError', 'exits with return code', 'Killed')


def _killpg(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _tail(path, n=6000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode('utf-8', 'replace')
    except Exception:
        return ''


def _wait_job(p, log_path, hard_timeout=2400):
    """Wait for a job, returning rc. Kills the process group and returns non-zero on a crash
    signature or hard_timeout; returns 0 even if the launcher hangs on teardown AFTER the worker
    printed success (the cache is already flushed at that point)."""
    start = time.time()
    worker_done = None
    while True:
        rc = p.poll()
        if rc is not None:
            return rc
        t = _tail(log_path)
        if any(s in t for s in _CRASH_SIGS):
            _killpg(p)
            return 1
        if worker_done is None and 'exits successfully' in t:
            worker_done = time.time()
        if worker_done is not None and time.time() - worker_done > 25:
            _killpg(p)   # launcher/NCCL teardown hang after a successful worker
            return 0
        if time.time() - start > hard_timeout:
            _killpg(p)
            return -1
        time.sleep(3)


def _job_latents(cache_dir):
    """Total latents actually cached under a job cache dir (0 if it crashed/produced nothing)."""
    tot = 0
    for ip in glob.glob(os.path.join(cache_dir, '**', 'latents', '_index.json'), recursive=True):
        try:
            tot += json.load(open(ip)).get('total', 0)
        except Exception:
            pass
    return tot


def run_waves(jobs, train_config_builder, num_gpus, base_port, log_dir, max_retry=2):
    """Run jobs NGPU-at-a-time, ROBUST to the intermittent decode-pipe crashes / teardown hangs
    seen under N-way concurrency: each job is verified to have cached latents, failed jobs are
    RETRIED (fresh job cache), and the failed set is returned so the caller can ABORT rather than
    merge a silently incomplete cache. GPU pinning uses `deepspeed --include localhost:<gpu>` with
    a distinct MASTER_PORT per GPU in the env (train.py only auto-picks a port when MASTER_PORT is
    absent). Returns (rcs dict, failed list)."""
    nwaves = max((j['wave'] for j in jobs), default=-1) + 1
    by_idx = {j['idx']: j for j in jobs}
    rcs = {}
    for w in range(nwaves):
        pending = [j['idx'] for j in jobs if j['wave'] == w]
        for attempt in range(max_retry + 1):
            if not pending:
                break
            procs = []
            for idx in pending:
                job = by_idx[idx]
                tr = train_config_builder(job)  # (re)writes the job config + assigns job['cache_dir']
                shutil.rmtree(job['cache_dir'], ignore_errors=True)  # fresh (no partial index on retry)
                g = job['gpu']
                port = base_port + g
                env = dict(os.environ, MASTER_PORT=str(port), NCCL_P2P_DISABLE='1',
                           NCCL_IB_DISABLE='1', PYTHONUNBUFFERED='1', TORCHINDUCTOR_COMPILE_THREADS='1')
                env.pop('CUDA_VISIBLE_DEVICES', None)  # --include does the GPU pinning
                log_path = os.path.join(log_dir, f"cache_job{idx:04d}.log")
                cmd = (f"deepspeed --include localhost:{g} --master_port {port} train.py --deepspeed "
                       f"--config {tr} --cache_only --master_port {port}")
                # start_new_session=True -> child is a process-group leader we can killpg if it hangs
                p = subprocess.Popen(cmd, shell=True, env=env, stdout=open(log_path, 'w'),
                                     stderr=subprocess.STDOUT, start_new_session=True)
                procs.append((idx, g, p, log_path))
                print(f"[mgpu] wave {w} attempt {attempt}: job {idx} -> GPU {g} "
                      f"({len(by_idx[idx]['shards'])} shard(s))", flush=True)
                time.sleep(2)  # stagger so N model loads don't hit the same instant
            still = []
            for idx, g, p, log_path in procs:
                rc = _wait_job(p, log_path)
                lat = _job_latents(by_idx[idx]['cache_dir'])
                ok = (rc == 0 and lat > 0)
                rcs[idx] = rc
                print(f"[mgpu] wave {w} attempt {attempt}: job {idx} (GPU {g}) rc={rc} "
                      f"latents={lat} ok={ok}", flush=True)
                if not ok:
                    still.append(idx)
            pending = still
            if pending and attempt < max_retry:
                print(f"[mgpu] wave {w}: retrying {pending} (freeing GPUs first)", flush=True)
                time.sleep(10)  # let killed processes release GPU memory before retry
        if pending:
            return rcs, pending
    return rcs, []


# ----------------------------------------------------------------------------- merging --
def _bucket_latents_dirs(cache_root):
    """bucket-relative-path -> abs latents dir, for every 'latents' dir with an _index.json."""
    out = {}
    for d in glob.glob(os.path.join(cache_root, '**', 'latents'), recursive=True):
        if os.path.isfile(os.path.join(d, '_index.json')):
            out[os.path.relpath(d, cache_root)] = d
    return out


def merge_caches(job_cache_dirs, target_cache_root, fingerprint='merged_multigpu'):
    """Concatenate every job cache's per-size-bucket latents shards into target_cache_root,
    rebuilding each bucket's _index.json. Pure file ops (glob/json/shutil) so it is unit
    testable without a GPU. Returns total merged latent rows."""
    rels = set()
    for jd in job_cache_dirs:
        rels |= set(_bucket_latents_dirs(jd).keys())
    total = 0
    for rel in sorted(rels):
        dst = os.path.join(target_cache_root, rel)
        os.makedirs(dst, exist_ok=True)
        # Clear any stale shards at the target so a re-run is idempotent (never leaves orphan
        # shard_*.parquet from a previous, larger merge that the new _index.json wouldn't list).
        for old in glob.glob(os.path.join(dst, 'shard_*.parquet')):
            os.remove(old)
        shards, n = [], 0
        for jd in job_cache_dirs:
            idx_path = os.path.join(jd, rel, '_index.json')
            if not os.path.isfile(idx_path):
                continue
            for s in json.load(open(idx_path)).get('shards', []):
                sf = os.path.join(jd, rel, s['file'])
                if not os.path.isfile(sf):
                    continue
                newname = f'shard_{n:05d}.parquet'
                shutil.copyfile(sf, os.path.join(dst, newname))
                shards.append({'file': newname, 'rows': s['rows'], 'uploaded': False})
                n += 1
        rows = sum(s['rows'] for s in shards)
        total += rows
        # Placeholder fingerprint: the training run reads with --trust_cache (image_spec-keyed),
        # so the fingerprint value is not used for correctness.
        with open(os.path.join(dst, '_index.json'), 'w') as f:
            json.dump({'fingerprint': fingerprint, 'shards': shards, 'total': rows, 'next_shard': n}, f)
        print(f"[mgpu]   {rel}: {n} shard(s), {rows} latents", flush=True)
    return total


# -------------------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description='Data-parallel latents caching across N GPUs.')
    ap.add_argument('--config', required=True, help='training toml (same file you train with)')
    ap.add_argument('--num_gpus', type=int, required=True)
    ap.add_argument('--shards_per_job', type=int, default=1,
                    help='shards per single-GPU job (default 1 = proven safe; only raise with '
                         'parquet_shard_lru >= this, or the decode pool deadlocks)')
    ap.add_argument('--master_port', type=int, default=29500, help='base port; job on GPU g uses base+g')
    ap.add_argument('--work_dir', default=None, help='scratch dir for per-job caches/configs '
                    '(default: <cache_root>/.mgpu_work)')
    ap.add_argument('--keep_jobs', action='store_true', help='keep per-job caches after merge (debug)')
    args = ap.parse_args()

    args.config = os.path.abspath(args.config)  # resolve before chdir (may be relative to user CWD)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from repo root (train.py lives here)
    train_cfg, dataset_cfg, ds_path = _load_configs(args.config)
    dir_index, directory = _single_parquet_directory(dataset_cfg)
    cache_root = directory['path']

    _all_dirs = dataset_cfg.get('directory', [])
    if isinstance(_all_dirs, dict):
        _all_dirs = [_all_dirs]
    _other = [d for d in _all_dirs if d.get('type') not in ('parquet', 'huggingface')]
    if _other:
        print(f"[mgpu] NOTE: {len(_other)} non-parquet [[directory]] ent(y/ies) are NOT pre-cached by "
              f"this tool; the normal --trust_cache training run will cache them single-GPU.", flush=True)

    from utils.parquet_source import resolve_parquet_source
    shards = resolve_parquet_source(directory).shard_paths
    if not shards:
        raise RuntimeError('resolve_parquet_source returned no shards')
    caption_col = directory.get('caption_column', 'caption')
    print(f"[mgpu] {len(shards)} source shard(s); caption_column={caption_col!r}; cache_root={cache_root}",
          flush=True)

    work_dir = args.work_dir or os.path.join(cache_root, '.mgpu_work')
    # Fresh work dir every run so a re-run never merges stale/other-config shards.
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    jobs = plan_jobs(shards, args.num_gpus, args.shards_per_job)
    nwaves = max((j['wave'] for j in jobs), default=-1) + 1
    print(f"[mgpu] {len(jobs)} job(s) of {args.shards_per_job} shard(s) across {args.num_gpus} "
          f"GPU(s) -> {nwaves} wave(s)", flush=True)

    def build_job_config(job):
        """Assign this job's per-job paths, write its dataset+train toml, return the train path."""
        job['cache_dir'] = os.path.join(work_dir, f"cache_j{job['idx']:04d}")
        jd = os.path.join(work_dir, f"ds_j{job['idx']:04d}.toml")
        jt = os.path.join(work_dir, f"tr_j{job['idx']:04d}.toml")
        jo = os.path.join(work_dir, f"out_j{job['idx']:04d}")
        _write_job_configs(train_cfg, dataset_cfg, dir_index, job['shards'], job['cache_dir'], jd, jt, jo)
        return jt

    t0 = time.time()
    rcs, failed = run_waves(jobs, build_job_config, args.num_gpus, args.master_port, work_dir)
    if failed:
        # ABORT rather than merge a silently incomplete cache (which would crash training hours in).
        print(f"[mgpu] ABORT: {len(failed)} job(s) could not cache latents after retries: "
              f"{sorted(failed)}.\n       NOT merging -- a partial cache would crash the --trust_cache "
              f"run later. Per-job logs are under {work_dir} (cache_job####.log). Common cause is OOM: "
              f"lower map_num_proc or --num_gpus and re-run.", flush=True)
        sys.exit(1)

    # Clear any prior diffusion-pipe cache under this root before merging, so a previously-populated
    # cache_root (from an earlier training run, or a different caption column) can't leak stale
    # metadata / grouped_metadata / text-embeddings into the --trust_cache run. Mirrors the proven
    # reference's rmtree(CACHE_ROOT); only the 'cache/' subtree (diffusion-pipe's namespace) is removed.
    stale = os.path.join(cache_root, 'cache')
    if os.path.isdir(stale):
        print(f"[mgpu] clearing prior cache subtree: {stale}", flush=True)
        shutil.rmtree(stale, ignore_errors=True)

    print('[mgpu] merging per-job caches ...', flush=True)
    total = merge_caches([j['cache_dir'] for j in jobs], cache_root)
    print(f"[mgpu] MERGE DONE: {total} latents -> {cache_root}  (wall {time.time()-t0:.0f}s)", flush=True)

    if not args.keep_jobs:
        shutil.rmtree(work_dir, ignore_errors=True)  # per-job caches, configs, out dirs, logs

    print('\n[mgpu] Cache built. Train with --trust_cache so the merged cache is adopted:\n'
          f"    deepspeed --num_gpus={args.num_gpus} train.py --deepspeed --config {args.config} --trust_cache\n"
          'Reminder: for a 2-phase before/after run with a different caption_column, cache each '
          'phase into its OWN cache root (see the per-phase note at the top of this file).')


if __name__ == '__main__':
    main()
