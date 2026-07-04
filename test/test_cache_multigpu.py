"""Unit tests for cache_multigpu (the data-parallel latents cache driver).

Runnable standalone (no pytest, no GPU, no torch):
    python test/test_cache_multigpu.py

Covers the pure/importable logic: job planning, the per-job config copy-override, and the
merge (completeness, shard renumbering, cross-bucket isolation). The GPU launching
(run_waves) is not unit-tested here -- it is exercised by real multi-GPU cache runs.
"""
import os
import sys
import glob
import json
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cache_multigpu as m


def test_plan_jobs():
    # 5 shards, 4 gpus, 1 shard/job -> 5 jobs over 2 waves, round-robin GPUs
    jobs = m.plan_jobs([f's{i}' for i in range(5)], num_gpus=4, shards_per_job=1)
    assert len(jobs) == 5
    assert [j['gpu'] for j in jobs] == [0, 1, 2, 3, 0]
    assert [j['wave'] for j in jobs] == [0, 0, 0, 0, 1]
    assert [j['shards'] for j in jobs] == [['s0'], ['s1'], ['s2'], ['s3'], ['s4']]
    # every shard assigned exactly once (no drop, no dup)
    allshards = [s for j in jobs for s in j['shards']]
    assert sorted(allshards) == [f's{i}' for i in range(5)]

    # 6 shards, 2 gpus, 2 shards/job -> 3 jobs over 2 waves
    jobs = m.plan_jobs([f's{i}' for i in range(6)], num_gpus=2, shards_per_job=2)
    assert [j['shards'] for j in jobs] == [['s0', 's1'], ['s2', 's3'], ['s4', 's5']]
    assert [j['gpu'] for j in jobs] == [0, 1, 0]
    assert [j['wave'] for j in jobs] == [0, 0, 1]
    print('test_plan_jobs OK')


def _make_fake_job_cache(root, bucket_rel, rows_per_shard):
    """Create a fake per-job cache: <root>/<bucket_rel>/latents/{shard_*.parquet,_index.json}."""
    d = os.path.join(root, bucket_rel, 'latents')
    os.makedirs(d)
    shards = []
    for i, rows in enumerate(rows_per_shard):
        fn = f'shard_{i:05d}.parquet'
        with open(os.path.join(d, fn), 'w') as f:
            f.write(f'fake-{root}-{bucket_rel}-{i}')  # unique content so a bad copy is detectable
        shards.append({'file': fn, 'rows': rows, 'uploaded': False})
    json.dump({'fingerprint': 'x', 'shards': shards, 'total': sum(rows_per_shard),
               'next_shard': len(shards)}, open(os.path.join(d, '_index.json'), 'w'))


def test_merge_caches():
    tmp = tempfile.mkdtemp(prefix='mgpu_merge_')
    try:
        b = 'cache/anima/cache_1.0_1024_1024'
        j0 = os.path.join(tmp, 'cache_j0000')
        j1 = os.path.join(tmp, 'cache_j0001')
        _make_fake_job_cache(j0, b, [3, 2])   # 5 rows across 2 shards
        _make_fake_job_cache(j1, b, [4])      # 4 rows across 1 shard
        target = os.path.join(tmp, 'cache_root')
        total = m.merge_caches([j0, j1], target)
        assert total == 9, total
        mdir = os.path.join(target, b, 'latents')
        idx = json.load(open(os.path.join(mdir, '_index.json')))
        assert idx['total'] == 9 and idx['next_shard'] == 3
        # shards renumbered contiguously, no collision, order j0-then-j1
        assert [s['file'] for s in idx['shards']] == [
            'shard_00000.parquet', 'shard_00001.parquet', 'shard_00002.parquet']
        assert [s['rows'] for s in idx['shards']] == [3, 2, 4]
        for s in idx['shards']:
            assert os.path.isfile(os.path.join(mdir, s['file']))
        print('test_merge_caches OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_two_buckets_no_collision():
    """Different size buckets each restart shard numbering at 0 but must not collide."""
    tmp = tempfile.mkdtemp(prefix='mgpu_merge2_')
    try:
        j0 = os.path.join(tmp, 'cache_j0000')
        _make_fake_job_cache(j0, 'cache/m/cache_A', [2])
        _make_fake_job_cache(j0, 'cache/m/cache_B', [5])
        target = os.path.join(tmp, 'root')
        assert m.merge_caches([j0], target) == 7
        for bk, rows in [('cache/m/cache_A', 2), ('cache/m/cache_B', 5)]:
            idx = json.load(open(os.path.join(target, bk, 'latents', '_index.json')))
            assert idx['total'] == rows
        print('test_merge_two_buckets_no_collision OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_skips_missing_job():
    """A crashed job with no _index.json is skipped, not fatal (its rows just aren't merged)."""
    tmp = tempfile.mkdtemp(prefix='mgpu_merge3_')
    try:
        b = 'cache/m/cache_A'
        j0 = os.path.join(tmp, 'cache_j0000')
        j1 = os.path.join(tmp, 'cache_j0001')  # will be empty (simulated crash)
        _make_fake_job_cache(j0, b, [3])
        os.makedirs(os.path.join(j1, b, 'latents'))  # dir exists but NO _index.json
        assert m.merge_caches([j0, j1], os.path.join(tmp, 'root')) == 3
        print('test_merge_skips_missing_job OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_idempotent_rerun():
    """Re-merging into the same target with FEWER shards must not leave orphan shard files."""
    tmp = tempfile.mkdtemp(prefix='mgpu_rerun_')
    try:
        b = 'cache/m/cache_A'
        target = os.path.join(tmp, 'root')
        big = os.path.join(tmp, 'big')
        _make_fake_job_cache(big, b, [1, 1, 1])          # first run: 3 shards
        assert m.merge_caches([big], target) == 3
        mdir = os.path.join(target, b, 'latents')
        assert len(glob.glob(os.path.join(mdir, 'shard_*.parquet'))) == 3
        small = os.path.join(tmp, 'small')
        _make_fake_job_cache(small, b, [2])              # re-run: only 1 shard
        assert m.merge_caches([small], target) == 2
        files = sorted(os.path.basename(x) for x in glob.glob(os.path.join(mdir, 'shard_*.parquet')))
        assert files == ['shard_00000.parquet'], files   # no orphans from the 3-shard run
        idx = json.load(open(os.path.join(mdir, '_index.json')))
        assert idx['total'] == 2 and len(idx['shards']) == 1
        print('test_merge_idempotent_rerun OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_job_configs():
    tmp = tempfile.mkdtemp(prefix='mgpu_cfg_')
    try:
        import toml
        dataset_cfg = {
            'resolutions': [1024], 'enable_ar_bucket': True, 'min_ar': 0.5, 'max_ar': 2.0,
            'cache_backend': 'parquet',
            'directory': [{'type': 'huggingface', 'dataset': 'user/ds', 'config': 'c1',
                           'split': 'train', 'path': '/real/cache_root', 'image_column': 'image',
                           'caption_column': 'cap_a', 'caption_type': 'json', 'num_repeats': 1,
                           'skip_empty_caption': True}],
        }
        train_cfg = {'output_dir': '/real/out', 'dataset': '/real/ds.toml', 'epochs': 2,
                     'model': {'type': 'anima', 'dtype': 'bfloat16'},
                     'adapter': {'type': 'lora', 'rank': 64}}
        jc = os.path.join(tmp, 'cj')
        jd = os.path.join(tmp, 'ds.toml')
        jt = os.path.join(tmp, 'tr.toml')
        jo = os.path.join(tmp, 'out')
        m._write_job_configs(train_cfg, dataset_cfg, 0, ['/abs/shardX.parquet'], jc, jd, jt, jo)
        ds = toml.load(jd)
        tr = toml.load(jt)
        d = ds['directory'][0]
        # target directory pinned to the one local shard, as plain parquet, into the job cache
        assert d['type'] == 'parquet'
        assert d['parquet_files'] == ['/abs/shardX.parquet']
        assert d['path'] == jc
        # hf-only re-resolution keys dropped
        assert 'dataset' not in d and 'config' not in d and 'split' not in d
        # all other column/config settings preserved verbatim (so cached latents match the real run)
        assert d['caption_column'] == 'cap_a' and d['caption_type'] == 'json'
        assert d['skip_empty_caption'] is True
        # parquet_shard_lru bumped to >= num_shards+1 (deadlock guard); 1 shard -> 2
        assert d['parquet_shard_lru'] == 2, d.get('parquet_shard_lru')
        assert ds['resolutions'] == [1024] and ds['enable_ar_bucket'] is True
        # train cfg: dataset + output_dir overridden, everything else preserved
        assert tr['dataset'] == jd and tr['output_dir'] == jo
        assert tr['model']['type'] == 'anima' and tr['adapter']['rank'] == 64 and tr['epochs'] == 2
        # originals must not be mutated (deep copy)
        assert dataset_cfg['directory'][0]['type'] == 'huggingface'
        assert train_cfg['output_dir'] == '/real/out'
        print('test_write_job_configs OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_plan_jobs()
    test_merge_caches()
    test_merge_two_buckets_no_collision()
    test_merge_skips_missing_job()
    test_merge_idempotent_rerun()
    test_write_job_configs()
    print('ALL cache_multigpu tests passed')
