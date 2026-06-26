"""Unit tests for utils.parquet_cache.ParquetCache.

Runnable standalone (no pytest required):
    python test/test_parquet_cache.py
or with pytest:
    pytest test/test_parquet_cache.py

Only depends on torch + pyarrow + numpy (not the full diffusion-pipe GPU stack).
"""

import os
import sys
import json
import shutil
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parquet_cache import ParquetCache, SHARD_FMT, INDEX_FILE


def _mkdir():
    return tempfile.mkdtemp(prefix='pqcache_test_')


def _sample_item(i, with_mask=True):
    return {
        # bf16 latent-like tensor (the Anima case numpy can't represent)
        'latents': torch.randn(16, 1, 8, 8, dtype=torch.bfloat16),
        # optional mask: None on some rows (nullable column)
        'mask': (torch.rand(8, 8, dtype=torch.float16) if with_mask else None),
        # variable-length-ish int tensors
        'attn_mask': torch.randint(0, 2, (5 + (i % 3),), dtype=torch.int64),
        'pooled': torch.randn(32, dtype=torch.float32),
        'flag': torch.ones(4, dtype=torch.bool),
        # non-tensor tuple with a None element (legacy loose-file image_spec shape)
        'image_spec': (None, f'/data/img_{i}.png'),
        'caption': f'a photo of object {i}',
        'caption_number': i % 4,
    }


def _assert_item_equal(got, ref, i):
    assert torch.equal(got['latents'], ref['latents']), f'latents mismatch at {i}'
    assert got['latents'].dtype == torch.bfloat16, f'latents dtype at {i}'
    if ref['mask'] is None:
        assert got['mask'] is None, f'expected None mask at {i}, got {type(got["mask"])}'
    else:
        assert torch.equal(got['mask'], ref['mask']), f'mask mismatch at {i}'
        assert got['mask'].dtype == torch.float16
    assert torch.equal(got['attn_mask'], ref['attn_mask']), f'attn_mask mismatch at {i}'
    assert got['attn_mask'].dtype == torch.int64
    assert got['attn_mask'].shape == ref['attn_mask'].shape, f'variable shape at {i}'
    assert torch.equal(got['pooled'], ref['pooled']), f'pooled mismatch at {i}'
    assert torch.equal(got['flag'], ref['flag']) and got['flag'].dtype == torch.bool
    assert tuple(got['image_spec']) == tuple(ref['image_spec']), f'image_spec at {i}: {got["image_spec"]}'
    assert got['caption'] == ref['caption']
    assert got['caption_number'] == ref['caption_number']


def test_roundtrip_dtypes_and_nullable():
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fp1', shard_size_mb=350, row_group_size=8)
        items = [_sample_item(i, with_mask=(i % 2 == 0)) for i in range(20)]
        for it in items:
            cache.add(it)
        cache.finalize_current_shard()
        assert len(cache) == 20

        # reopen fresh (simulates the train-time read path on every rank)
        rd = ParquetCache(d, 'fp1', row_group_size=8)
        assert len(rd) == 20
        for i in range(20):
            _assert_item_equal(rd[i], items[i], i)

        # reconstructed tensors must be owning + writable (matches torch.load semantics)
        t = rd[0]['latents']
        t.add_(1)  # would raise if read-only
        print('test_roundtrip_dtypes_and_nullable OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_multishard_flush():
    d = _mkdir()
    try:
        # tiny shard size forces many shards; each latent ~ 16*8*8*2 = 2048 bytes
        cache = ParquetCache(d, 'fp2', shard_size_mb=0.01, row_group_size=4)
        items = [_sample_item(i) for i in range(30)]
        for it in items:
            cache.add(it)
        cache.finalize_current_shard()
        n_shards = len(list(os.scandir(d)))  # incl index, but count parquet
        n_parquet = len([p for p in os.listdir(d) if p.endswith('.parquet')])
        assert n_parquet >= 2, f'expected multiple shards, got {n_parquet}'
        assert len(cache) == 30
        rd = ParquetCache(d, 'fp2', row_group_size=4)
        assert len(rd) == 30
        for i in range(30):
            _assert_item_equal(rd[i], items[i], i)
        print(f'test_multishard_flush OK ({n_parquet} shards)')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_continuation():
    d = _mkdir()
    try:
        # write first 12 across a couple shards, finalize
        cache = ParquetCache(d, 'fp3', shard_size_mb=0.01, row_group_size=4)
        items = [_sample_item(i) for i in range(30)]
        for it in items[:12]:
            cache.add(it)
        cache.finalize_current_shard()
        first_len = len(cache)
        assert first_len == 12

        # reopen with same fingerprint -> length preserved (resume point)
        cache2 = ParquetCache(d, 'fp3', shard_size_mb=0.01, row_group_size=4)
        assert len(cache2) == 12, 'resume should see exactly the flushed rows'
        # continue from where we left off (this mirrors _map_and_cache select(range(len, N)))
        for it in items[12:]:
            cache2.add(it)
        cache2.finalize_current_shard()
        assert len(cache2) == 30

        rd = ParquetCache(d, 'fp3', row_group_size=4)
        assert len(rd) == 30
        for i in range(30):
            _assert_item_equal(rd[i], items[i], i)
        print('test_resume_continuation OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_crash_midbuffer_loses_only_unflushed():
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fp4', shard_size_mb=0.01, row_group_size=4)
        items = [_sample_item(i) for i in range(30)]
        # add 20 but only enough to flush some shards; leave a tail buffered, do NOT finalize
        for it in items[:20]:
            cache.add(it)
        # simulate crash: drop the object without finalize
        flushed = len(cache)
        del cache
        # reopen: length == only durably-flushed rows; no corruption
        rd = ParquetCache(d, 'fp4', row_group_size=4)
        assert len(rd) == flushed, (len(rd), flushed)
        assert len(rd) <= 20
        for i in range(len(rd)):
            _assert_item_equal(rd[i], items[i], i)
        print(f'test_crash_midbuffer_loses_only_unflushed OK (flushed={flushed})')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_orphan_pruning():
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fp5', shard_size_mb=0.01, row_group_size=4)
        for it in (_sample_item(i) for i in range(12)):
            cache.add(it)
        cache.finalize_current_shard()
        with open(os.path.join(d, INDEX_FILE)) as f:
            next_shard = json.load(f)['next_shard']
        # simulate a crash that wrote a shard file but never committed the index
        orphan = os.path.join(d, SHARD_FMT.format(next_shard))
        shutil.copyfile(os.path.join(d, SHARD_FMT.format(0)), orphan)
        assert os.path.exists(orphan)
        # reopening must prune the orphan (id >= next_shard, not in index)
        rd = ParquetCache(d, 'fp5', row_group_size=4)
        assert not os.path.exists(orphan), 'orphan shard should have been pruned'
        assert len(rd) == 12
        print('test_orphan_pruning OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fingerprint_change_clears():
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fpA', shard_size_mb=350, row_group_size=4)
        for it in (_sample_item(i) for i in range(8)):
            cache.add(it)
        cache.finalize_current_shard()
        assert len(cache) == 8
        # new fingerprint -> existing cache invalidated and cleared
        rd = ParquetCache(d, 'fpB', row_group_size=4)
        assert len(rd) == 0
        assert len([p for p in os.listdir(d) if p.endswith('.parquet')]) == 0
        print('test_fingerprint_change_clears OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_none_mask_column():
    # The common parquet path (e.g. deepfashion) has mask=None for EVERY row, so a
    # whole column (+ its __shape/__dtype side columns) is all-null. Must round-trip.
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fpNone', shard_size_mb=0.01, row_group_size=4)
        items = [_sample_item(i, with_mask=False) for i in range(10)]
        for it in items:
            cache.add(it)
        cache.finalize_current_shard()
        rd = ParquetCache(d, 'fpNone', row_group_size=4)
        assert len(rd) == 10
        for i in range(10):
            got = rd[i]
            assert got['mask'] is None, i
            assert torch.equal(got['latents'], items[i]['latents'])
            assert got['caption_number'] == items[i]['caption_number']
        print('test_all_none_mask_column OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_empty_finalize_noop():
    d = _mkdir()
    try:
        cache = ParquetCache(d, 'fpE', row_group_size=4)
        cache.finalize_current_shard()  # no rows added
        assert len(cache) == 0
        assert len([p for p in os.listdir(d) if p.endswith('.parquet')]) == 0
        print('test_empty_finalize_noop OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


ALL_TESTS = [
    test_roundtrip_dtypes_and_nullable,
    test_multishard_flush,
    test_resume_continuation,
    test_crash_midbuffer_loses_only_unflushed,
    test_orphan_pruning,
    test_fingerprint_change_clears,
    test_all_none_mask_column,
    test_empty_finalize_noop,
]


if __name__ == '__main__':
    for t in ALL_TESTS:
        t()
    print(f'\nAll {len(ALL_TESTS)} ParquetCache tests passed.')
