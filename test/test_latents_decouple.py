"""Unit tests for the caption-independent latents cache + iteration-order repeats.

Validates the core behavior of utils/dataset.py SizeBucketDataset.cache_latents
without importing the full GPU stack (replicates the exact identity-fingerprint and
iteration-order-build logic on HF datasets.Dataset objects):
- the latents identity fingerprint is SHARED across caption sets + repeat manifests,
- the iteration_order key (full metadata fingerprint) DIFFERS so they don't clobber,
- per-row num_repeats yields that many iteration-order entries pointing at the same
  latents_idx / caption_number.

Needs: datasets (HF). Runs in the CPU testvenv:  python test/test_latents_decouple.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import datasets
from datasets.fingerprint import Hasher


# Mirror SizeBucketDataset._latents_identity_fingerprint exactly.
def _identity_fp(ds):
    drop = {'caption', 'num_repeats'}
    cols = [c for c in ds.column_names if c not in drop]
    return Hasher.hash(ds.select_columns(cols).to_dict())


def _meta(captions, repeats):
    n = len(captions)
    return datasets.Dataset.from_dict({
        'image_spec': [['\x00parquet\x00/s.parquet\x00col\x00image', str(i)] for i in range(n)],
        'mask_file': [None] * n,
        'size_bucket': [[1024, 1024, 1]] * n,
        'is_video': [False] * n,
        'caption': captions,
        'num_repeats': repeats,
    })


def test_identity_fingerprint_caption_and_repeat_independent():
    # Same images, DIFFERENT captions (vlm vs animetimm) and DIFFERENT repeats.
    d_vlm = _meta([['a woman'], ['a truck'], ['a car']], [1, 1, 1])
    d_anime = _meta([['1girl'], ['vehicle'], ['automobile']], [3, 1, 8])
    # latents cache key must be IDENTICAL -> encode once, reuse across both.
    assert _identity_fp(d_vlm) == _identity_fp(d_anime), 'identity fingerprint must be caption/repeat-independent'
    # iteration_order key (full fingerprint) must DIFFER -> coexist, no clobber.
    assert d_vlm._fingerprint != d_anime._fingerprint, 'io_key must differ across caption/repeat sets'
    # And changing ONLY repeats (same captions) still changes io_key but not identity.
    d_r1 = _meta([['x'], ['y']], [1, 1])
    d_r2 = _meta([['x'], ['y']], [4, 2])
    assert _identity_fp(d_r1) == _identity_fp(d_r2)
    assert d_r1._fingerprint != d_r2._fingerprint
    print('test_identity_fingerprint_caption_and_repeat_independent OK')


def _build_iteration_order(ds):
    """Replicates cache_latents' equal-captions iteration-order build with repeats."""
    image_spec_to_latents_idx = {tuple(s): i for i, s in enumerate(ds['image_spec'])}
    has_repeats = 'num_repeats' in ds.column_names
    # equal-captions check
    num_captions = None
    for ex in ds.select_columns(['caption']):
        n = len(ex['caption'])
        assert num_captions is None or n == num_captions
        num_captions = n
    sel = ['image_spec', 'caption'] + (['num_repeats'] if has_repeats else [])
    iob = [[] for _ in range(num_captions)]
    seed = 0
    for ex in ds.select_columns(sel):
        image_spec = ex['image_spec']
        captions = list(ex['caption'])
        rep = int(ex['num_repeats']) if has_repeats else 1
        seed += 1
        latents_idx = image_spec_to_latents_idx[tuple(image_spec)]
        for i, caption in enumerate(captions):
            entry = (tuple(image_spec), latents_idx, caption, i)
            iob[i].extend([entry] * rep)
    order = []
    for l in iob:
        order.extend(l)
    return order


def test_iteration_order_repeats():
    # 3 images, 1 caption each, repeats 1/3/5
    ds = _meta([['c0'], ['c1'], ['c2']], [1, 3, 5])
    order = _build_iteration_order(ds)
    assert len(order) == 1 + 3 + 5, len(order)         # entry count = sum(repeats)
    by_idx = Counter(e[1] for e in order)               # latents_idx histogram
    assert by_idx[0] == 1 and by_idx[1] == 3 and by_idx[2] == 5
    # every replicated entry reuses the SAME caption_number (-> same TE row)
    for e in order:
        assert e[3] == 0
    print('test_iteration_order_repeats OK')


def test_multi_caption_repeats():
    # 2 images, 2 captions each, repeats 2 and 3 -> 2*2 + 2*3 = 10 entries
    ds = _meta([['a', 'b'], ['c', 'd']], [2, 3])
    order = _build_iteration_order(ds)
    assert len(order) == (2 * 2) + (2 * 3), len(order)
    # latents_idx 0 appears 2(captions)*2(rep)=4, idx1 = 2*3=6
    by_idx = Counter(e[1] for e in order)
    assert by_idx[0] == 4 and by_idx[1] == 6
    print('test_multi_caption_repeats OK')


def test_no_repeats_column_defaults_to_one():
    ds = datasets.Dataset.from_dict({
        'image_spec': [['p', '0'], ['p', '1']],
        'caption': [['a'], ['b']],
    })
    order = _build_iteration_order(ds)
    assert len(order) == 2
    print('test_no_repeats_column_defaults_to_one OK')


ALL = [test_identity_fingerprint_caption_and_repeat_independent, test_iteration_order_repeats,
       test_multi_caption_repeats, test_no_repeats_column_defaults_to_one]

if __name__ == '__main__':
    for t in ALL:
        t()
    print(f'\nAll {len(ALL)} latents-decouple tests passed.')
