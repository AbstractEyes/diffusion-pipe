"""Unit tests for utils.bucket_manifest + the key-column wiring in parquet_source.

Runnable standalone (no torch/deepspeed):
    python test/test_bucket_manifest.py
"""

import os
import io
import sys
import json
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.bucket_manifest import BucketManifest, load_bucket_manifest
from utils.parquet_source import resolve_parquet_source, iter_parquet_rows

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def test_manifest_obj():
    m = BucketManifest({'a': 3, 'b': 1}, {'a': 'subjA'}, default_repeats=2)
    assert m.repeats_for('a') == 3
    assert m.repeats_for('b') == 1
    assert m.repeats_for('missing') == 2  # default
    assert m.bucket_for('a') == 'subjA'
    assert m.bucket_for('b') is None
    assert m.repeats_for('a') >= 1 and m.repeats_for('x') >= 1
    print('test_manifest_obj OK')


def test_load_json_flat_and_structured():
    d = tempfile.mkdtemp(prefix='bm_test_')
    try:
        flat = os.path.join(d, 'flat.json')
        with open(flat, 'w') as f:
            json.dump({'id1': 4, 'id2': 1}, f)
        m = load_bucket_manifest(flat)
        assert m.repeats_for('id1') == 4 and m.repeats_for('id2') == 1

        structured = os.path.join(d, 'structured.json')
        with open(structured, 'w') as f:
            json.dump({'default_repeats': 2, 'rows': {
                'id1': {'num_repeats': 5, 'bucket': 'truck'},
                'id2': {'num_repeats': 1},
            }}, f)
        m = load_bucket_manifest(structured)
        assert m.repeats_for('id1') == 5 and m.bucket_for('id1') == 'truck'
        assert m.repeats_for('id2') == 1
        assert m.repeats_for('unknown') == 2
        print('test_load_json_flat_and_structured OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_load_parquet_manifest():
    d = tempfile.mkdtemp(prefix='bm_test_')
    try:
        path = os.path.join(d, 'manifest.parquet')
        tbl = pa.table({
            'key': pa.array(['a', 'b', 'c']),
            'num_repeats': pa.array([2, 3, 1], type=pa.int32()),
            'bucket': pa.array(['x', 'x', 'y']),
        })
        pq.write_table(tbl, path)
        m = load_bucket_manifest(path)
        assert m.repeats_for('a') == 2 and m.repeats_for('b') == 3 and m.repeats_for('c') == 1
        assert m.bucket_for('a') == 'x' and m.bucket_for('c') == 'y'
        print('test_load_parquet_manifest OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _png(color, size=(8, 8)):
    b = io.BytesIO()
    Image.new('RGB', size, color).save(b, format='PNG')
    return b.getvalue()


def test_key_column_surfaced():
    d = tempfile.mkdtemp(prefix='bm_test_')
    try:
        tbl = pa.table({
            'image': pa.array([{'bytes': _png((1, 2, 3)), 'path': None},
                               {'bytes': _png((4, 5, 6)), 'path': None}],
                              type=pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
            'image_width': pa.array([8, 8], type=pa.int32()),
            'image_height': pa.array([8, 8], type=pa.int32()),
            'caption': pa.array(['cat', 'dog']),
            'id': pa.array(['row_a', 'row_b']),
        })
        pq.write_table(tbl, os.path.join(d, 's0.parquet'))
        # without a manifest configured, key column is not read (key is None)
        cfg = {'type': 'parquet', 'parquet_files': os.path.join(d, '*.parquet'),
               'caption_column': 'caption', 'caption_type': 'text'}
        rows = list(iter_parquet_rows(resolve_parquet_source(cfg)))
        assert all(r['key'] is None for r in rows)
        # with a manifest configured, the key column ('id') is surfaced
        cfg2 = dict(cfg); cfg2['bucket_manifest'] = 'dummy'; cfg2['manifest_key_column'] = 'id'
        rows2 = list(iter_parquet_rows(resolve_parquet_source(cfg2)))
        assert [r['key'] for r in rows2] == ['row_a', 'row_b']
        print('test_key_column_surfaced OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


ALL_TESTS = [
    test_manifest_obj,
    test_load_json_flat_and_structured,
    test_load_parquet_manifest,
    test_key_column_surfaced,
]

if __name__ == '__main__':
    for t in ALL_TESTS:
        t()
    print(f'\nAll {len(ALL_TESTS)} bucket_manifest tests passed.')
