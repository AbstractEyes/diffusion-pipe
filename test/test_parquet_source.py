"""Unit tests for utils.parquet_source (caption extraction, image_spec helpers,
columnar row iteration, and shard-LRU image decode).

Runnable standalone (no pytest, no torch/deepspeed required):
    python test/test_parquet_source.py
Depends only on pyarrow + pillow.
"""

import os
import io
import sys
import json
import glob
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parquet_source import (
    make_parquet_spec, is_parquet_spec, parse_parquet_spec, extract_caption,
    resolve_parquet_source, iter_parquet_rows, ParquetImageReader,
)

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


# ----- a realistic deepfashion-style caption_vlm_json -----
VLM = json.dumps({
    "subjects": [
        {"name": "woman", "attributes": []},
        {"name": "t-shirt", "attributes": ["black"]},
        {"name": "jeans", "attributes": ["blue", "high-waisted"]},
    ],
    "actions": ["standing", "hand in pocket"],
    "setting": "unknown",
})
SRC = json.dumps({"deepfashion_caption": "a woman wearing a black shirt and jeans"})


def test_extract_caption():
    # text mode = verbatim
    assert extract_caption("hello world", 'text') == "hello world"
    assert extract_caption(VLM, 'text') == VLM  # verbatim-JSON training case
    # json mode, structured flatten (setting 'unknown' dropped)
    flat = extract_caption(VLM, 'json')
    assert flat == "woman, black t-shirt, blue high-waisted jeans, standing, hand in pocket", flat
    # json mode with a path to a plain string
    assert extract_caption(SRC, 'json', 'deepfashion_caption') == "a woman wearing a black shirt and jeans"
    # sentinels / empties -> None
    for s in ('', '__PARSEFAIL__', '__NO_TAGS__', None):
        assert extract_caption(s, 'json') is None, s
        assert extract_caption(s, 'text') is None, s
    # malformed json -> None
    assert extract_caption('{not json', 'json') is None
    # missing path -> None
    assert extract_caption(SRC, 'json', 'nonexistent') is None
    print('test_extract_caption OK')


def test_image_spec_helpers():
    spec = make_parquet_spec('/data/shard_0.parquet', 42, 'image')
    assert is_parquet_spec(spec)
    assert isinstance(spec[0], str) and isinstance(spec[1], str)
    path, col, row = parse_parquet_spec(spec)
    assert path == '/data/shard_0.parquet' and col == 'image' and row == 42
    # legacy specs are not parquet specs and never collide
    assert not is_parquet_spec((None, '/some/file.png'))
    assert not is_parquet_spec(('archive.tar', 'member.png'))
    assert not is_parquet_spec((None, None))
    print('test_image_spec_helpers OK')


def _make_png_bytes(color, size=(8, 8)):
    im = Image.new('RGB', size, color)
    b = io.BytesIO()
    im.save(b, format='PNG')
    return b.getvalue()


def _write_parquet(path, rows):
    images = [{'bytes': r['bytes'], 'path': None} for r in rows]
    table = pa.table({
        'image': pa.array(images, type=pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
        'image_width': pa.array([r['w'] for r in rows], type=pa.int32()),
        'image_height': pa.array([r['h'] for r in rows], type=pa.int32()),
        'caption_vlm_json': pa.array([r['cap'] for r in rows], type=pa.string()),
    })
    pq.write_table(table, path)


def test_local_parquet_roundtrip():
    d = tempfile.mkdtemp(prefix='pqsrc_test_')
    try:
        rows0 = [
            {'bytes': _make_png_bytes((255, 0, 0), (16, 16)), 'w': 16, 'h': 16, 'cap': VLM},
            {'bytes': _make_png_bytes((0, 255, 0), (32, 16)), 'w': 32, 'h': 16, 'cap': SRC},
            {'bytes': _make_png_bytes((0, 0, 255), (16, 32)), 'w': 16, 'h': 32, 'cap': ''},  # empty caption
        ]
        rows1 = [
            {'bytes': _make_png_bytes((128, 128, 0), (24, 24)), 'w': 24, 'h': 24, 'cap': VLM},
        ]
        _write_parquet(os.path.join(d, 'shard_0000.parquet'), rows0)
        _write_parquet(os.path.join(d, 'shard_0001.parquet'), rows1)

        cfg = {
            'type': 'parquet',
            'parquet_files': os.path.join(d, '*.parquet'),
            'image_column': 'image',
            'caption_column': 'caption_vlm_json',
            'caption_type': 'json',
        }
        source = resolve_parquet_source(cfg)
        assert len(source.shard_paths) == 2

        all_rows = list(iter_parquet_rows(source))
        assert len(all_rows) == 4, len(all_rows)
        # source order preserved: shard0 rows 0,1,2 then shard1 row 0
        assert all_rows[0]['width'] == 16 and all_rows[0]['height'] == 16
        assert all_rows[1]['width'] == 32 and all_rows[1]['height'] == 16
        assert all_rows[3]['width'] == 24
        # row_in_shard resets per shard
        assert all_rows[0]['row_in_shard'] == 0 and all_rows[2]['row_in_shard'] == 2
        assert all_rows[3]['row_in_shard'] == 0
        # captions extracted; the empty-caption row yields []
        assert all_rows[0]['captions'] == ['woman, black t-shirt, blue high-waisted jeans, standing, hand in pocket']
        assert all_rows[2]['captions'] == []
        # image_spec is a parquet spec pointing at the right (shard, row)
        for r in all_rows:
            assert is_parquet_spec(r['image_spec'])
            p, c, rr = parse_parquet_spec(r['image_spec'])
            assert p == r['parquet_path'] and c == 'image' and rr == r['row_in_shard']

        # ParquetImageReader returns the exact bytes, decodable by PIL
        reader = ParquetImageReader(lru=1)  # lru=1 forces re-read across shards
        for r in all_rows:
            p, c, rr = parse_parquet_spec(r['image_spec'])
            raw = reader.read_cell_bytes(p, c, rr)
            im = Image.open(io.BytesIO(raw))
            assert im.size == (r['width'], r['height']), (im.size, r['width'], r['height'])
        print('test_local_parquet_roundtrip OK')
    finally:
        shutil.rmtree(d, ignore_errors=True)


ALL_TESTS = [
    test_extract_caption,
    test_image_spec_helpers,
    test_local_parquet_roundtrip,
]

if __name__ == '__main__':
    for t in ALL_TESTS:
        t()
    print(f'\nAll {len(ALL_TESTS)} parquet_source tests passed.')
