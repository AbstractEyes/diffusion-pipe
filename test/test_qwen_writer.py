"""Unit test for the qwen_extraction ShardWriter: it must produce parquet whose `image` column
round-trips through utils/parquet_source.ParquetImageReader (the diffusion-pipe ingestion path).

Runs in the CPU testvenv (datasets + pyarrow + pillow): python test/test_qwen_writer.py
"""
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image

from qwen_extraction import qwen_lightning_extraction as qx
from utils.parquet_source import ParquetImageReader


def _png(w, h, color):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), (w, h)


def _row(rid, w, h):
    return {
        "id": rid, "image_width": w, "image_height": h,
        "prompt": f"prompt {rid}", "source_prompt": f"src {rid}",
        "race": "caucasian", "race_injected": True, "is_tail": False,
        "gender": "woman", "age_band": "25-35", "hair": "brunette",
        "seed": 123 + int(rid), "width_ratio": f"{w}x{h}", "policy_version": "augment-v1",
    }


def test_write_and_roundtrip():
    d = tempfile.mkdtemp()
    try:
        w = qx.ShardWriter(d, rank=0, out_repo=None, upload=False, shard_size_mb=999)
        specs = [("0", 1024, 1024, (200, 100, 50)),
                 ("1", 832, 1216, (10, 220, 30)),
                 ("2", 1216, 832, (30, 40, 240))]
        expect = {}
        for rid, ww, hh, col in specs:
            png, _ = _png(ww, hh, col)
            w.add(_row(rid, ww, hh), png)
            expect[rid] = (ww, hh)
        w.finalize_current_shard()

        import pyarrow.parquet as pq
        shard = os.path.join(d, "rank0", "shard_r0_00000.parquet")
        assert os.path.exists(shard), "shard not written"
        pf = pq.ParquetFile(shard)
        cols = set(pf.schema_arrow.names)
        for need in ("id", "image", "image_width", "image_height", "prompt", "race"):
            assert need in cols, (need, cols)
        ids = pf.read(columns=["id"]).column("id").to_pylist()
        del pf

        # round-trip the image column exactly as diffusion-pipe would
        reader = ParquetImageReader(lru=4)
        for row in range(3):
            data = reader.read_cell_bytes(shard, "image", row)
            assert isinstance(data, (bytes, bytearray)) and len(data) > 0
            img = Image.open(io.BytesIO(data))
            img.load()
            assert (img.width, img.height) == expect[ids[row]], (ids[row], img.size)
        del reader
        print("test_write_and_roundtrip OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_index():
    d = tempfile.mkdtemp()
    try:
        w = qx.ShardWriter(d, rank=2, out_repo=None, upload=False, shard_size_mb=999)
        for rid in ("5", "9", "13"):
            png, _ = _png(64, 64, (1, 2, 3))
            w.add(_row(rid, 64, 64), png)
        w.finalize_current_shard()
        # reopen -> done_ids recovered from index, orphan-prune leaves the shard intact
        w2 = qx.ShardWriter(d, rank=2, out_repo=None, upload=False, shard_size_mb=999)
        assert {"5", "9", "13"} <= w2.done_ids, w2.done_ids
        assert os.path.exists(os.path.join(d, "rank2", "shard_r2_00000.parquet"))
        print("test_resume_index OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


ALL = [test_write_and_roundtrip, test_resume_index]

if __name__ == "__main__":
    for t in ALL:
        t()
    print(f"\nAll {len(ALL)} qwen_writer tests passed.")
