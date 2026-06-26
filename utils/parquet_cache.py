"""Multi-column parquet cache backend for diffusion-pipe.

Drop-in alternative to utils.cache.Cache. Same public API:
    __init__(path, fingerprint, ...), __len__, __getitem__(int)->dict,
    add(item: dict), finalize_current_shard(), clear(), .fingerprint, .path

Instead of torch.save-per-item into opaque binary shards (the legacy Cache), this
stores each cached item as a row in parquet shards with ONE COLUMN PER dict key.
Tensors are stored as raw little-endian bytes plus `<key>__shape` / `<key>__dtype`
side columns, so every dtype (including bfloat16 / float8, which numpy can't
represent) round-trips bit-exactly. Shards are capped at ~350 MB so they are
friendly to HuggingFace upload, and each finalized shard can be backed up to a HF
dataset repo and re-downloaded on a fresh machine.

Design constraints (must match legacy Cache so the read path is unchanged):
- __len__ counts only fully-flushed rows. This is the crash-safe resume point that
  utils.dataset._map_and_cache relies on (it does dataset.select(range(len(cache), N))).
- __getitem__ returns freshly-allocated, owning, writable tensors (like torch.load).
"""

import os
import io
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq


SHARD_FMT = 'shard_{:05d}.parquet'
INDEX_FILE = '_index.json'
DEFAULT_SHARD_SIZE_MB = 350
DEFAULT_ROW_GROUP_SIZE = 64
ROW_GROUP_LRU = 4

# torch dtype <-> string. We always route tensor bytes through torch (never numpy)
# so dtypes numpy lacks (bfloat16, float8_*) survive a round-trip.
_TORCH_DTYPE_TO_STR = {
    torch.float32: 'float32', torch.float64: 'float64', torch.float16: 'float16',
    torch.bfloat16: 'bfloat16', torch.int64: 'int64', torch.int32: 'int32',
    torch.int16: 'int16', torch.int8: 'int8', torch.uint8: 'uint8', torch.bool: 'bool',
    torch.float8_e4m3fn: 'float8_e4m3fn', torch.float8_e5m2: 'float8_e5m2',
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}


def _tensor_to_bytes(t):
    # Reinterpret the contiguous storage as raw bytes via torch, never numpy, so
    # bfloat16 / float8 are preserved. uint8 view -> numpy -> tobytes is zero-info-loss.
    return t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def _bytes_to_tensor(buf, shape, dtype_str):
    dt = _STR_TO_TORCH_DTYPE[dtype_str]
    # bytearray makes the buffer owning + writable so the result matches torch.load
    # semantics (callers may mutate, e.g. .unsqueeze_, randn_like target, collate stack).
    flat = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    return flat.view(dt).reshape(tuple(int(x) for x in shape))


class ParquetCache:
    def __init__(self, path, fingerprint, shard_size_mb=DEFAULT_SHARD_SIZE_MB,
                 hf_repo=None, hf_upload=False, row_group_size=DEFAULT_ROW_GROUP_SIZE):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.shard_size_mb = shard_size_mb
        self.hf_repo = hf_repo
        self.hf_upload = hf_upload
        self.row_group_size = row_group_size
        os.makedirs(self.path, exist_ok=True)

        # write-side buffer
        self._buf = []          # list[dict] of arrow-ready encoded rows
        self._buf_nbytes = 0    # running estimate of raw tensor payload bytes

        # read-side state (rebuilt lazily; guarded against DataLoader worker fork)
        self._pid = os.getpid()
        self._open_files = {}   # shard_id -> pq.ParquetFile
        self._rg_cache = OrderedDict()  # (shard_id, rg_id) -> pa.Table (LRU)
        self._cum = None        # cumulative row offsets per shard, for idx -> (shard, row)

        self._load_or_init_index()

    # ---------------------------------------------------------------- index --
    def _index_path(self):
        return self.path / INDEX_FILE

    def _load_or_init_index(self):
        if self._index_path().exists():
            with open(self._index_path()) as f:
                self.index = json.load(f)
            if self.index.get('fingerprint') != self.fingerprint:
                print('[PARQUET-CACHE] Fingerprint changed, clearing existing cache')
                self.clear()
                return
            # Prune orphan shards left by a crash between shard write and index commit.
            self._prune_orphans()
        else:
            self.index = {'fingerprint': self.fingerprint, 'shards': [], 'total': 0, 'next_shard': 0}
            self._write_index()
        self._cum = None
        print(f'[PARQUET-CACHE] {self.path.name}: existing length {len(self)}')

    def _write_index(self):
        tmp = self.path / (INDEX_FILE + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.index, f)
        os.replace(tmp, self._index_path())

    def _prune_orphans(self):
        valid = {s['file'] for s in self.index['shards']}
        for p in self.path.glob('shard_*.parquet'):
            if p.name not in valid:
                print(f'[PARQUET-CACHE] Pruning orphan shard {p.name}')
                os.remove(p)
        for p in self.path.glob('shard_*.parquet.tmp'):
            os.remove(p)

    def __len__(self):
        # Only durably-flushed rows. The in-memory buffer is intentionally excluded.
        return self.index['total']

    # ---------------------------------------------------------------- write --
    def add(self, item: dict):
        row, nbytes = self._encode_row(item)
        self._buf.append(row)
        self._buf_nbytes += nbytes
        if self._buf_nbytes >= self.shard_size_mb * 1_000_000:
            self.finalize_current_shard()

    def _encode_row(self, item):
        row = {}
        nbytes = 0
        for k, v in item.items():
            if torch.is_tensor(v):
                b = _tensor_to_bytes(v)
                row[k] = b
                row[k + '__shape'] = [int(x) for x in v.shape]
                row[k + '__dtype'] = _TORCH_DTYPE_TO_STR[v.dtype]
                nbytes += len(b)
            elif v is None:
                row[k] = None
                row[k + '__shape'] = None
                row[k + '__dtype'] = None
            elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
                # unbatch_iter in utils.dataset splits batches so every cached item
                # key is a single tensor/scalar/None. List-of-tensors should never
                # reach here; fail loudly rather than ship an untested byte-splitter.
                raise NotImplementedError(
                    f'ParquetCache v1 does not support list-of-tensors for key {k!r}'
                )
            elif isinstance(v, (list, tuple)):
                row[k + '__json'] = json.dumps(list(v))
            elif isinstance(v, str):
                row[k] = v
            elif isinstance(v, (bool, int, float)):
                row[k] = v
            else:
                raise TypeError(f'ParquetCache cannot encode key {k!r} of type {type(v)}')
        return row, nbytes

    def finalize_current_shard(self):
        if not self._buf:
            return  # no-op, matches legacy semantics
        shard_id = self.index['next_shard']
        fname = SHARD_FMT.format(shard_id)
        table = pa.Table.from_pylist(self._buf)
        tmp = self.path / (fname + '.tmp')
        pq.write_table(
            table, tmp,
            compression='zstd', compression_level=3,
            row_group_size=self.row_group_size,
            use_dictionary=False,
        )
        os.replace(tmp, self.path / fname)  # atomic publish: file exists before index
        n = len(self._buf)
        self.index['shards'].append({'file': fname, 'rows': n, 'uploaded': False})
        self.index['total'] += n
        self.index['next_shard'] += 1
        self._write_index()  # index is now the authoritative length
        self._buf = []
        self._buf_nbytes = 0
        self._cum = None
        if self.hf_upload and self.hf_repo:
            self._upload_shard(shard_id)

    def clear(self):
        for p in self.path.glob('shard_*.parquet'):
            os.remove(p)
        for p in self.path.glob('shard_*.parquet.tmp'):
            os.remove(p)
        if self._index_path().exists():
            os.remove(self._index_path())
        self._buf = []
        self._buf_nbytes = 0
        self._open_files = {}
        self._rg_cache.clear()
        self.index = {'fingerprint': self.fingerprint, 'shards': [], 'total': 0, 'next_shard': 0}
        self._write_index()
        self._cum = None

    # ----------------------------------------------------------------- read --
    def _ensure_reader(self):
        # Reset inherited handles when a DataLoader worker forks this process.
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self._open_files = {}
            self._rg_cache = OrderedDict()
        if self._cum is None:
            self._cum = []
            total = 0
            for s in self.index['shards']:
                self._cum.append((total, total + s['rows'], s['file']))
                total += s['rows']

    def _locate(self, idx):
        for start, end, fname in self._cum:
            if start <= idx < end:
                return fname, idx - start
        raise IndexError(f'index {idx} out of range for ParquetCache of length {len(self)}')

    def _get_file(self, fname):
        if fname not in self._open_files:
            local = self.path / fname
            if not local.exists() and self.hf_repo:
                self._download_shard(fname)
            self._open_files[fname] = pq.ParquetFile(local, memory_map=True)
        return self._open_files[fname]

    def _get_row_group(self, fname, rg_id):
        key = (fname, rg_id)
        tbl = self._rg_cache.get(key)
        if tbl is None:
            tbl = self._get_file(fname).read_row_group(rg_id)
            self._rg_cache[key] = tbl
            while len(self._rg_cache) > ROW_GROUP_LRU:
                self._rg_cache.popitem(last=False)
        else:
            self._rg_cache.move_to_end(key)
        return tbl

    def __getitem__(self, idx):
        idx = int(idx)
        self._ensure_reader()
        fname, local_row = self._locate(idx)
        rg_id = local_row // self.row_group_size
        row_in_rg = local_row % self.row_group_size
        table = self._get_row_group(fname, rg_id)
        return self._decode_row(table, row_in_rg)

    def _decode_row(self, table, r):
        names = table.column_names
        nameset = set(names)
        out = {}
        # Reconstruct tensors / scalars; skip the side columns (__shape/__dtype/__json).
        for name in names:
            if name.endswith('__shape') or name.endswith('__dtype'):
                continue
            if name.endswith('__json'):
                base = name[:-len('__json')]
                val = table.column(name)[r].as_py()
                out[base] = tuple(json.loads(val)) if val is not None else None
                continue
            shape_col = name + '__shape'
            if shape_col in nameset:
                # tensor-valued column
                buf = table.column(name)[r].as_py()
                if buf is None:
                    out[name] = None
                else:
                    shape = table.column(shape_col)[r].as_py()
                    dtype_str = table.column(name + '__dtype')[r].as_py()
                    out[name] = _bytes_to_tensor(buf, shape, dtype_str)
            else:
                # native scalar column (caption str, caption_number int, ...)
                out[name] = table.column(name)[r].as_py()
        return out

    # ------------------------------------------------------------- HF backup --
    def _repo_subpath(self, fname):
        # Encode the full cache subpath so shards from different buckets/encoders
        # never collide on the Hub. cache dirs always contain a 'cache' segment.
        parts = self.path.parts
        if 'cache' in parts:
            rel = parts[parts.index('cache'):]
        else:
            rel = parts[-3:]
        return '/'.join(rel) + '/' + fname

    def _upload_shard(self, shard_id):
        from huggingface_hub import upload_file
        fname = SHARD_FMT.format(shard_id)
        try:
            upload_file(
                path_or_fileobj=str(self.path / fname),
                path_in_repo=self._repo_subpath(fname),
                repo_id=self.hf_repo,
                repo_type='dataset',
            )
            # also publish the index so a fresh machine can discover shards
            upload_file(
                path_or_fileobj=str(self._index_path()),
                path_in_repo=self._repo_subpath(INDEX_FILE),
                repo_id=self.hf_repo,
                repo_type='dataset',
            )
            for s in self.index['shards']:
                if s['file'] == fname:
                    s['uploaded'] = True
            self._write_index()
            print(f'[PARQUET-CACHE] Uploaded {fname} to {self.hf_repo}')
        except Exception as e:
            print(f'[PARQUET-CACHE] WARNING: failed to upload {fname}: {e}')

    def _download_shard(self, fname):
        from huggingface_hub import hf_hub_download
        try:
            local = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self._repo_subpath(fname),
                repo_type='dataset',
            )
            os.makedirs(self.path, exist_ok=True)
            target = self.path / fname
            if not target.exists():
                import shutil
                shutil.copyfile(local, target)
            print(f'[PARQUET-CACHE] Downloaded {fname} from {self.hf_repo}')
        except Exception as e:
            raise FileNotFoundError(
                f'Shard {fname} missing locally and download from {self.hf_repo} failed: {e}'
            )


# Quick self-test
if __name__ == '__main__':
    import tempfile
    d = tempfile.mkdtemp()
    cache = ParquetCache(d, 'fp-test', shard_size_mb=0.001, row_group_size=4)
    items = []
    for i in range(10):
        item = {
            'latents': torch.randn(2, 3, 4, dtype=torch.bfloat16),
            'mask': (torch.rand(4, 4, dtype=torch.float16) if i % 2 else None),
            'attn_mask': torch.ones(5, dtype=torch.int64),
            'image_spec': ('\x00parquet\x00/some/file.parquet', str(i)),
            'caption': f'caption {i}',
            'caption_number': i,
        }
        items.append(item)
        cache.add(item)
    cache.finalize_current_shard()
    assert len(cache) == 10, len(cache)
    cache2 = ParquetCache(d, 'fp-test', row_group_size=4)
    assert len(cache2) == 10
    for i in range(10):
        got = cache2[i]
        ref = items[i]
        assert torch.equal(got['latents'], ref['latents']), i
        assert got['latents'].dtype == torch.bfloat16
        if ref['mask'] is None:
            assert got['mask'] is None, i
        else:
            assert torch.equal(got['mask'], ref['mask']), i
        assert got['caption'] == ref['caption']
        assert got['caption_number'] == i
        assert tuple(got['image_spec']) == ref['image_spec'], (got['image_spec'], ref['image_spec'])
    print('ParquetCache self-test OK')
