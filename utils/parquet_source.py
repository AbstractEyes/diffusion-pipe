"""Parquet / HuggingFace dataset source helpers for diffusion-pipe.

Intentionally lightweight: depends only on pyarrow + utils.common (NOT torch /
datasets / deepspeed / comfy), so the source-reading core is unit-testable
without the full training stack.

Two roles:
- resolve_parquet_source + iter_parquet_rows: enumerate rows and read the small
  width/height/caption columns (NEVER the image bytes) for fast, PIL-free
  aspect-ratio bucketing.
- ParquetImageReader: lazily decode image bytes for a single row during latent
  caching, holding whole-shard image columns in a small LRU so the caching worker
  pool stays shard-local (each source shard is typically one large row group).
"""

import os
import glob
import json
from collections import OrderedDict

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# image_spec helpers.
#
# A parquet row is referenced exactly like a .tar member, as a 2-tuple image_spec,
# so all downstream code (bucketing, iteration_order, image_spec dict keys) keeps
# working unchanged. The first element encodes the parquet file path AND the image
# column (so the decoder is self-contained); the second is the row index as a
# string (keeps the Arrow column type uniformly (str, str)). Kept here (not in
# utils.common) so this module stays importable without torch/deepspeed.
# ---------------------------------------------------------------------------
PARQUET_SENTINEL = '\x00parquet\x00'
_PARQUET_COL_SEP = '\x00col\x00'
CAPTION_SENTINELS = {'', '__PARSEFAIL__', '__NO_TAGS__'}


def make_parquet_spec(parquet_path, row_index, image_column):
    return (
        PARQUET_SENTINEL + str(parquet_path) + _PARQUET_COL_SEP + str(image_column),
        str(int(row_index)),
    )


def is_parquet_spec(spec):
    return isinstance(spec[0], str) and spec[0].startswith(PARQUET_SENTINEL)


def parse_parquet_spec(spec):
    head = spec[0][len(PARQUET_SENTINEL):]
    path, _, column = head.partition(_PARQUET_COL_SEP)
    return path, column, int(spec[1])


def _flatten_caption_node(node):
    '''Flatten an arbitrary JSON caption node into a single caption string.

    Handles the common structured-caption shapes used by image-caption datasets
    ({"name","attributes"} subjects and {"subjects","actions","setting"} objects)
    and degrades gracefully to a recursive value-join for arbitrary JSON.
    '''
    if node is None:
        return ''
    if isinstance(node, str):
        s = node.strip()
        return '' if s.lower() == 'unknown' else s
    if isinstance(node, bool):
        return str(node)
    if isinstance(node, (int, float)):
        return str(node)
    if isinstance(node, list):
        parts = [_flatten_caption_node(x) for x in node]
        return ', '.join(p for p in parts if p)
    if isinstance(node, dict):
        # subject of the form {"name": ..., "attributes": [...]}
        if 'name' in node:
            attrs = node.get('attributes') or []
            attr_str = ' '.join(str(a).strip() for a in attrs if str(a).strip())
            name = str(node.get('name', '')).strip()
            return (attr_str + ' ' + name).strip()
        # structured caption {"subjects": [...], "actions": [...], "setting": ...}
        if any(k in node for k in ('subjects', 'actions', 'setting')):
            parts = []
            for k in ('subjects', 'actions', 'setting'):
                if k in node:
                    s = _flatten_caption_node(node[k])
                    if s:
                        parts.append(s)
            return ', '.join(parts)
        # arbitrary dict: join values
        parts = [_flatten_caption_node(v) for v in node.values()]
        return ', '.join(p for p in parts if p)
    return ''


def extract_caption(raw_value, caption_type='text', json_path=None):
    '''Extract a caption string from a parquet cell value.

    caption_type='text': the cell is the caption verbatim (incl. a raw JSON string
        if you want verbatim-JSON training).
    caption_type='json': json.loads the cell, optionally navigate a dot-separated
        json_path, then flatten to a string.
    Returns None for missing / sentinel / empty captions.
    '''
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip() in CAPTION_SENTINELS:
        return None
    if caption_type == 'text':
        s = raw_value.strip() if isinstance(raw_value, str) else str(raw_value)
        return s or None
    if caption_type == 'json':
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        try:
            obj = json.loads(raw_value)
        except (ValueError, TypeError):
            return None
        node = obj
        if json_path:
            for part in json_path.split('.'):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    node = None
                    break
        s = _flatten_caption_node(node)
        return s or None
    raise ValueError(f'Unknown caption_type {caption_type!r} (expected "text" or "json")')


class ParquetSource:
    def __init__(self, shard_paths, image_column, width_column, height_column,
                 caption_columns, caption_type, caption_json_path):
        self.shard_paths = shard_paths
        self.image_column = image_column
        self.width_column = width_column
        self.height_column = height_column
        self.caption_columns = caption_columns
        self.caption_type = caption_type
        self.caption_json_path = caption_json_path


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_parquet_source(directory_config, num_proc=4):
    '''Resolve a [[directory]] config of type 'huggingface' or 'parquet' into a
    ParquetSource (list of local parquet shard paths + column config).'''
    t = directory_config.get('type')
    image_column = directory_config.get('image_column', 'image')
    width_column = directory_config.get('width_column', 'image_width')
    height_column = directory_config.get('height_column', 'image_height')
    caption_column = directory_config.get('caption_column', 'caption')
    caption_columns = [caption_column] if isinstance(caption_column, str) else list(caption_column)
    caption_type = directory_config.get('caption_type', 'text')
    caption_json_path = directory_config.get('caption_json_path', None)

    if t == 'huggingface':
        from huggingface_hub import snapshot_download
        repo = directory_config['dataset']
        config = directory_config.get('config', 'default')
        patterns = [f'{config}/*.parquet', f'data/{config}/*.parquet', f'**/{config}/*.parquet']
        local_dir = snapshot_download(
            repo, repo_type='dataset', allow_patterns=patterns,
            max_workers=max(1, min(4, num_proc)),
        )
        shards = []
        for sub in (os.path.join(local_dir, config), os.path.join(local_dir, 'data', config)):
            if os.path.isdir(sub):
                shards = sorted(glob.glob(os.path.join(sub, '*.parquet')))
                if shards:
                    break
        if not shards:
            allp = sorted(glob.glob(os.path.join(local_dir, '**', '*.parquet'), recursive=True))
            shards = [s for s in allp if config in s.replace('\\', '/').split('/')] or allp
        if not shards:
            raise RuntimeError(f'No parquet shards found for dataset {repo!r} config {config!r}')
        # Narrow to a split only if the layout encodes split in the filename/dir
        # (e.g. train-00000-of-..., or .../train/...). Never widen.
        split = directory_config.get('split')
        if split:
            norm = lambda s: s.replace('\\', '/')
            split_shards = [
                s for s in shards
                if os.path.basename(s).startswith(split + '-')
                or os.path.basename(s).startswith(split + '_')
                or ('/' + split + '/') in norm(s)
            ]
            if split_shards:
                shards = split_shards
    elif t == 'parquet':
        files = (directory_config.get('parquet_files')
                 or directory_config.get('data_files')
                 or directory_config.get('parquet_path'))
        if files is None:
            raise ValueError("type='parquet' requires 'parquet_files' (a glob string or list of paths)")
        if isinstance(files, str):
            shards = sorted(glob.glob(files))
        else:
            shards = sorted(files)
        if not shards:
            raise RuntimeError(f'No parquet files matched {files!r}')
    else:
        raise ValueError(f'Unknown parquet source type {t!r} (expected "huggingface" or "parquet")')

    return ParquetSource(shards, image_column, width_column, height_column,
                         caption_columns, caption_type, caption_json_path)


def iter_parquet_rows(source):
    '''Yield one dict per source row with width/height/captions, reading ONLY the
    small metadata columns (never the image bytes). Rows are yielded in source
    (shard, row) order; row_in_shard is the shard-relative index used by the
    image_spec so ParquetImageReader can fetch the cell.'''
    cap_cols = source.caption_columns
    for shard in source.shard_paths:
        pf = pq.ParquetFile(shard)
        cols = _dedup([source.width_column, source.height_column] + cap_cols)
        row_in_shard = 0
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            w = tbl.column(source.width_column).to_pylist()
            h = tbl.column(source.height_column).to_pylist()
            capdata = {c: tbl.column(c).to_pylist() for c in cap_cols}
            for r in range(len(w)):
                captions = []
                for c in cap_cols:
                    cap = extract_caption(capdata[c][r], source.caption_type, source.caption_json_path)
                    if cap:
                        captions.append(cap)
                yield {
                    'parquet_path': shard,
                    'row_in_shard': row_in_shard,
                    'image_spec': make_parquet_spec(shard, row_in_shard, source.image_column),
                    'width': w[r],
                    'height': h[r],
                    'captions': captions,
                }
                row_in_shard += 1


class ParquetImageReader:
    '''Decode image bytes for a single (shard, row) cell during latent caching.

    Source shards are usually a single large row group, so per-row random reads
    would re-read the whole shard. We therefore read a shard's image column ONCE
    (whole column) and keep it in a small LRU. When the caching worker pool
    iterates source-ordered metadata, all workers stay within the same shard
    region, so each shard column is materialised about once per worker.
    '''
    def __init__(self, lru=8):
        self.lru = max(1, lru)
        self._files = {}
        self._cols = OrderedDict()  # (path, column) -> pyarrow ChunkedArray

    def _column(self, parquet_path, image_column):
        key = (parquet_path, image_column)
        arr = self._cols.get(key)
        if arr is not None:
            self._cols.move_to_end(key)
            return arr
        pf = self._files.get(parquet_path)
        if pf is None:
            pf = pq.ParquetFile(parquet_path, memory_map=True)
            self._files[parquet_path] = pf
        arr = pf.read(columns=[image_column]).column(image_column)
        self._cols[key] = arr
        while len(self._cols) > self.lru:
            old_key, _ = self._cols.popitem(last=False)
        return arr

    def read_cell_bytes(self, parquet_path, image_column, row):
        '''Return the raw encoded image bytes for the given cell.'''
        cell = self._column(parquet_path, image_column)[row].as_py()
        if isinstance(cell, dict):
            data = cell.get('bytes')
            if data is None and cell.get('path'):
                with open(cell['path'], 'rb') as f:
                    data = f.read()
            return data
        if isinstance(cell, (bytes, bytearray)):
            return bytes(cell)
        raise TypeError(f'Unexpected image cell type {type(cell)} in {parquet_path}')
