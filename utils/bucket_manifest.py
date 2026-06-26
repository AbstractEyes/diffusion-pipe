"""Generic per-row bucket / repeat manifest.

NOTE: This is the generic plumbing that a higher-level "subject bucketing" pass
(developed separately, NOT part of the upstream diffusion-pipe PR) will produce
and feed into the parquet input source. This module defines only the interface +
a simple file loader; it contains no subject-clustering logic.

A manifest maps a stable per-row key (e.g. the dataset's `id` column) to:
  - num_repeats: how many times that row participates per epoch (default 1), and
  - bucket (optional): a logical label, recorded for downstream/analysis use.

The parquet input source applies num_repeats by replicating a row's caption list,
which adds that many iteration-order entries pointing at the single cached latent
(the image is still VAE-encoded only once). When no manifest is configured the
behavior is identical to plain per-directory num_repeats.

Dependency-light (stdlib + optional pyarrow) so it stays importable without torch.
"""

import json


class BucketManifest:
    def __init__(self, repeats_by_key=None, bucket_by_key=None, default_repeats=1):
        self.repeats_by_key = repeats_by_key or {}
        self.bucket_by_key = bucket_by_key or {}
        self.default_repeats = max(1, int(default_repeats))

    def repeats_for(self, key):
        return max(1, int(self.repeats_by_key.get(key, self.default_repeats)))

    def bucket_for(self, key):
        return self.bucket_by_key.get(key)

    def __len__(self):
        return len(self.repeats_by_key)

    def __repr__(self):
        return (f'BucketManifest(rows={len(self.repeats_by_key)}, '
                f'default_repeats={self.default_repeats})')


def _from_dict(data):
    if isinstance(data, dict) and 'rows' in data:
        default = data.get('default_repeats', 1)
        rows = data['rows']
        repeats, buckets = {}, {}
        for key, v in rows.items():
            if isinstance(v, dict):
                repeats[key] = v.get('num_repeats', v.get('repeats', 1))
                if v.get('bucket') is not None:
                    buckets[key] = v['bucket']
            else:
                repeats[key] = v
        return BucketManifest(repeats, buckets, default)
    if isinstance(data, dict):
        # flat {key: num_repeats}
        return BucketManifest({k: int(v) for k, v in data.items()})
    raise ValueError('Unrecognized bucket manifest structure')


def load_bucket_manifest(path):
    '''Load a BucketManifest from a .json file or a .parquet file.

    JSON: either {"default_repeats": N, "rows": {key: {"num_repeats": n, "bucket": b}}}
          or a flat {key: num_repeats}.
    Parquet: columns [key, num_repeats] (+ optional [bucket]).
    '''
    path = str(path)
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        cols = tbl.column_names
        key_col = 'key' if 'key' in cols else cols[0]
        rep_col = 'num_repeats' if 'num_repeats' in cols else ('repeats' if 'repeats' in cols else None)
        keys = tbl.column(key_col).to_pylist()
        repeats = tbl.column(rep_col).to_pylist() if rep_col else [1] * len(keys)
        repeats_by_key = {k: int(r) for k, r in zip(keys, repeats)}
        bucket_by_key = {}
        if 'bucket' in cols:
            for k, b in zip(keys, tbl.column('bucket').to_pylist()):
                if b is not None:
                    bucket_by_key[k] = b
        return BucketManifest(repeats_by_key, bucket_by_key)
    with open(path) as f:
        return _from_dict(json.load(f))
