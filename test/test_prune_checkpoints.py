"""Unit tests for prune_checkpoints.prune_global_steps.

Runnable standalone (no pytest, no GPU, no torch):
    python test/test_prune_checkpoints.py
"""
import os
import sys
import glob
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import prune_checkpoints as m


def _make_out(root):
    """A realistic run dir: a timestamp subdir with global_step* resume dirs + adapter dirs."""
    ts = os.path.join(root, '20260101_00-00-00')
    for s in (293, 584, 875, 3125, 6250):
        d = os.path.join(ts, f'global_step{s}')
        os.makedirs(d)
        open(os.path.join(d, 'mp_rank_00_model_states.pt'), 'w').write('x')  # fake resume shard
    # exported adapters (must NEVER be pruned)
    for name in ('epoch1', 'step3125', 'epoch2'):
        d = os.path.join(ts, name)
        os.makedirs(d)
        open(os.path.join(d, 'adapter_model.safetensors'), 'w').write('w')
    return ts


def test_keep_newest():
    tmp = tempfile.mkdtemp(prefix='prune_')
    try:
        ts = _make_out(tmp)
        found, removed = m.prune_global_steps(tmp, keep=2)
        assert found == 5 and removed == 3, (found, removed)
        # newest 2 by STEP NUMBER kept (not lexical: global_step6250/3125, not 875/584)
        kept = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ts, 'global_step*')))
        assert kept == ['global_step3125', 'global_step6250'], kept
        # adapters untouched
        assert len(glob.glob(os.path.join(ts, '**', 'adapter_model.safetensors'), recursive=True)) == 3
        print('test_keep_newest OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_keep_zero_deletes_all_resume_but_not_adapters():
    tmp = tempfile.mkdtemp(prefix='prune0_')
    try:
        ts = _make_out(tmp)
        found, removed = m.prune_global_steps(tmp, keep=0)
        assert found == 5 and removed == 5
        assert glob.glob(os.path.join(ts, 'global_step*')) == []
        assert len(glob.glob(os.path.join(ts, '**', 'adapter_model.safetensors'), recursive=True)) == 3
        print('test_keep_zero_deletes_all_resume_but_not_adapters OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fewer_than_keep_is_noop():
    tmp = tempfile.mkdtemp(prefix='prunefew_')
    try:
        ts = os.path.join(tmp, 'ts')
        os.makedirs(os.path.join(ts, 'global_step10'))
        found, removed = m.prune_global_steps(tmp, keep=3)
        assert found == 1 and removed == 0
        assert os.path.isdir(os.path.join(ts, 'global_step10'))
        print('test_fewer_than_keep_is_noop OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_dir_noop():
    tmp = tempfile.mkdtemp(prefix='pruneempty_')
    try:
        assert m.prune_global_steps(tmp, keep=3) == (0, 0)
        print('test_empty_dir_noop OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_keep_newest()
    test_keep_zero_deletes_all_resume_but_not_adapters()
    test_fewer_than_keep_is_noop()
    test_empty_dir_noop()
    print('ALL prune_checkpoints tests passed')
