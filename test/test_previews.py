"""Unit tests for utils/previews.py (training previews).

Runnable standalone on CPU (no pytest, no GPU, no deepspeed):
    python test/test_previews.py
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from PIL import Image

from utils import previews as pv


def _settings(**over):
    raw = {'prompts': ['A lighthouse, at dusk!', 'an empty office at night', 'a pool', 'fog'], **over}
    return pv.parse_samples_config({'samples': raw})


class FakeModel:
    """Duck-typed stand-in for a pipeline: records calls; sample() returns noise drawn from the generator."""

    def __init__(self, fail_on=(), fail_schedule=False, text_encoders=1):
        self.fail_on = set(fail_on)
        self.fail_schedule = fail_schedule
        self.calls = []
        self._tes = [object()] * text_encoders

    def get_conds(self, inputs):
        return ()

    def vae_decode(self, latents):
        return latents

    def encode_sample_prompt(self, prompt, negative_prompt='', cfg=1):
        return (torch.zeros(1),), ((torch.zeros(1),) if cfg > 1 else None)

    def get_text_encoders(self):
        return self._tes

    def set_sample_schedule(self, steps, shift):
        self.calls.append(('schedule', steps, shift))
        if self.fail_schedule:
            raise RuntimeError('schedule boom')

    def prepare_block_swap_inference(self, disable_block_swap=False):
        self.calls.append(('swap_inference',))

    def prepare_block_swap_training(self):
        self.calls.append(('swap_training',))

    def sample(self, w, h, conds=None, unconds=None, cfg=None, generator=None, show_progress=True):
        idx = int(conds[0].item())
        self.calls.append(('sample', idx, w, h, cfg, show_progress))
        if idx in self.fail_on:
            raise RuntimeError(f'sample boom {idx}')
        return torch.rand((1, h, w, 3), generator=generator)


class FakeWriter:
    def __init__(self):
        self.images = []

    def add_image(self, tag, img, step, dataformats='CHW'):
        self.images.append((tag, img.shape, step, dataformats))

    def flush(self):
        pass


def _conds(n):
    # conds[i][0][0] carries the prompt index so FakeModel.sample can tell which prompt it is rendering
    return [((torch.tensor([float(i)]),), (torch.zeros(1),)) for i in range(n)]


def test_parse_defaults_and_off():
    assert pv.parse_samples_config({}) is None
    assert pv.parse_samples_config({'samples': {'prompts': []}}) is None
    s = pv.parse_samples_config({'samples': {'prompts': 'one prompt'}})
    assert s['prompts'] == ['one prompt']
    assert (s['width'], s['height'], s['steps'], s['seed']) == (1024, 1024, 30, 42)
    assert s['cfg'] == 4.0 and s['shift'] == 3.0 and s['before_first_step'] is True
    s = pv.parse_samples_config({'samples': {'prompts': ['a'], 'cfg': 5, 'width': 768}})
    assert isinstance(s['cfg'], float) and s['cfg'] == 5.0 and s['width'] == 768
    print('test_parse_defaults_and_off OK')


def test_parse_rejects_bad_values():
    bad = [
        {'prompts': ['a'], 'widht': 1024},          # typo -> unknown key
        {'prompts': ['a'], 'width': 1000},          # not a multiple of 16
        {'prompts': ['a'], 'steps': 0},
        {'prompts': ['a'], 'steps': True},          # bool is not an int here
        {'prompts': ['a'], 'cfg': 0},
        {'prompts': ['a'], 'seed': 1.5},
        {'prompts': ['a', '']},                     # empty prompt
        {'prompts': ['a'], 'negative_prompt': 3},
        {'prompts': ['a'], 'before_first_step': 'yes'},
    ]
    for raw in bad:
        try:
            pv.parse_samples_config({'samples': raw})
        except ValueError:
            continue
        raise AssertionError(f'accepted bad config {raw}')
    try:
        pv.parse_samples_config({'samples': ['not', 'a', 'table']})
        raise AssertionError('accepted a non-table')
    except ValueError:
        pass
    print('test_parse_rejects_bad_values OK')


def test_assign_and_names():
    for n, world in ((5, 2), (4, 4), (3, 8), (1, 1)):
        shares = [pv.assign_prompts(n, r, world) for r in range(world)]
        flat = sorted(i for s in shares for i in s)
        assert flat == list(range(n)), (n, world, shares)
    assert pv.assign_prompts(5, 0, 2) == [0, 2, 4] and pv.assign_prompts(5, 1, 2) == [1, 3]
    assert pv.image_filename(0, 'A lighthouse, at dusk!') == '00_a-lighthouse-at-dusk.png'
    assert pv.image_filename(7, '!!!') == '07_prompt.png'
    assert len(pv.slugify('x' * 100)) == 40
    print('test_assign_and_names OK')


def test_check_supported():
    cfg = {'model': {'type': 'fake'}, 'pipeline_stages': 1}
    pv.check_supported(FakeModel(), cfg)
    for model, config in ((object(), cfg),
                          (FakeModel(), {**cfg, 'pipeline_stages': 2}),
                          (FakeModel(text_encoders=0), cfg)):
        try:
            pv.check_supported(model, config)
        except ValueError:
            continue
        raise AssertionError('accepted an unsupported setup')
    print('test_check_supported OK')


def test_to_pil():
    img = pv.to_pil(torch.full((1, 32, 48, 3), 2.0))  # clamps to 1.0
    assert img.size == (48, 32) and img.mode == 'RGB' and img.getpixel((0, 0)) == (255, 255, 255)
    img = pv.to_pil(torch.zeros((16, 16, 3)))
    assert img.getpixel((5, 5)) == (0, 0, 0)
    print('test_to_pil OK')


def test_render_two_ranks_and_tensorboard():
    tmp = tempfile.mkdtemp(prefix='previews_')
    try:
        s = _settings(width=64, height=32, steps=7, shift=2.5)
        barriers = []
        writer = FakeWriter()
        m1, m0 = FakeModel(), FakeModel()
        r1 = pv.PreviewRenderer(m1, s, _conds(4), tmp, None, rank=1, world=2,
                                barrier=lambda: barriers.append(1), device='cpu')
        r0 = pv.PreviewRenderer(m0, s, _conds(4), tmp, writer, rank=0, world=2,
                                barrier=lambda: barriers.append(0), device='cpu')
        w1 = r1.render('epoch2', 100)          # rank 1 first, so rank 0 sees every file when it logs
        w0 = r0.render('epoch2', 100)
        out = os.path.join(tmp, 'samples', 'epoch2')
        assert [os.path.basename(p) for p in w0] == ['00_a-lighthouse-at-dusk.png', '02_a-pool.png']
        assert [os.path.basename(p) for p in w1] == ['01_an-empty-office-at-night.png', '03_fog.png']
        assert sorted(f for f in os.listdir(out) if f.endswith('.png')) == sorted(os.path.basename(p) for p in w0 + w1)
        assert os.path.exists(os.path.join(out, 'prompts.txt'))
        assert Image.open(w0[0]).size == (64, 32)
        assert barriers == [1, 0]
        assert [c for c in m0.calls if c[0] == 'schedule'] == [('schedule', 7, 2.5)]
        assert ('swap_inference',) in m0.calls and m0.calls[-1] == ('swap_training',)
        assert all(c[5] is False for c in m0.calls if c[0] == 'sample')     # no progress bars
        assert [t for t, *_ in writer.images] == ['samples/00', 'samples/01', 'samples/02', 'samples/03']
        assert all(shape == (32, 64, 3) and step == 100 and fmt == 'HWC' for _, shape, step, fmt in writer.images)
        print('test_render_two_ranks_and_tensorboard OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_same_seed_same_image_across_saves():
    tmp = tempfile.mkdtemp(prefix='previews_seed_')
    try:
        s = _settings(width=32, height=32)
        r = pv.PreviewRenderer(FakeModel(), s, _conds(4), tmp, None, device='cpu')
        a = r.render('epoch1', 10)
        torch.rand(1000)  # training draws from the global RNG between saves
        b = r.render('epoch2', 20)
        for pa, pb in zip(a, b):
            assert open(pa, 'rb').read() == open(pb, 'rb').read(), (pa, pb)
        assert open(a[0], 'rb').read() != open(a[1], 'rb').read()   # different prompts, different seeds
        # the global RNG stream is left as it was
        torch.manual_seed(0); x = torch.rand(3)
        torch.manual_seed(0); r.render('epoch3', 30); y = torch.rand(3)
        assert torch.equal(x, y)
        print('test_same_seed_same_image_across_saves OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_failures_never_raise():
    tmp = tempfile.mkdtemp(prefix='previews_fail_')
    try:
        s = _settings(width=32, height=32)
        barriers = []
        m = FakeModel(fail_on={1})
        r = pv.PreviewRenderer(m, s, _conds(4), tmp, FakeWriter(), barrier=lambda: barriers.append(0), device='cpu')
        written = r.render('epoch4', 40)
        assert [os.path.basename(p)[:2] for p in written] == ['00', '02', '03']
        assert barriers == [0]
        m = FakeModel(fail_schedule=True)
        r = pv.PreviewRenderer(m, s, _conds(4), tmp, None, barrier=lambda: barriers.append(1), device='cpu')
        assert r.render('epoch5', 50) == []
        assert barriers == [0, 1] and m.calls[-1] == ('swap_training',)
        print('test_failures_never_raise OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_mode_during_sampling_then_restored():
    tmp = tempfile.mkdtemp(prefix='previews_mode_')
    try:
        m = FakeModel()
        m.pipeline_model = torch.nn.Dropout(0.5)       # stands in for the pipeline module
        modes = []
        orig = m.sample
        def spy(*a, **k):
            modes.append(m.pipeline_model.training)
            return orig(*a, **k)
        m.sample = spy
        r = pv.PreviewRenderer(m, _settings(width=32, height=32), _conds(4), tmp, None, device='cpu')
        r.render('epoch1', 1)
        assert modes == [False] * 4 and m.pipeline_model.training is True
        m.pipeline_model.eval()                           # a model that was already in eval mode stays there
        r.render('epoch2', 2)
        assert m.pipeline_model.training is False
        print('test_eval_mode_during_sampling_then_restored OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_encode_prompts():
    s = _settings(cfg=1.0)
    out = pv.encode_prompts(FakeModel(), s)
    assert len(out) == 4 and all(u is None for _, u in out)
    s = _settings()
    assert all(u is not None for _, u in pv.encode_prompts(FakeModel(), s))
    print('test_encode_prompts OK')


if __name__ == '__main__':
    test_parse_defaults_and_off()
    test_parse_rejects_bad_values()
    test_assign_and_names()
    test_check_supported()
    test_to_pil()
    test_render_two_ranks_and_tensorboard()
    test_same_seed_same_image_across_saves()
    test_failures_never_raise()
    test_eval_mode_during_sampling_then_restored()
    test_encode_prompts()
    print('ALL previews tests passed')
