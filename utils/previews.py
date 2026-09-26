"""Training previews: render a few fixed prompts with the in-training weights at every save.

Opt-in through a [samples] table in the training TOML:

    [samples]
    prompts = ['a lighthouse on a cliff at dusk', 'an empty office at night']
    negative_prompt = ''        # used when cfg > 1
    width = 1024                # positive multiples of 16
    height = 1024
    steps = 30
    cfg = 4.0
    shift = 3.0                 # flow-matching schedule shift used for sampling
    seed = 42                   # prompt i always starts from seed + i, so every save is comparable
    before_first_step = true    # also render once before training (skipped when resuming)

Every save writes <run_dir>/samples/<save name>/NN_<slug>.png and a prompts.txt, and logs the images to
TensorBoard under samples/NN. The prompts are encoded once, right after caching, while the text encoder is
still loaded. Under data parallelism every rank holds the same weights, so the prompts are split across the
ranks. A failure while rendering is printed and skipped: previews never stop training.

Needs a model that implements get_conds() and vae_decode() (anima / cosmos_predict2 among others), text
embeddings computed by the pipeline's text encoders, and pipeline_stages = 1.

This module stays import-light (torch, numpy and PIL are imported where used) so it can be unit-tested on CPU.
"""
import os
import re
import time
import traceback

DEFAULTS = {
    'negative_prompt': '',
    'width': 1024,
    'height': 1024,
    'steps': 30,
    'cfg': 4.0,
    'shift': 3.0,
    'seed': 42,
    'before_first_step': True,
}
KNOWN_KEYS = set(DEFAULTS) | {'prompts'}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_samples_config(config):
    """The validated [samples] settings, or None when previews are off (no table, or no prompts)."""
    raw = config.get('samples')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('[samples] must be a TOML table')
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ValueError(f'[samples] unknown keys {sorted(unknown)}; known keys are {sorted(KNOWN_KEYS)}')
    prompts = raw.get('prompts')
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        return None
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError('[samples] prompts must be a list of non-empty strings')
    s = dict(DEFAULTS)
    s.update({k: v for k, v in raw.items() if k != 'prompts'})
    s['prompts'] = list(prompts)
    for k in ('width', 'height'):
        if not _is_int(s[k]) or s[k] <= 0 or s[k] % 16:
            raise ValueError(f'[samples] {k} must be a positive multiple of 16, got {s[k]!r}')
    if not _is_int(s['steps']) or s['steps'] < 1:
        raise ValueError(f"[samples] steps must be a positive integer, got {s['steps']!r}")
    if not _is_int(s['seed']):
        raise ValueError(f"[samples] seed must be an integer, got {s['seed']!r}")
    for k in ('cfg', 'shift'):
        if not _is_number(s[k]) or s[k] <= 0:
            raise ValueError(f'[samples] {k} must be a positive number, got {s[k]!r}')
        s[k] = float(s[k])
    if not isinstance(s['negative_prompt'], str):
        raise ValueError('[samples] negative_prompt must be a string')
    if not isinstance(s['before_first_step'], bool):
        raise ValueError('[samples] before_first_step must be true or false')
    return s


def check_supported(model, config):
    """Raise a clear error at startup when previews can't run for this model or layout."""
    model_type = config.get('model', {}).get('type', '?')
    missing = [m for m in ('get_conds', 'vae_decode', 'sample', 'encode_sample_prompt', 'set_sample_schedule')
               if not callable(getattr(model, m, None))]
    if missing:
        raise ValueError(f'[samples] previews are not supported for model type {model_type!r} '
                         f'(missing {", ".join(missing)}); remove the [samples] table')
    if config.get('pipeline_stages', 1) != 1:
        raise ValueError('[samples] previews need pipeline_stages = 1 (each rank must hold the whole model)')
    if len(model.get_text_encoders()) == 0:
        raise ValueError(f'[samples] previews need the text embeddings to come from the pipeline text encoders; '
                         f'model type {model_type!r} is configured without them (e.g. cache_text_embeddings = false)')


def assign_prompts(n, rank, world):
    """The prompt indices one rank renders: every world-th prompt, starting at its rank."""
    return [i for i in range(n) if i % world == rank]


def slugify(text, maxlen=40):
    s = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return s[:maxlen].rstrip('-') or 'prompt'


def image_filename(idx, prompt):
    return f'{idx:02d}_{slugify(prompt)}.png'


def encode_prompts(model, settings):
    """(conds, unconds) per prompt as CPU tensors. Call while the text encoders are still loaded."""
    return [model.encode_sample_prompt(p, settings['negative_prompt'], settings['cfg']) for p in settings['prompts']]


def to_pil(img):
    """A (1, H, W, C) or (H, W, C) image tensor with values in [0, 1] -> PIL.Image (RGB)."""
    import numpy as np
    from PIL import Image
    t = img[0] if img.ndim == 4 else img
    arr = (t.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


class PreviewRenderer:
    """Renders the [samples] prompts at each save. Call render() on every rank; it never raises."""

    def __init__(self, model, settings, conds, run_dir, tb_writer=None, rank=0, world=1, barrier=None,
                 device='cuda', disable_block_swap=False):
        assert len(conds) == len(settings['prompts'])
        self.model = model
        self.settings = settings
        self.conds = conds
        self.run_dir = str(run_dir)
        self.tb_writer = tb_writer
        self.rank = rank
        self.world = world
        self.barrier = barrier
        self.device = device
        self.disable_block_swap = disable_block_swap

    def _render_mine(self, out_dir, name):
        import torch
        from utils.isolate_rng import isolate_rng
        s = self.settings
        written = []
        swap_prepared = False
        try:
            self.model.prepare_block_swap_inference(disable_block_swap=self.disable_block_swap)
            swap_prepared = True
            self.model.set_sample_schedule(s['steps'], s['shift'])
            with torch.no_grad(), isolate_rng(include_cuda=self.device != 'cpu'):
                for i in assign_prompts(len(s['prompts']), self.rank, self.world):
                    try:
                        conds, unconds = self.conds[i]
                        gen = torch.Generator(device=self.device).manual_seed(s['seed'] + i)
                        img = self.model.sample(w=s['width'], h=s['height'], conds=conds, unconds=unconds,
                                                cfg=s['cfg'], generator=gen, show_progress=False)
                        path = os.path.join(out_dir, image_filename(i, s['prompts'][i]))
                        to_pil(img).save(path)
                        written.append(path)
                    except Exception:
                        print(f'[samples] {name}: prompt {i} failed on rank {self.rank}, skipped:\n{traceback.format_exc()}')
        except Exception:
            print(f'[samples] {name}: rendering failed on rank {self.rank}, skipped:\n{traceback.format_exc()}')
        finally:
            if swap_prepared:
                try:
                    self.model.prepare_block_swap_training()
                except Exception:
                    print(f'[samples] {name}: restoring training mode failed:\n{traceback.format_exc()}')
            if self.device != 'cpu':
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return written

    def render(self, name, step):
        t0 = time.time()
        out_dir = os.path.join(self.run_dir, 'samples', name)
        written = []
        try:
            os.makedirs(out_dir, exist_ok=True)
            if self.rank == 0:
                with open(os.path.join(out_dir, 'prompts.txt'), 'w', encoding='utf-8') as f:
                    s = self.settings
                    f.write(f"# step {step} | {s['width']}x{s['height']} | steps {s['steps']} | cfg {s['cfg']} | "
                            f"shift {s['shift']} | seed {s['seed']} + index\n")
                    if s['cfg'] > 1:
                        f.write(f"# negative: {s['negative_prompt']}\n")
                    for i, p in enumerate(s['prompts']):
                        f.write(f'{image_filename(i, p)}\t{p}\n')
            written = self._render_mine(out_dir, name)
        except Exception:
            print(f'[samples] {name}: failed on rank {self.rank}, skipped:\n{traceback.format_exc()}')
        if self.barrier is not None:
            self.barrier()
        if self.rank == 0:
            self._log_tensorboard(out_dir, step)
            n = sum(1 for f in os.listdir(out_dir) if f.endswith('.png')) if os.path.isdir(out_dir) else 0
            print(f"[samples] {name}: {n} of {len(self.settings['prompts'])} previews in {out_dir} "
                  f'({time.time() - t0:.0f} s)')
        return written

    def _log_tensorboard(self, out_dir, step):
        if self.tb_writer is None:
            return
        try:
            import numpy as np
            from PIL import Image
            for i, p in enumerate(self.settings['prompts']):
                path = os.path.join(out_dir, image_filename(i, p))
                if os.path.exists(path):  # other ranks' files exist on a single node; skip what isn't visible
                    self.tb_writer.add_image(f'samples/{i:02d}', np.asarray(Image.open(path).convert('RGB')),
                                             step, dataformats='HWC')
            self.tb_writer.flush()
        except Exception:
            print(f'[samples] TensorBoard logging failed, skipped:\n{traceback.format_exc()}')
