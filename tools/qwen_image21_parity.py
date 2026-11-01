"""Qwen-Image-2.1 parity check on a CUDA card with the real weights: the training code path against ComfyUI's own.

Every check prints one ledger row (PASS / FAIL / INFO / SKIP) with its numbers; a failing check never stops the
run, and nothing is written to disk. The checks:
  text encoder  a batch of captions through the caption-caching function (padded to the longest caption) against
                ComfyUI encoding each caption alone
  vae           the training encode (pixels in [-1, 1], RGB) against a plain ComfyUI VAE (pixels in [0, 1])
  layer chain   to_layers() against ComfyUI's QwenImage21 forward on the same latents, timestep and text, both under
                bf16 autocast; beside the run-to-run spread, the inference-dtype forward, and (on cards with at least
                48 GB) the bf16-vs-fp32 spread of the forward itself
  padding       the shorter caption's row in a padded batch against the same row alone
  lora step     one checkpointed forward/backward at 512 px with a rank-4 LoRA: finite loss, every LoRA gets a gradient;
                then the same step with cfg = 4 (the distillation-preserving loss, empty caption as the uncond)

Run it from the repository (python tools/qwen_image21_parity.py) or paste the whole file into one notebook cell.
Set the paths below (or the DIFFUSION_PIPE_DIR / QWEN_IMAGE21_DIR environment variables). Memory: the transformer in
bf16 (14.2 GB) plus activations; the text encoder and the VAE are unloaded before the transformer is loaded.
"""
import gc
import os
import platform
import sys
import time
import traceback

REPO = os.environ.get('DIFFUSION_PIPE_DIR') or (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else '/content/diffusion-pipe')
MODELS = os.environ.get('QWEN_IMAGE21_DIR', '/content/models/qwen_image_2.1')  # the Comfy-Org/Qwen-Image-2.1 layout
DIFFUSION_MODEL = os.path.join(MODELS, 'diffusion_models', 'qwen_image_2.1_bf16.safetensors')
VAE_PATH = os.path.join(MODELS, 'vae', 'qwen_image_2.1_vae_bf16.safetensors')
TEXT_ENCODER = os.path.join(MODELS, 'text_encoders', 'qwen3vl_8b_int8_convrot.safetensors')
IMAGE = None         # path to a picture to encode; None = a synthetic picture
RESOLUTION = 1024
TIMESTEP = 0.6
SEED = 0
CAPTIONS = [
    'a red fox',
    'a red fox asleep under an old oak tree in fresh snow, early morning light, shallow depth of field',
    '{"subjects": [{"type": "woman", "description": "a woman in a yellow raincoat holding a black umbrella"}], '
    '"setting": "a rainy city street at night", "style": "photograph"}',
]

sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _stub_deepspeed():
    # deepspeed is only needed for training; it does not install on Windows
    try:
        import deepspeed  # noqa: F401
        return
    except Exception:
        pass
    import importlib.machinery
    import logging
    import types

    def stub(name, **attrs):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    ds = stub('deepspeed')
    ds.comm = stub('deepspeed.comm')
    ds.comm.comm = stub('deepspeed.comm.comm', get_rank=lambda: 0, get_world_size=lambda: 1,
                        is_initialized=lambda: False, barrier=lambda *args, **kwargs: None)
    ds.utils = stub('deepspeed.utils')
    ds.utils.logging = stub('deepspeed.utils.logging', logger=logging.getLogger('deepspeed'))


_stub_deepspeed()
import utils.common  # noqa: E402
utils.common.AUTOCAST_DTYPE = torch.bfloat16  # as train.py sets it from [model] dtype, before the models import
import utils.dataset  # noqa: E402,F401  (train.py's import order: the dataset code before the models)
from models import qwen_image21  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

LEDGER = []


def record(check, status, detail):
    LEDGER.append((check, status, detail))
    print(f'[{status}] {check}: {detail}', flush=True)


def rel(a, b):
    a, b = a.detach().float(), b.detach().float().to(a.device)
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def cos(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten().to(a.device)
    return float(F.cosine_similarity(a, b, dim=0))


def run_check(name, fn):
    try:
        fn()
        return True
    except Exception as e:
        record(name, 'FAIL', f'{type(e).__name__}: {e}')
        traceback.print_exc()
        return False


def release(wrapper):
    # a model the remaining checks do not need: off the card and out of RAM
    comfy.model_management.unload_all_models()
    wrapper._model = None
    gc.collect()
    torch.cuda.empty_cache()


def picture(size):
    if IMAGE is not None:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(IMAGE).convert('RGB')
        img = TF.center_crop(TF.resize(img, size), [size, size])
        return TF.to_tensor(img)[None] * 2 - 1
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    return torch.stack([xx, yy, torch.sin(6 * (xx ** 2 + yy ** 2))])[None].clamp(-1, 1)


def chain(layers, inputs, checkpoint=False):
    h = layers[0](inputs)
    for layer in layers[1:-1]:
        h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False) if checkpoint else layer(h)
    return layers[-1](h)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - t0) * 1000


if platform.system() == 'Windows':
    torch.cuda.set_per_process_memory_fraction(0.73)  # a WDDM desktop card spills past its memory instead of failing
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
config = {
    'model': {
        'type': 'qwen_image21',
        'diffusion_model': DIFFUSION_MODEL,
        'vae': VAE_PATH,
        'text_encoders': [{'path': TEXT_ENCODER, 'type': 'qwen_image'}],
        'dtype': torch.bfloat16,
        'timestep_sample_method': 'logit_normal',
    },
    'pipeline_stages': 1,
    'reentrant_activation_checkpointing': False,
}
pipe = qwen_image21.QwenImage21Pipeline(config)
state = {}


def check_text_encoder():
    te = pipe.get_text_encoders()[0]
    te.load_model_if_needed()
    fn = pipe.get_call_text_encoder_fn(te)
    out, ms = timed(lambda: fn(CAPTIONS, is_video=[False] * len(CAPTIONS)))
    emb_b, mask_b = out['text_embeds_0'].cpu(), out['attention_mask_0'].cpu()
    alone = []
    for caption in CAPTIONS + ['']:
        o = te.encode_from_tokens_scheduled(te.tokenize(caption))
        alone.append(o[0][0][0].to(torch.bfloat16).cpu())  # (L, 4096); one caption has no padding
    state['emb_empty'] = alone.pop()  # the empty caption, the unconditional branch of cfg training
    lengths = [a.shape[0] for a in alone]
    ok = mask_b.sum(dim=1).tolist() == lengths and emb_b.shape[1] == max(lengths)
    worst_rel = max(rel(emb_b[i, :lengths[i]], alone[i]) for i in range(len(CAPTIONS)))
    worst_cos = min(cos(emb_b[i, :lengths[i]], alone[i]) for i in range(len(CAPTIONS)))
    record('text encoder', 'PASS' if ok and worst_cos >= 0.999 else 'FAIL',
           f'token lengths {lengths}, padded batch {tuple(emb_b.shape)}; batch vs alone worst relative {worst_rel:.2e}, '
           f'worst cosine {worst_cos:.6f} (batch shapes round differently); batch of {len(CAPTIONS)} in {ms:.0f} ms')
    state.update(emb_batch=emb_b, mask_batch=mask_b, emb_alone=alone)
    release(te)


def check_vae():
    vae = pipe.get_vae()
    vae.load_model_if_needed()
    plain = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_PATH))  # ComfyUI's own crop + RGBA pad
    for size in (RESOLUTION, 512):
        pixels = picture(size).to('cuda', torch.bfloat16)
        lat, ms = timed(lambda: pipe.vae_encode(pixels))
        ref = pipe.latent_format.process_in(plain.encode(((pixels + 1) / 2).movedim(1, -1)))
        r = rel(lat, ref)
        record(f'vae {size}px', 'PASS' if r == 0.0 else 'FAIL',
               f'latents {tuple(lat.shape)}; training encode vs ComfyUI VAE relative {r:.2e}; encode {ms:.0f} ms')
        state[f'latents_{size}'] = lat.float().cpu()
    del plain
    release(vae)


def load_transformer():
    _, ms = timed(pipe.load_diffusion_model)
    pipe.diffusion_model.to('cuda')
    params = sum(p.numel() for p in pipe.diffusion_model.parameters())
    record('load', 'INFO', f'{params / 1e9:.2f}B parameters through the training loader in {ms / 1000:.1f} s; '
                           f'{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated')


def noisy(latents, generator_seed):
    g = torch.Generator(device='cpu').manual_seed(generator_seed)
    x0 = latents.cuda()
    noise = torch.randn(x0.shape, generator=g).cuda()
    t = torch.full((x0.shape[0],), TIMESTEP, device='cuda')
    return (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise, t


def check_layer_chain():
    model = pipe.diffusion_model.eval()
    layers = pipe.to_layers()
    x_t, t = noisy(state[f'latents_{RESOLUTION}'], SEED)
    ctx = state['emb_alone'][1][None].cuda()
    mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device='cuda')
    v_chain, ms = timed(lambda: chain(layers, (x_t, t, ctx, mask)))
    v_again = chain(layers, (x_t, t, ctx, mask))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        v_ref = model(x_t, t, ctx, transformer_options={})
    v_infer = model(x_t.to(torch.bfloat16), t, ctx.to(torch.bfloat16), transformer_options={})  # as ComfyUI samples
    parity, repeat, infer = rel(v_chain, v_ref), rel(v_again, v_chain), rel(v_infer, v_ref)
    bar = None
    if torch.cuda.get_device_properties(0).total_memory >= 48 * 2**30:
        dtypes = {n: p.dtype for n, p in model.named_parameters()}
        model.float()
        v_fp32 = model(x_t, t, ctx.float(), transformer_options={})
        for n, p in model.named_parameters():
            p.data = p.data.to(dtypes[n])  # bf16 -> fp32 -> bf16 is exact
        bar = rel(v_ref, v_fp32)
    status = 'PASS' if (parity <= 0.5 * bar if bar is not None else parity <= 1e-2) else 'FAIL'
    record('layer chain', status,
           f'chain vs ComfyUI forward relative {parity:.2e} (cosine {cos(v_chain, v_ref):.6f}); run-to-run {repeat:.2e}; '
           f'bf16-vs-fp32 spread of the forward {"%.2e" % bar if bar is not None else "not measured (card < 48 GB)"}; '
           f'chain forward at {RESOLUTION}px {ms:.0f} ms')
    record('inference dtype', 'INFO',
           f'ComfyUI forward with bf16 latents and timestep (as at sampling) vs under autocast: relative {infer:.2e}')
    state['v_ref'] = v_ref


def check_padding():
    model = pipe.diffusion_model.eval()
    layers = pipe.to_layers()
    lengths = state['mask_batch'].sum(dim=1).tolist()
    short, long = lengths.index(min(lengths)), lengths.index(max(lengths))
    x_t, t = noisy(state[f'latents_{RESOLUTION}'], SEED)
    x2, t2 = x_t.repeat(2, 1, 1, 1), t.repeat(2)
    ctx = state['emb_batch'][[short, long]].cuda()
    mask = state['mask_batch'][[short, long]].cuda().bool()
    v_pad = chain(layers, (x2, t2, ctx, mask))[:1]
    alone_ctx = state['emb_alone'][short][None].cuda()
    alone_mask = torch.ones(alone_ctx.shape[:2], dtype=torch.bool, device='cuda')
    v_two_alone = chain(layers, (x2, t2, alone_ctx.repeat(2, 1, 1), alone_mask.repeat(2, 1)))[:1]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        v_alone = model(x_t, t, alone_ctx, transformer_options={})
    padded, batch_rounding = rel(v_pad, v_alone), rel(v_two_alone, v_alone)
    record('padding', 'PASS' if padded <= max(2 * batch_rounding, 1e-2) else 'FAIL',
           f'row with {min(lengths)} of {max(lengths)} text tokens in a padded batch vs alone: relative {padded:.2e} '
           f'(cosine {cos(v_pad, v_alone):.6f}); same row in an unpadded batch of 2 vs alone {batch_rounding:.2e}')


def check_lora_step():
    pipe.configure_adapter({'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.bfloat16})
    model = pipe.diffusion_model.train()
    inputs = {
        'latents': state['latents_512'],
        'mask': None,
        'text_embeds_0': [state['emb_alone'][1]],
        'attention_mask_0': [torch.ones(state['emb_alone'][1].shape[0], dtype=torch.int64)],
    }
    lora_b = [(n, p) for n, p in model.named_parameters() if 'lora_B' in n]
    for cfg in (1.0, 4.0):
        # cfg 4: the distillation-preserving loss, with the empty caption as the unconditional branch
        pipe.cfg = cfg
        pipe.uncond_dict = {'text_embeds_0': state['emb_empty'],
                            'attention_mask_0': torch.ones(state['emb_empty'].shape[0], dtype=torch.int64)}
        layers = pipe.to_layers()
        torch.manual_seed(SEED)
        features, (target, _) = pipe.prepare_inputs(inputs, timestep_quantile=0.5)
        features = tuple(f.cuda() for f in features)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            def step():
                out = chain(layers, features, checkpoint=True)
                loss = pipe.get_loss_fn()(out, (target.cuda(), torch.tensor([])))
                loss.backward()
                return loss
            loss, ms = timed(step)
        with_grad = sum(1 for _, p in lora_b if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        ok = torch.isfinite(loss).item() and with_grad == len(lora_b) == 6 * len(model.transformer_blocks)
        record('lora step' if cfg == 1 else f'cfg {cfg:g} step', 'PASS' if ok else 'FAIL',
               f'loss {loss.item():.4f}; {with_grad} of {len(lora_b)} LoRA modules got a finite non-zero gradient; '
               f'forward+backward at 512px {ms:.0f} ms, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
    pipe.cfg = 1.0


run_check('text encoder', check_text_encoder)
run_check('vae', check_vae)
loaded = run_check('load', load_transformer)
for name, fn in (('layer chain', check_layer_chain), ('padding', check_padding), ('lora step', check_lora_step)):
    needs = ('emb_alone', f'latents_{RESOLUTION}', 'latents_512')
    if not loaded or any(k not in state for k in needs):
        record(name, 'SKIP', 'an earlier step failed (see above)')
        continue
    run_check(name, fn)

print('\nLEDGER')
for check, status, detail in LEDGER:
    print(f'  {status:4s}  {check:16s} {detail}')
print(f'{sum(s == "PASS" for _, s, _ in LEDGER)} PASS, {sum(s == "FAIL" for _, s, _ in LEDGER)} FAIL, '
      f'{sum(s == "SKIP" for _, s, _ in LEDGER)} SKIP')
