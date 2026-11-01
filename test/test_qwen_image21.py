"""CPU checks for Qwen-Image-2.1 (models/qwen_image21.py and the ComfyPipeline code it relies on).

Runnable standalone on CPU (no pytest, no GPU, no model files):
    python test/test_qwen_image21.py

Needs the ComfyUI submodule at v0.43.0 or newer (comfy/ldm/qwen_image21) and its Python dependencies.
deepspeed is stubbed when it cannot be imported (e.g. on Windows); nothing here trains.

Covers: the model wrappers importing against the real ComfyUI tree; the training layer chain against
ComfyUI's own QwenImage21 forward on a small random model (fp32 and fp64); a padded batch row against the
same row alone; LoRA targeting (all block Linears, exclude_modules) and gradient flow into every LoRA;
the RGBA channel padding of the VAE input; text-encoder batching with the 2.1 tokenizer's keep_vision key.
"""
import importlib
import importlib.machinery
import logging
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def _stub_deepspeed():
    try:
        import deepspeed  # noqa: F401
        return False
    except Exception:
        pass

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
    return True


DEEPSPEED_STUBBED = _stub_deepspeed()

import torch  # noqa: E402
import utils.common  # noqa: E402
utils.common.AUTOCAST_DTYPE = torch.float32  # train.py sets this from [model] dtype; CUDA autocast is off on CPU anyway

if not torch.cuda.is_available():
    # ComfyUI picks its device and attention function at import. Without CUDA it must be told to run on the CPU
    # (its --cpu flag), and to use PyTorch SDPA as it does on NVIDIA cards: its default CPU attention (sub-quadratic)
    # adds boolean masks instead of applying them, so no padding would be masked.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submodules', 'ComfyUI'))
    import comfy.cli_args  # noqa: E402
    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.use_pytorch_cross_attention = True

# Same order as train.py: the dataset code (and pyarrow) first, then the models. On Windows, pyarrow.dataset
# loaded after transformers' modeling code crashes the interpreter.
import utils.dataset  # noqa: E402,F401
from models import base  # noqa: E402
from models import qwen_image21  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.ldm.qwen_image21.model as qwen21_model  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 1))


def _pipeline(**model_config):
    # CommonPipeline.__init__ puts the sampling scheduler on CUDA; skip it here (sampling is not tested).
    original_init = base.CommonPipeline.__init__
    base.CommonPipeline.__init__ = lambda self, *args, **kwargs: None
    try:
        config = {
            'model': {
                'type': 'qwen_image21',
                'diffusion_model': 'unused.safetensors',
                'vae': 'unused.safetensors',
                'text_encoders': [{'path': 'unused.safetensors', 'type': 'qwen_image'}],
                'dtype': torch.float32,
                **model_config,
            },
            'pipeline_stages': 1,
        }
        pipe = qwen_image21.QwenImage21Pipeline(config)
    finally:
        base.CommonPipeline.__init__ = original_init
    return pipe


def _tiny_model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    model = qwen21_model.QwenImage21Transformer2DModel(
        in_channels=16, out_channels=16, num_layers=2, attention_head_dim=32, num_attention_heads=2,
        context_in_dim=24, mlp_ratio=3, axes_dims_rope=(8, 12, 12), dtype=dtype, device='cpu',
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1:
                p.normal_(0, 1 / math.sqrt(p.shape[-1]))
            else:
                p.normal_(0, 0.1)
    return model.eval()


def _inputs(batch=2, txt_len=7, h=6, w=5, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 16, h, w, generator=g, dtype=dtype)
    t = torch.rand(batch, generator=g, dtype=dtype)
    context = torch.randn(batch, txt_len, 24, generator=g, dtype=dtype)
    mask = torch.ones(batch, txt_len, dtype=torch.bool)
    return x, t, context, mask


def _chain(pipe, *inputs):
    h = inputs
    for layer in pipe.to_layers():
        h = layer(h)
    return h


def _reference(model, x, t, context):
    # ComfyUI's own forward: no padding support, one text segment with a causal mask, the target image last
    return model(x, t, context, transformer_options={})


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_imports():
    # the other wrappers import models/base.py (and so the ComfyUI tree) at module level
    for name in ('models.cosmos_predict2', 'models.sana', 'models.sdxl', 'utils.dataset'):
        importlib.import_module(name)
    # utils/patches.py needs deepspeed's pipeline engine; check the ComfyUI names it imports instead
    from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, apply_mod  # noqa: F401
    from comfy.ldm.flux.math import attention  # noqa: F401
    import comfy.sd1_clip
    assert comfy.sd1_clip.ClipTokenWeightEncoder.encode_token_weights is base.encode_token_weights
    assert comfy.sd1_clip.SD1Tokenizer.tokenize_with_weights is base.tokenize_with_weights
    assert hasattr(comfy.sd.CLIPType, 'QWEN_IMAGE')
    print(f'test_imports OK (deepspeed stubbed: {DEEPSPEED_STUBBED})')


def test_layer_chain_matches_comfy_forward():
    pipe = _pipeline()
    with torch.no_grad():
        for dtype in (torch.float32, torch.float64):
            model = _tiny_model(dtype=dtype)
            pipe.diffusion_model = model
            x, t, context, mask = _inputs(dtype=dtype)
            out = _chain(pipe, x, t, context, mask)
            ref = _reference(model, x, t, context)
            assert out.shape == ref.shape == x.shape, (out.shape, ref.shape)
            rel = _rel(out, ref)
            print(f'  {str(dtype):14s} chain vs ComfyUI forward: max |diff| / max |ref| = {rel:.2e}')
            assert rel < (1e-5 if dtype == torch.float32 else 1e-12), rel
    print('test_layer_chain_matches_comfy_forward OK')


def test_padded_row_matches_unpadded():
    # A batch pads every caption to the longest one (attention_mask 0 on the padding). The padded row must
    # come out as it does alone, which is how ComfyUI runs it at inference.
    pipe = _pipeline()
    with torch.no_grad():
        model = _tiny_model(dtype=torch.float64)
        pipe.diffusion_model = model
        x, t, context, mask = _inputs(batch=2, txt_len=9, dtype=torch.float64)
        short = 4
        mask[0, short:] = False
        context[0, short:] = 0.0
        out = _chain(pipe, x, t, context, mask)
        alone = _reference(model, x[:1], t[:1], context[:1, :short])
        rel = _rel(out[:1], alone)
        print(f'  padded row (4 of 9 text tokens) vs the same row alone, fp64: {rel:.2e}')
        assert rel < 1e-12, rel
    print('test_padded_row_matches_unpadded OK')


def test_prepare_inputs_variable_length_text():
    pipe = _pipeline()
    latents = torch.randn(2, 16, 6, 5)
    inputs = {
        'latents': latents,
        'mask': None,
        'text_embeds_0': [torch.randn(4, 24), torch.randn(9, 24)],
        'attention_mask_0': [torch.ones(4, dtype=torch.int64), torch.ones(9, dtype=torch.int64)],
    }
    (noisy, t, context, mask), (target, loss_mask) = pipe.prepare_inputs(inputs)
    assert noisy.shape == target.shape == latents.shape
    assert context.shape == (2, 9, 24) and mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [4, 9]
    assert ((t > 0) & (t < 1)).all() and loss_mask is None
    print('test_prepare_inputs_variable_length_text OK')


def _lora_targets(model):
    return sorted(name for name, module in model.named_modules() if hasattr(module, 'lora_A'))


def test_lora_targets_and_gradients():
    adapter = {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.float32}

    pipe = _pipeline()
    pipe.diffusion_model = _tiny_model()
    pipe.configure_adapter(dict(adapter))
    targets = _lora_targets(pipe.diffusion_model)
    per_block = ['attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'img_mlp.gate_up', 'img_mlp.out']
    expected = sorted(f'transformer_blocks.{i}.{n}' for i in range(2) for n in per_block)
    assert targets == expected, targets

    # every LoRA sits in the forward path: the zero-init B matrices all receive gradient
    pipe.diffusion_model.train()
    x, t, context, mask = _inputs()
    out = _chain(pipe, x, t, context, mask)
    out.square().mean().backward()
    for name, p in pipe.diffusion_model.named_parameters():
        if 'lora_B' in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name

    pipe = _pipeline()
    pipe.diffusion_model = _tiny_model()
    pipe.configure_adapter({**adapter, 'exclude_modules': ['img_mlp.gate_up', 'img_mlp.out']})
    targets = _lora_targets(pipe.diffusion_model)
    assert targets == sorted(t for t in expected if '.attn.' in t), targets
    print(f'test_lora_targets_and_gradients OK ({len(expected)} targets; {len(targets)} with the feed-forward excluded)')


def test_cfg_training():
    # cfg = w: the output fed to the loss is (out + (w - 1) * out_uncond) / w, the unconditional branch (empty
    # caption) run through the same layers without gradient; without the uncond inputs (sampling) the output is plain.
    w = 4.0
    with torch.no_grad():
        model = _tiny_model(dtype=torch.float64)
        x, t, context, mask = _inputs(batch=2, txt_len=9, dtype=torch.float64)
        g = torch.Generator().manual_seed(3)
        uncond = torch.randn(2, 3, 24, generator=g, dtype=torch.float64)
        uncond_mask = torch.ones(2, 3, dtype=torch.bool)
        pipe = _pipeline(cfg=w)
        pipe.diffusion_model = model
        out = _chain(pipe, x, t, context, mask, uncond, uncond_mask)
        expected = (_reference(model, x, t, context) + (w - 1) * _reference(model, x, t, uncond)) / w
        rel = _rel(out, expected)
        plain = _rel(_chain(pipe, x, t, context, mask), _reference(model, x, t, context))
    print(f'  cfg {w:g} chain vs (forward + {w - 1:g} * uncond forward) / {w:g}, fp64: {rel:.2e}; sampling path: {plain:.2e}')
    assert rel < 1e-12 and plain < 1e-12, (rel, plain)

    # the uncond branch carries no gradient: every LoRA gradient is exactly 1/w of the plain chain's
    adapter = {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.float64}
    pipe_plain = _pipeline()
    pipe_plain.diffusion_model = model
    pipe_plain.configure_adapter(adapter)
    pipe.diffusion_model = model
    params = [p for n, p in model.named_parameters() if 'lora_' in n]
    weights = torch.randn(x.shape, generator=g, dtype=torch.float64)
    grads_plain = torch.autograd.grad((_chain(pipe_plain, x, t, context, mask) * weights).sum(), params)
    grads_cfg = torch.autograd.grad((_chain(pipe, x, t, context, mask, uncond, uncond_mask) * weights).sum(), params)
    worst = max(_rel(gc * w, gp) for gc, gp in zip(grads_cfg, grads_plain) if gp.abs().max() > 0)
    print(f'  LoRA gradients, cfg chain x {w:g} vs plain chain: worst relative {worst:.2e}')
    assert worst < 1e-10, worst

    # prepare_inputs appends the cached empty-caption conds, one copy per row
    pipe = _pipeline(cfg=w)
    pipe.uncond_dict = {'text_embeds_0': torch.randn(3, 24), 'attention_mask_0': torch.ones(3, dtype=torch.int64)}
    inputs = {'latents': torch.randn(2, 16, 6, 5), 'mask': None,
              'text_embeds_0': [torch.randn(4, 24), torch.randn(9, 24)],
              'attention_mask_0': [torch.ones(4, dtype=torch.int64), torch.ones(9, dtype=torch.int64)]}
    features, _ = pipe.prepare_inputs(inputs)
    assert len(features) == 6 and features[4].shape == (2, 3, 24) and features[5].dtype == torch.bool
    assert torch.equal(features[4][0], pipe.uncond_dict['text_embeds_0'])
    print('test_cfg_training OK')


class _FakeVAE:
    def __init__(self, sd=None):
        self.crop_input = True
        self.output_channels = 4
        self.pad_channel_value = 1.0

    def throw_exception_if_invalid(self):
        pass

    def spacial_compression_encode(self):
        return 16


def test_vae_rgba_padding():
    pipe = _pipeline()
    original = (comfy.utils.load_torch_file, comfy.sd.VAE)
    comfy.utils.load_torch_file = lambda *args, **kwargs: {}
    comfy.sd.VAE = _FakeVAE
    try:
        pipe.vae.load_model_if_needed()
    finally:
        comfy.utils.load_torch_file, comfy.sd.VAE = original
    vae = pipe.vae._model

    pixels = torch.rand(1, 40, 36, 3)  # (b, h, w, c): cropped to multiples of 16, centred
    out = vae.vae_encode_crop_pixels(pixels)
    assert out.shape == (1, 32, 32, 4), out.shape
    assert torch.equal(out[..., :3], pixels[:, 4:36, 2:34, :]) and (out[..., 3] == 1.0).all()

    vae.output_channels, vae.pad_channel_value, vae.crop_input = 3, None, False  # an RGB VAE: unchanged
    assert torch.equal(vae.vae_encode_crop_pixels(pixels), pixels)
    print('test_vae_rgba_padding OK')


class _FakeTextEncoder:
    """The real 2.1 tokenizer; an encoder that returns the token ids as the embedding, so padding is visible."""

    def __init__(self):
        import comfy.text_encoders.qwen_image21
        self.tokenizer = comfy.text_encoders.qwen_image21.QwenImage21Tokenizer()
        self.pad = getattr(self.tokenizer, self.tokenizer.clip).pad_token
        self.keep_vision_seen = []

    def tokenize(self, text):
        return self.tokenizer.tokenize_with_weights(text)

    def encode_from_tokens_scheduled(self, tokens):
        self.keep_vision_seen.append(tokens.get('keep_vision'))
        rows = tokens[self.tokenizer.clip]
        assert len({len(r) for r in rows}) == 1, 'rows must be padded to one length'
        ids = torch.tensor([[tok for tok, *_ in r] for r in rows])
        return [[ids.unsqueeze(-1).float(), {'attention_mask': (ids != self.pad).long()}]]


def test_text_encoder_batching():
    pipe = _pipeline()
    te = _FakeTextEncoder()
    pipe.text_encoders = [te]
    fn = pipe.get_call_text_encoder_fn(te)
    captions = ['a red fox', 'a red fox sleeping under a large oak tree in the snow, early morning light']
    out = fn(captions, is_video=[False, False])
    emb, mask = out['text_embeds_0'], out['attention_mask_0']
    lengths = [len(te.tokenize(c)[te.tokenizer.clip][0]) for c in captions]
    assert emb.shape[:2] == mask.shape == (2, max(lengths)), (emb.shape, mask.shape, lengths)
    assert mask.sum(dim=1).tolist() == lengths
    alone = [t for t, *_ in te.tokenize(captions[0])[te.tokenizer.clip][0]]
    assert emb[0, :lengths[0], 0].long().tolist() == alone
    assert te.keep_vision_seen == [False]
    print(f'test_text_encoder_batching OK (token lengths {lengths})')


ALL_TESTS = [
    test_imports,
    test_layer_chain_matches_comfy_forward,
    test_padded_row_matches_unpadded,
    test_prepare_inputs_variable_length_text,
    test_lora_targets_and_gradients,
    test_cfg_training,
    test_vae_rgba_padding,
    test_text_encoder_batching,
]

if __name__ == '__main__':
    for test in ALL_TESTS:
        test()
    print(f'\nAll {len(ALL_TESTS)} qwen_image21 tests passed.')
