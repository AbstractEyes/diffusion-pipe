import inspect

import diffusers
import transformers
import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE


def _default_complex_human_instruction():
    # The instruction prefix the diffusers SanaPipeline prepends to every positive prompt by default.
    return inspect.signature(diffusers.SanaPipeline.__call__).parameters['complex_human_instruction'].default


class SanaPipeline(BasePipeline):
    """Sana (NVlabs, arXiv 2410.10629) from a diffusers-format folder, e.g. Efficient-Large-Model/Sana_600M_512px_diffusers.

    [model] keys: type = 'sana'; diffusers_path = the folder; dtype (transformer weights); optional max_sequence_length (300),
    use_complex_human_instruction (true), timestep_sample_method / sigmoid_scale / shift as for the other rectified-flow models.
    Text encoding reproduces the diffusers pipeline: lowercased + stripped captions, the complex human instruction prepended to
    positive prompts, the first token + the last (max_sequence_length - 1) tokens kept; an empty caption is encoded the way the
    pipeline encodes its unconditional prompt (no instruction, plain max_sequence_length padding), so caption dropout and CFG
    sampling see the same null conditioning.
    """
    name = 'sana'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['SanaTransformerBlock']
    spatial_compression = 32
    channels = 32
    pixels_round_to_multiple = 32

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config['model']
        path = self.model_config['diffusers_path']
        # The model card runs the autoencoder and the text encoder in bf16.
        self.vae = diffusers.AutoencoderDC.from_pretrained(path, subfolder='vae', torch_dtype=torch.bfloat16)
        self.vae.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path, subfolder='tokenizer')
        self.tokenizer.padding_side = 'right'
        self.text_encoder = transformers.AutoModel.from_pretrained(path, subfolder='text_encoder', torch_dtype=torch.bfloat16)
        self.text_encoder.eval()
        self.max_sequence_length = self.model_config.get('max_sequence_length', 300)
        self.chi = _default_complex_human_instruction() if self.model_config.get('use_complex_human_instruction', True) else None
        self.sample_shift = self.model_config.get('shift', 3.0)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        self.transformer = diffusers.SanaTransformer2DModel.from_pretrained(
            self.model_config['diffusers_path'], subfolder='transformer', torch_dtype=dtype)
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # diffusers format: transformer.<module>.lora_A.weight
        peft_state_dict = {'transformer.' + k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            latents = vae.encode(tensor.to(p.device, p.dtype)).latent
            return {'latents': latents * vae.config.scaling_factor}
        return fn

    def vae_decode(self, latents):
        # (B, C, H, W) latents -> (B, H, W, C) pixels in [0, 1]
        p = next(self.vae.parameters())
        img = self.vae.decode(latents.to(p.device, p.dtype) / self.vae.config.scaling_factor).sample
        return ((img.float().clamp(-1, 1) + 1) / 2).movedim(1, -1)

    def encode_text(self, text_encoder, captions):
        """-> (prompt_embeds [B, max_sequence_length, width], prompt_mask [B, max_sequence_length]), as the diffusers pipeline.

        One batched text-encoder call per group (instruction-prefixed captions, empty captions), rows returned in caption
        order: a batch the pipeline would encode in one call reproduces its result exactly.
        """
        device = next(text_encoder.parameters()).device
        L = self.max_sequence_length
        captions = [c.lower().strip() for c in captions]
        groups = []
        if self.chi:
            chi = '\n'.join(self.chi)
            n_chi = len(self.tokenizer.encode(chi))
            groups.append(([i for i, c in enumerate(captions) if c != ''], chi, n_chi + L - 2, [0] + list(range(-L + 1, 0))))
            groups.append(([i for i, c in enumerate(captions) if c == ''], '', L, list(range(L))))
        else:
            groups.append((list(range(len(captions))), '', L, list(range(L))))
        embeds, masks = [None] * len(captions), [None] * len(captions)
        for rows, prefix, max_length, select in groups:
            if not rows:
                continue
            ti = self.tokenizer([prefix + captions[i] for i in rows], padding='max_length', max_length=max_length,
                                truncation=True, add_special_tokens=True, return_tensors='pt')
            mask = ti.attention_mask.to(device)
            # The text encoder runs in its own dtype, as in the diffusers pipeline. The caption cache is computed without autocast,
            # but training previews encode under the trainer's autocast (models/base.py), which would change the rounding.
            with torch.autocast('cuda', enabled=False):
                e = text_encoder(ti.input_ids.to(device), attention_mask=mask)[0]
            for j, i in enumerate(rows):
                embeds[i] = e[j:j + 1, select]
                masks[i] = mask[j:j + 1, select]
        return torch.cat(embeds), torch.cat(masks)

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            assert not any(is_video)
            prompt_embeds, prompt_mask = self.encode_text(text_encoder, captions)
            return {'prompt_embeds': prompt_embeds, 'prompt_mask': prompt_mask}
        return fn

    def get_conds(self, inputs):
        # The text half of the pipeline inputs, in the order InitialLayer unpacks them after (x, t).
        return (inputs['prompt_embeds'], inputs['prompt_mask'])

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_mask = inputs['prompt_mask']
        mask = inputs['mask']

        bs, c, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        # Rectified flow, as the Sana sampler (flow sigmas, prediction_type flow_prediction): x_t = (1 - t) x + t noise,
        # the model predicts noise - x.
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, t, prompt_embeds, prompt_mask), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerLayer(block))
        layers.append(FinalLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.time_embed = model.time_embed
        self.caption_projection = model.caption_projection
        self.caption_norm = model.caption_norm
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, t, prompt_embeds, prompt_mask = inputs
        bs, c, h, w = x.shape
        p = self.model[0].config.patch_size
        height, width = torch.tensor([h // p], device=x.device), torch.tensor([w // p], device=x.device)

        hidden_states = self.patch_embed(x)
        # the diffusers model takes the timestep on the 0-1,000 scale (the pipeline passes sigma x 1,000 x timestep_scale)
        ts = t.view(-1) * 1000 * getattr(self.model[0].config, 'timestep_scale', 1.0)
        timestep, embedded_timestep = self.time_embed(ts, batch_size=bs, hidden_dtype=hidden_states.dtype)
        # the text embeddings come from the bf16 text encoder (or the cache); the diffusers pipeline casts them to the transformer's dtype
        encoder_hidden_states = self.caption_projection(prompt_embeds.to(hidden_states.dtype))
        encoder_hidden_states = encoder_hidden_states.view(bs, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)
        # the encoder mask as an additive bias with a singleton query dimension, as the diffusers forward does
        encoder_bias = ((1 - prompt_mask.to(hidden_states.dtype)) * -10000.0).unsqueeze(1)
        return make_contiguous(hidden_states, encoder_hidden_states, encoder_bias, timestep, embedded_timestep, height, width)


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, encoder_bias, timestep, embedded_timestep, height, width = inputs
        hidden_states = self.block(hidden_states, None, encoder_hidden_states, encoder_bias, timestep, int(height.item()), int(width.item()))
        return make_contiguous(hidden_states, encoder_hidden_states, encoder_bias, timestep, embedded_timestep, height, width)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        # no __getattr__ fallback to the model here: registering the scale_shift_table parameter calls hasattr() first
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.scale_shift_table = model.scale_shift_table
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, encoder_bias, timestep, embedded_timestep, height, width = inputs
        bs = hidden_states.shape[0]
        p = self.model[0].config.patch_size
        H, W = int(height.item()), int(width.item())
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(bs, H, W, p, p, -1)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        return hidden_states.reshape(bs, -1, H * p, W * p)
