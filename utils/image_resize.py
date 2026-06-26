"""Image preprocessing helpers (CPU compositing + GPU batched resize).

Kept dependency-light (torch + torchvision + PIL + numpy, NO deepspeed/comfy) so the
CPU/GPU resize equivalence can be unit-tested without the full training stack.
"""

import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def composite_to_rgb(pil_img):
    '''Convert any PIL image to RGB, compositing transparency onto a white
    background (so alpha images don't turn black). Shared by the CPU resize path
    and the GPU resize path so they agree on pixels before resizing.'''
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        return canvas.convert('RGB')
    return pil_img.convert('RGB')


def convert_crop_and_resize(pil_img, width_and_height):
    return ImageOps.fit(composite_to_rgb(pil_img), width_and_height)


def pil_to_uint8_chw(pil_img):
    '''RGB PIL image -> (3, H, W) uint8 CPU tensor (no resize, no normalize).'''
    arr = np.asarray(pil_img, dtype=np.uint8)  # (H, W, 3)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()


def fit_crop_box(W, H, target_w, target_h):
    '''Center cover-crop box (top, left, height, width) replicating PIL
    ImageOps.fit (bleed=0, centering=0.5): crop to the target aspect ratio, then
    the caller resizes that crop to the target size.'''
    output_ratio = target_w / target_h
    if (W / H) >= output_ratio:
        crop_h = H
        crop_w = int(round(output_ratio * H))
    else:
        crop_w = W
        crop_h = int(round(W / output_ratio))
    crop_w = max(1, min(crop_w, W))
    crop_h = max(1, min(crop_h, H))
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    return top, left, crop_h, crop_w


def gpu_resize_batch(uint8_list, target_w, target_h, add_frame_dim, device):
    '''Resize+center-crop a list of variable-size uint8 (C,H,W) CPU tensors to a
    common (target_h, target_w) on `device`, then normalize to [-1, 1]. Mirrors the
    CPU convert_crop_and_resize + ToTensor + Normalize path (bicubic, antialias).
    Returns (B, C, H, W), or (B, C, 1, H, W) when add_frame_dim (video VAEs).'''
    out = []
    for img in uint8_list:
        img = img.to(device, non_blocking=True)
        _, h, w = img.shape
        top, left, ch, cw = fit_crop_box(w, h, target_w, target_h)
        r = TF.resized_crop(
            img, top, left, ch, cw, [target_h, target_w],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        out.append(r)
    batch = torch.stack(out).float().div_(255).mul_(2).sub_(1)  # [-1, 1]
    if add_frame_dim:
        batch = batch.unsqueeze(2)  # (B, C, 1, H, W) for video VAEs
    return batch
