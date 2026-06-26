"""Unit tests for utils.image_resize.

Validates:
- fit_crop_box matches PIL ImageOps.fit's center cover-crop geometry.
- gpu_resize_batch shape/dtype/range, frame-dim handling, and that its output is
  close to the CPU convert_crop_and_resize + ToTensor + Normalize baseline (not
  bit-exact: torch vs PIL bicubic differ slightly, which is why GPU resize uses a
  separate cache fingerprint).

Runs on CPU (device='cpu'); depends only on torch + torchvision + pillow + numpy.
    python test/test_image_resize.py
"""

import os
import sys

import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.image_resize import (
    composite_to_rgb, convert_crop_and_resize, pil_to_uint8_chw,
    fit_crop_box, gpu_resize_batch,
)


def _cpu_baseline(pil_img, tw, th):
    # Mirror PreprocessMediaFile's CPU path: ImageOps.fit -> ToTensor -> Normalize(0.5,0.5)
    cropped = convert_crop_and_resize(pil_img, (tw, th))
    t = TF.to_tensor(cropped)  # [0,1], (C,H,W)
    return t * 2 - 1  # [-1,1]


def test_fit_crop_box_matches_pil():
    # For several source sizes and a square target, our crop box should equal the
    # region PIL's ImageOps.fit crops (center, target aspect ratio).
    for (W, H) in [(512, 512), (800, 400), (400, 800), (1024, 768), (37, 100)]:
        tw, th = 256, 256
        top, left, ch, cw = fit_crop_box(W, H, tw, th)
        # target aspect ratio preserved in the crop
        assert abs((cw / ch) - (tw / th)) < 0.02, (W, H, cw, ch)
        # crop fits inside the source and is centered
        assert 0 <= left and left + cw <= W
        assert 0 <= top and top + ch <= H
        assert abs(left - (W - cw) // 2) <= 1 and abs(top - (H - ch) // 2) <= 1
    print('test_fit_crop_box_matches_pil OK')


def test_gpu_resize_shape_and_range():
    imgs = [
        pil_to_uint8_chw(Image.new('RGB', (640, 480), (10, 20, 30))),
        pil_to_uint8_chw(Image.new('RGB', (200, 400), (200, 100, 50))),
        pil_to_uint8_chw(Image.new('RGB', (512, 512), (0, 0, 0))),
    ]
    out = gpu_resize_batch(imgs, 256, 256, add_frame_dim=False, device='cpu')
    assert out.shape == (3, 3, 256, 256), out.shape
    assert out.dtype == torch.float32
    assert out.min() >= -1.0001 and out.max() <= 1.0001
    # video VAEs: extra frame dim
    out_v = gpu_resize_batch(imgs, 256, 256, add_frame_dim=True, device='cpu')
    assert out_v.shape == (3, 3, 1, 256, 256), out_v.shape
    print('test_gpu_resize_shape_and_range OK')


def test_gpu_close_to_cpu_baseline():
    # Solid colors must be reproduced (almost) exactly by both paths.
    for color in [(10, 20, 30), (250, 5, 128)]:
        pil = Image.new('RGB', (480, 640), color)
        u8 = pil_to_uint8_chw(pil)
        gpu = gpu_resize_batch([u8], 256, 256, add_frame_dim=False, device='cpu')[0]
        cpu = _cpu_baseline(pil, 256, 256)
        assert torch.allclose(gpu, cpu, atol=2 / 255), (gpu.mean().item(), cpu.mean().item())

    # Natural-ish gradient: paths should be close on average (bicubic differs at edges).
    import numpy as np
    grad = np.zeros((300, 500, 3), dtype=np.uint8)
    grad[..., 0] = np.linspace(0, 255, 500, dtype=np.uint8)[None, :]
    grad[..., 1] = np.linspace(0, 255, 300, dtype=np.uint8)[:, None]
    pil = Image.fromarray(grad)
    u8 = pil_to_uint8_chw(pil)
    gpu = gpu_resize_batch([u8], 256, 256, add_frame_dim=False, device='cpu')[0]
    cpu = _cpu_baseline(pil, 256, 256)
    mae = (gpu - cpu).abs().mean().item()
    assert mae < 0.02, f'mean abs diff too high: {mae}'  # ~ a few /255
    print(f'test_gpu_close_to_cpu_baseline OK (gradient MAE={mae:.4f})')


def test_composite_alpha_to_white():
    # RGBA with transparent region must composite to white, not black.
    rgba = Image.new('RGBA', (10, 10), (255, 0, 0, 0))  # fully transparent red
    rgb = composite_to_rgb(rgba)
    assert rgb.mode == 'RGB'
    assert rgb.getpixel((0, 0)) == (255, 255, 255), rgb.getpixel((0, 0))
    print('test_composite_alpha_to_white OK')


ALL_TESTS = [
    test_fit_crop_box_matches_pil,
    test_gpu_resize_shape_and_range,
    test_gpu_close_to_cpu_baseline,
    test_composite_alpha_to_white,
]

if __name__ == '__main__':
    for t in ALL_TESTS:
        t()
    print(f'\nAll {len(ALL_TESTS)} image_resize tests passed.')
