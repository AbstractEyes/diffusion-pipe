"""Aleph relay adapters for diffusion trunks (runner-2 line, feat/aleph-adapter).

RelayPatch2D: per-block residual relay — the certified RelayPatchwork recipe
(frozen-trunk relay adapters; multi-slot closed-form aleph read; zero-init out
weight AND bias; near-zero scalar gate) applied to DiT hidden states of any
leading shape [..., d]. Toggle contract: enabled=False returns x untouched
(code-path skip) => all-off is bit-exact with the stock model.

Wiring (cosmos_predict2 / anima):
  [model] aleph_relay = true            # attach to every block
          aleph_relay_every = 1         # or 2 for the half-stack arm
          aleph_relay_lr = 1e-3         # its param-group lr (freeze trunk via
                                        # base/self_attn/cross_attn/mlp/mod = 0)
          aleph_relay_path = '...'      # optional: resume a saved stack
The blocks apply the relay at the very end of Block.forward via a getattr hook
(no-op when the attribute is absent). Params live at blocks.N.aleph_relay.* so
original_name stamping, param-group bucketing, and saving all work unchanged.

Optimizer: pure Adam, wd=0 on these params ([optimizer] type = 'adam' — added
in this branch; never 'adamw' on aleph paths).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = math.sqrt(2.0)
PSI = 1.533751168755204288118041


def super_fibonacci_s3(n: int, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64)
    s = (i + 0.5) / n
    r = torch.sqrt(s)
    R = torch.sqrt(1.0 - s)
    alpha = 2.0 * math.pi * i / PHI
    beta = 2.0 * math.pi * i / PSI
    q = torch.stack([r * torch.sin(alpha), r * torch.cos(alpha),
                     R * torch.sin(beta), R * torch.cos(beta)], dim=-1)
    return F.normalize(q, dim=-1).to(dtype)


class AlephAddress(nn.Module):
    """Closed-form soft read over 2K oriented half-axes [+A;-A]:
    m_hat = Sum_k sinh(u_k)A_k / Sum_k cosh(u_k), u = cos(x,A)/tau, computed
    stably via max-|u| factor-out. No argmax/softmax-routing/VQ/EMA anywhere —
    the codebook trains only through the consumer's gradient."""

    def __init__(self, K: int = 64, D: int = 4, tau: float = 0.1,
                 init: str = "random"):
        super().__init__()
        self.K, self.D, self.tau = K, D, tau
        if init == "fibonacci":
            assert D == 4
            A = super_fibonacci_s3(K)
        else:
            A = F.normalize(torch.randn(K, D), dim=-1)
        self.codebook = nn.Parameter(A)

    def m_hat(self, x: torch.Tensor) -> torch.Tensor:
        A = F.normalize(self.codebook, dim=-1)
        u = (F.normalize(x, dim=-1) @ A.transpose(-1, -2)) / self.tau
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        return ((ep - en) @ A) / (ep + en).sum(dim=-1, keepdim=True)


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class RelayPatch2D(nn.Module):
    def __init__(self, d: int, n_slots: int = 16, K: int = 64,
                 tau: float = 0.1, hidden: int = 178, init: str = "random"):
        super().__init__()
        self.d, self.n_slots, self.hidden = d, n_slots, hidden
        self.proj = nn.Linear(d, n_slots * 4, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(K, 4, tau, init)
        self.consume = nn.Sequential(
            nn.Linear(n_slots * 4, hidden), SquaredReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, d))
        nn.init.zeros_(self.consume[-1].weight)   # zero-init out: weight
        nn.init.zeros_(self.consume[-1].bias)     # AND bias (bias-leak law)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.enabled = True

    def assert_zero_init(self):
        assert self.consume[-1].weight.abs().max().item() == 0.0
        assert self.consume[-1].bias.abs().max().item() == 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        lead = x.shape[:-1]
        slots = self.proj(x).view(*lead, self.n_slots, 4)
        feats = self.addr.m_hat(slots).reshape(*lead, self.n_slots * 4)
        return x + torch.sigmoid(self.gate) * self.consume(feats)

    @classmethod
    def from_state_dict(cls, sd: dict) -> "RelayPatch2D":
        d = sd["proj.weight"].shape[1]
        n_slots = sd["proj.weight"].shape[0] // 4
        hidden = sd["consume.0.weight"].shape[0]
        K = sd["addr.codebook"].shape[0]
        m = cls(d, n_slots=n_slots, K=K, hidden=hidden)
        m.load_state_dict(sd, strict=True)
        return m


def attach_aleph_relays(transformer, model_channels: int, every: int = 1,
                        relay_path: str | None = None,
                        dtype: torch.dtype | None = None) -> int:
    """Attach RelayPatch2D to transformer.blocks[i] for i % every == 0, AFTER
    the DiT is materialized (never under init_empty_weights — fresh relays are
    not in the trunk state dict and would be left on meta). Returns site count.
    relay_path: optional saved stack {'relays': {block_idx: state_dict}}.
    DTYPE LAW (Phil 2026-07-16): adapter dtype MATCHES the trunk block dtype —
    an fp32 adapter on a bf16 trunk spins fp32 noise into the environment;
    test at the model's actual fp/bf sizes. Gauges stay fp32 downstream."""
    saved = None
    if relay_path is not None:
        saved = torch.load(relay_path, map_location="cpu", weights_only=True)
        saved = {int(k): v for k, v in saved["relays"].items()}
    n = 0
    for i, block in enumerate(transformer.blocks):
        if i % every != 0:
            continue
        if saved is not None:
            relay = RelayPatch2D.from_state_dict(saved[i])
        else:
            relay = RelayPatch2D(model_channels)
            relay.assert_zero_init()
        # dtype law: use the DECLARED trunk dtype (sniffing the first
        # param can read a still-meta fp32 tensor — caught by the R0b gate)
        block_dtype = dtype or next(block.parameters()).dtype
        block.aleph_relay = relay.to(block_dtype)
        n += 1
    return n


def aleph_relay_state(transformer) -> dict:
    """Collect {'relays': {block_idx: state_dict}} for adapter-only saving."""
    out = {}
    for i, block in enumerate(transformer.blocks):
        r = getattr(block, "aleph_relay", None)
        if r is not None:
            out[str(i)] = {k: v.detach().cpu() for k, v in
                           r.state_dict().items()}
    return {"relays": out}
