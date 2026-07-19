"""Aleph adapters for diffusion trunks — the amoe-lora bake-in.

This module is now a THIN SHIM over the amoe-lora package
(https://huggingface.co/AbstractPhil/amoe-lora, `pip install
amoe-lora[diffusion]`): when amoe is installed its certified classes are
used and checkpoint formats unify with the whole aleph line; when it is
not, the vendored fallback below keeps the fork fully standalone.

Two adapter modes, selected by [model] aleph_relay_mode:
  'relay'       (default) — RelayPatch2D per block: the certified
                per-block residual relay (relay-favorable 3-for-3
                substrates: Lune flow / SD15-core eps / Anima DiT flow).
  'multiband3'  — MultibandDelta per block: three band experts combined
                by cosine crossfade windows on the sigma axis (surgical
                band lesions, 2-seed). Band edges/xfade are LAW
                constants (exp008), not config keys. The pipeline sets
                per-micro-batch windows from the sampled timesteps.

Wiring (cosmos_predict2 / anima / sdxl):
  [model] aleph_relay = true
          aleph_relay_mode = 'relay'    # or 'multiband3'
          aleph_relay_rank = 16         # multiband3 per-band rank
          aleph_relay_every = 1         # or 2 for the half-stack arm
          aleph_relay_lr = 1e-3         # freeze trunk via the *_lr = 0 buckets
          aleph_relay_path = '...'      # optional: resume a saved stack
Optimizer: pure Adam, wd=0 ([optimizer] type = 'adam'; never 'adamw' on
aleph paths — train.py enforces this when aleph_relay is configured).

DTYPE LAW (Phil 2026-07-16): adapter dtype MATCHES the DECLARED trunk
dtype — an fp32 adapter on a bf16 trunk spins fp32 noise into the
environment, and sniffing the first param can read a still-meta fp32
tensor (the R0b gate catch). Gauges stay fp32 downstream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                        # the bake-in: prefer amoe-lora
    from amoe.diffusion.core.relay import RelayPatch2D
    from amoe.diffusion.core.multiband import MultibandDelta, band_weights
    from amoe.diffusion.laws import BAND_EDGES, N_BANDS, XFADE
    HAVE_AMOE = True
except ImportError:                         # vendored fallback (standalone)
    HAVE_AMOE = False
    N_BANDS = 3
    BAND_EDGES = (0.35, 0.75)
    XFADE = 0.06

    class _SquaredReLU(nn.Module):
        def forward(self, x):
            return F.relu(x) ** 2

    class _AlephAddress(nn.Module):
        """Closed-form soft read over 2K oriented half-axes [+A;-A]:
        m_hat = Sum_k sinh(u_k)A_k / Sum_k cosh(u_k), u = cos(x,A)/tau.
        No argmax/softmax-routing/VQ/EMA anywhere. Carries the `home`
        init snapshot (drift gauge) as of the amoe format."""

        def __init__(self, K: int = 64, D: int = 4, tau: float = 0.1):
            super().__init__()
            self.K, self.D, self.tau = K, D, tau
            A = F.normalize(torch.randn(K, D), dim=-1)
            self.codebook = nn.Parameter(A)
            self.register_buffer("home", A.detach().clone())

        def m_hat(self, x: torch.Tensor) -> torch.Tensor:
            A = F.normalize(self.codebook, dim=-1)
            u = (F.normalize(x, dim=-1) @ A.transpose(-1, -2)) / self.tau
            m = u.abs().amax(dim=-1, keepdim=True)
            ep, en = torch.exp(u - m), torch.exp(-u - m)
            return ((ep - en) @ A) / (ep + en).sum(dim=-1, keepdim=True)

    class RelayPatch2D(nn.Module):
        def __init__(self, d: int, n_slots: int = 16, K: int = 64,
                     tau: float = 0.1, hidden: int = 178):
            super().__init__()
            self.d, self.n_slots, self.hidden = d, n_slots, hidden
            self.proj = nn.Linear(d, n_slots * 4, bias=False)
            nn.init.orthogonal_(self.proj.weight)
            self.addr = _AlephAddress(K, 4, tau)
            self.consume = nn.Sequential(
                nn.Linear(n_slots * 4, hidden), _SquaredReLU(),
                nn.LayerNorm(hidden), nn.Linear(hidden, d))
            nn.init.zeros_(self.consume[-1].weight)   # zero-init: weight
            nn.init.zeros_(self.consume[-1].bias)     # AND bias (leak law)
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
            sd = dict(sd)
            d = sd["proj.weight"].shape[1]
            n_slots = sd["proj.weight"].shape[0] // 4
            hidden = sd["consume.0.weight"].shape[0]
            K = sd["addr.codebook"].shape[0]
            if "addr.home" not in sd:      # pre-0.2 fork saves: rebuild,
                sd["addr.home"] = F.normalize(   # drift gauge void — loud
                    sd["addr.codebook"].detach().float(), dim=-1)
                print("aleph_relay: addr.home reconstructed from codebook "
                      "(pre-amoe save; drift gauge starts here)")
            m = cls(d, n_slots=n_slots, K=K, hidden=hidden)
            m.load_state_dict(sd, strict=True)
            return m

    def band_weights(s01: torch.Tensor) -> torch.Tensor:
        def ramp(x):
            t = ((x / XFADE).clamp(-1, 1) + 1) / 2
            return 0.5 - 0.5 * torch.cos(t * math.pi)
        e1, e2 = BAND_EDGES
        up1, up2 = ramp(s01 - e1), ramp(s01 - e2)
        return torch.stack([1 - up1, up1 * (1 - up2), up1 * up2], dim=-1)

    class MultibandDelta(nn.Module):
        needs_bands = True

        def __init__(self, d: int, r: int = 16):
            super().__init__()
            self.d, self.r = d, r
            self.down = nn.ModuleList(nn.Linear(d, r, bias=False)
                                      for _ in range(N_BANDS))
            self.up = nn.ModuleList(nn.Linear(r, d) for _ in range(N_BANDS))
            for dn, up in zip(self.down, self.up):
                nn.init.orthogonal_(dn.weight)
                nn.init.zeros_(up.weight)
                nn.init.zeros_(up.bias)
            self.gates = nn.Parameter(torch.full((N_BANDS,), -3.0))
            self.enabled = True
            self.band_enabled = [True] * N_BANDS

        def assert_zero_init(self):
            for up in self.up:
                assert up.weight.abs().max().item() == 0.0
                assert up.bias.abs().max().item() == 0.0

        def forward(self, x, w_bands):
            if not self.enabled:
                return x
            if w_bands is None:
                raise RuntimeError("multiband3 needs per-batch band "
                                   "windows (set from the sampled t)")
            g = torch.sigmoid(self.gates)
            delta = 0
            w = w_bands.view(w_bands.shape[0],
                             *([1] * (x.ndim - 2)), N_BANDS)
            for b in range(N_BANDS):
                if not self.band_enabled[b]:
                    continue
                delta = delta + g[b] * w[..., b:b + 1] * self.up[b](
                    self.down[b](x))
            return x + delta if not isinstance(delta, int) else x

        @classmethod
        def from_state_dict(cls, sd: dict) -> "MultibandDelta":
            d = sd["down.0.weight"].shape[1]
            r = sd["down.0.weight"].shape[0]
            m = cls(d, r=r)
            m.load_state_dict(sd, strict=True)
            return m


# amoe's MultibandDelta predates the fork's needs_bands convention;
# stamp it so the Block.forward hook can dispatch uniformly.
if not hasattr(MultibandDelta, "needs_bands"):
    MultibandDelta.needs_bands = True


def _make(mode: str, d: int, rank: int):
    if mode == "relay":
        m = RelayPatch2D(d)
    elif mode == "multiband3":
        m = MultibandDelta(d, r=rank)
    else:
        raise ValueError(f"unknown aleph_relay_mode '{mode}' "
                         "(use 'relay' or 'multiband3')")
    m.assert_zero_init()
    return m


def _from_sd(mode: str, sd: dict):
    return (RelayPatch2D if mode == "relay"
            else MultibandDelta).from_state_dict(sd)


def _load_stack(relay_path: str) -> tuple[dict, str]:
    """Accept the fork-legacy {'relays': {idx: sd}} shape AND the amoe
    'amoe.diffusion.anchor' v1 (.pt) format."""
    saved = torch.load(relay_path, map_location="cpu", weights_only=True)
    if "relays" in saved:
        stack = {int(k): v for k, v in saved["relays"].items()}
        mode = ("multiband3" if "down.0.weight" in next(iter(stack.values()))
                else "relay")
        return stack, mode
    if saved.get("format") == "amoe.diffusion.anchor":
        per: dict[int, dict] = {}
        for k, v in saved["adapters"].items():
            i, param = k.split(".", 1)
            per.setdefault(int(i), {})[param] = v
        kind = saved.get("meta", {}).get("adapter", {}).get("kind", "relay")
        return per, kind
    raise ValueError(f"unrecognized aleph stack at {relay_path}")


def attach_aleph_relays(transformer, model_channels: int, every: int = 1,
                        relay_path: str | None = None,
                        dtype: torch.dtype | None = None,
                        mode: str = "relay", rank: int = 16) -> int:
    """Attach adapters to transformer.blocks[i] for i % every == 0, AFTER
    the DiT is materialized (never under init_empty_weights — fresh
    adapters are not in the trunk state dict and would be left on meta).
    Returns site count. DTYPE LAW: pass the DECLARED trunk dtype."""
    saved, saved_mode = (None, None)
    if relay_path is not None:
        saved, saved_mode = _load_stack(relay_path)
        if saved_mode != mode:
            print(f"aleph_relay: saved stack is '{saved_mode}', overriding "
                  f"configured mode '{mode}'")
            mode = saved_mode
    n = 0
    for i, block in enumerate(transformer.blocks):
        if i % every != 0:
            continue
        if saved is not None:
            relay = _from_sd(mode, saved[i])
        else:
            relay = _make(mode, model_channels, rank)
        # dtype law: DECLARED trunk dtype (never sniff a maybe-meta param)
        block_dtype = dtype or next(block.parameters()).dtype
        block.aleph_relay = relay.to(block_dtype)
        n += 1
    return n


class UNetBlockWithAleph(nn.Module):
    """Wrap one diffusers BasicTransformerBlock; the adapter reads its
    hidden-states output. The attribute is NAMED aleph_relay so param
    original_name stamping, the get_param_groups bucket, and adapter-only
    saves all key on the same 'aleph_relay' substring as the DiT path."""

    def __init__(self, block: nn.Module, relay: nn.Module):
        super().__init__()
        self.block = block
        self.aleph_relay = relay

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.aleph_relay(out[0]),) + out[1:]
        return self.aleph_relay(out)


def attach_unet_relays(unet, relay_path: str | None = None,
                       dtype: torch.dtype | None = None,
                       mode: str = "relay", rank: int = 16) -> int:
    """SDXL/SD15 path: wrap every BasicTransformerBlock (site enumeration
    is asserted non-empty; the SDXL site COUNT is pinned at the first real
    run — recorded, never guessed). Relay mode only for now: the UNet
    layer split does not yet plumb per-micro-batch band windows
    (documented-deferred; use the cosmos/anima path or amoe.diffusion
    natively for multiband)."""
    if mode != "relay":
        raise NotImplementedError(
            "aleph_relay_mode='multiband3' on the SDXL path is deferred — "
            "the UNet pipeline layers do not yet carry band windows. "
            "Use mode='relay' here, or the cosmos/anima path.")
    from diffusers.models.attention import BasicTransformerBlock
    sites = [(name, mod, mod.norm1.normalized_shape[0])
             for name, mod in unet.named_modules()
             if isinstance(mod, BasicTransformerBlock)]
    assert sites, "no BasicTransformerBlocks found in this UNet"
    saved = None
    if relay_path is not None:
        saved, saved_mode = _load_stack(relay_path)
        assert saved_mode == "relay", saved_mode
        assert len(saved) == len(sites), (len(saved), len(sites))
    for i, (name, block, d) in enumerate(sites):
        relay = (_from_sd("relay", saved[i]) if saved is not None
                 else _make("relay", d, rank))
        block_dtype = dtype or next(block.parameters()).dtype
        wrap = UNetBlockWithAleph(block, relay.to(block_dtype))
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = wrap
        else:
            setattr(parent, last, wrap)
    print(f"aleph(unet): {len(sites)} BasicTransformerBlock sites, "
          f"widths {sorted(set(s[2] for s in sites))}")
    return len(sites)


def unet_aleph_state(unet, meta: dict | None = None) -> dict:
    """Adapter-only save for the UNet path — same dual-shape blob as
    aleph_relay_state (fork-legacy {'relays'} + amoe anchor v1 fields)."""
    relays, adapters, widths, names = {}, {}, [], []
    j = 0
    for name, mod in unet.named_modules():
        if isinstance(mod, UNetBlockWithAleph):
            sd = {k: v.detach().cpu()
                  for k, v in mod.aleph_relay.state_dict().items()}
            relays[str(j)] = sd
            for k, v in sd.items():
                adapters[f"{j}.{k}"] = v
            widths.append(getattr(mod.aleph_relay, "d", None))
            names.append(name)
            j += 1
    m = {"adapter": {"kind": "relay"},
         "substrate": {"family": "sdxl_unet", "n_sites": j,
                       "site_names": names, "widths": widths},
         "imported_from": "diffusion_pipe"}
    if meta:
        m.update(meta)
    return {"format": "amoe.diffusion.anchor", "version": 1, "meta": m,
            "adapters": adapters, "relays": relays}


def aleph_relay_state(transformer, meta: dict | None = None) -> dict:
    """Adapter-only save. Emits BOTH shapes in one blob: the fork-legacy
    {'relays': {idx: sd}} (resume path, older tools) and the amoe
    'amoe.diffusion.anchor' v1 fields (format/meta/adapters) so the same
    file loads through amoe.diffusion.load_diffusion_anchor unchanged."""
    relays, adapters, kind, widths, names = {}, {}, "relay", [], []
    j = 0
    for i, block in enumerate(transformer.blocks):
        r = getattr(block, "aleph_relay", None)
        if r is None:
            continue
        sd = {k: v.detach().cpu() for k, v in r.state_dict().items()}
        relays[str(i)] = sd
        for k, v in sd.items():
            adapters[f"{j}.{k}"] = v
        kind = "multiband3" if getattr(r, "needs_bands", False) else "relay"
        widths.append(getattr(r, "d", None))
        names.append(f"blocks.{i}")
        j += 1
    m = {"adapter": {"kind": kind},
         "substrate": {"family": "cosmos_dit", "n_sites": j,
                       "site_names": names, "widths": widths},
         "imported_from": "diffusion_pipe"}
    if meta:
        m.update(meta)
    return {"format": "amoe.diffusion.anchor", "version": 1, "meta": m,
            "adapters": adapters, "relays": relays}
