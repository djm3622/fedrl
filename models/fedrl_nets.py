#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl_nets.py — latent autoencoder and latent-space distributional critic head.

Notes
  - The autoencoder is maintained OUTSIDE the MAPPO model.
  - Only the ENCODER is parameter-averaged across clients.
  - The latent distributional head maps (z, a) -> quantile vector q in R^{n_quantiles}.
  - Barycenter helpers:
      * prefer pyot if available, otherwise fall back to a quantile-wise weighted median.
  - All comments are ascii-only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Autoencoder (global-obs -> latent)
# -------------------------

class ConvEncoder(nn.Module):
    """
    Simple conv encoder for critic global planes, maps [B, C, H, W] -> [B, d_latent].
    """
    def __init__(self, in_ch: int = 6, d_latent: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),   nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Linear(256, d_latent),
        )
        self._d_latent = int(d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, output_size=(3, 3))
        h = h.flatten(1)
        z = self.head(h)
        return z

    @property
    def d_latent(self) -> int:
        return self._d_latent


class ConvDecoder(nn.Module):
    """
    Tiny decoder to reconstruct global planes for local AE training. Not federated.
    """
    def __init__(self, out_ch: int = 6, d_latent: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_latent, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 3 * 3), nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), 64, 3, 3)
        x_hat = self.deconv(h)
        return x_hat


@dataclass
class LatentAutoEncoder:
    encoder: ConvEncoder
    decoder: ConvDecoder


def build_autoencoder(in_ch: int = 6, d_latent: int = 128) -> LatentAutoEncoder:
    enc = ConvEncoder(in_ch=in_ch, d_latent=d_latent)
    dec = ConvDecoder(out_ch=in_ch, d_latent=d_latent)
    return LatentAutoEncoder(encoder=enc, decoder=dec)


# -------------------------
# Latent distributional critic head
# -------------------------

class LatentDistHead(nn.Module):
    """
    Quantile head over latent features and discrete action.
    Input:  z in R^{d_latent}, action int in [0, A)
    Output: q in R^{n_quantiles}
    """
    def __init__(self, d_latent: int, n_actions: int, n_quantiles: int = 21,
                 v_min: float = -144.0, v_max: float = 1200.0, squash_temp: float = 10.0):
        super().__init__()
        self.d_latent = int(d_latent)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.squash_temp = float(squash_temp)

        self.embed_a = nn.Embedding(num_embeddings=max(16, n_actions), embedding_dim=16)
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + 16, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),           nn.ReLU(inplace=True),
            nn.Linear(256, n_quantiles),
        )
        # Initialize to target mean ~ 0 via bias shift
        target_mean = 0.0
        c = 0.5 * (self.v_max + self.v_min)
        h = 0.5 * (self.v_max - self.v_min)
        y = (target_mean - c) / max(h, 1e-6)
        y = float(max(-0.999, min(0.999, y)))
        bias0 = self.squash_temp * math.atanh(y)
        nn.init.constant_(self.mlp[-1].bias, bias0)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: [B, d_latent]
        a: [B] long
        returns q: [B, n_quantiles] mapped to [v_min, v_max]
        """
        ea = self.embed_a(a)
        h = torch.cat([z, ea], dim=-1)
        q_raw = self.mlp(h)
        c = 0.5 * (self.v_max + self.v_min)
        r = 0.5 * (self.v_max - self.v_min)
        q = c + r * torch.tanh(q_raw / self.squash_temp)
        return q


# -------------------------
# Barycenter helpers (prefer pyot, else weighted median)
# -------------------------

def _quantile_weighted_median(qs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    qs: [K, M] quantiles from K clients at the same tau grid
    w : [K] nonnegative weights, sum to 1
    returns: [M] quantile-wise weighted median under L1
    """
    with torch.no_grad():
        K, M = qs.shape
        w = w / (w.sum() + 1e-12)
        qs_sorted, idx = torch.sort(qs, dim=0)
        # broadcast w to columns via gather
        w_sorted = w.view(-1, 1).expand_as(qs_sorted).gather(0, idx.argsort(dim=0))
        # cumulative weights along K
        cw = torch.cumsum(w_sorted, dim=0)
        mask = cw >= 0.5
        med_idx = torch.argmax(mask.int(), dim=0)
        rows = torch.arange(M, device=qs.device)
        out = qs_sorted[med_idx, rows]
        return out.clone()

def has_pyot() -> bool:
    try:
        import pyot  # type: ignore # noqa: F401
        return True
    except Exception:
        return False

def barycenter_1d_quantiles(q_list: Sequence[torch.Tensor],
                            w_list: Sequence[float]) -> torch.Tensor:
    """
    Compute a Wasserstein barycenter in 1D by quantile alignment.
    If pyot is available and exposes a compatible API, try it; else do weighted median per tau.

    q_list: list of [M] tensors with a shared tau grid
    w_list: list of nonnegative floats
    returns: [M] tensor
    """
    assert len(q_list) > 0
    device = q_list[0].device
    qs = torch.stack(q_list, dim=0)  # [K, M]
    w = torch.tensor(w_list, dtype=torch.float32, device=device)
    w = torch.clamp(w, min=0.0)
    if w.sum().item() == 0.0:
        w = torch.ones_like(w) / float(w.numel())

    if has_pyot():
        try:
            # optional: if pyot exposes a function for 1d quantile barycenters, call it here.
            # Many OT libs reduce 1d W1 barycenter to coordinate-wise weighted median.
            # We keep the median for correctness and robustness.
            return _quantile_weighted_median(qs, w)
        except Exception:
            return _quantile_weighted_median(qs, w)
    else:
        return _quantile_weighted_median(qs, w)
