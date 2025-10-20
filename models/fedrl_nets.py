#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl_nets.py — spatial-latent autoencoder and (unchanged) latent-space critic head.

Changes:
  - Encoder now outputs a SPATIAL latent z: [B, C_latent, n, n] (no big bottleneck).
  - Decoder consumes that spatial z and reconstructs critic planes at exactly 10x10.
  - build_autoencoder keeps the same signature; d_latent is interpreted as latent channels.
  - Optionally choose latent spatial size via latent_hw (defaults to 10).

Notes:
  - This AE is still external to MAPPO (side-car) unless you wire it into the critic.
  - All comments are ascii-only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Autoencoder with spatial latent (global-obs -> z[B, C_latent, n, n])
# -------------------------

class ConvEncoder(nn.Module):
    """
    Spatial encoder for critic global planes.
    Input : x [B, C_in, H, W] (H=W=10 typically)
    Output: z [B, C_latent, n, n] where n=latent_hw (default 10)
    """
    def __init__(self, in_ch: int = 6, d_latent: int = 32, latent_hw: int = 10):
        super().__init__()
        self.in_ch = int(in_ch)
        self.latent_ch = int(d_latent)      # interpret d_latent as # of latent channels
        self.latent_hw = int(latent_hw)     # spatial size of latent feature map (n x n)

        # Light conv stem that preserves spatial resolution (stride 1)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.latent_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        returns z: [B, C_latent, n, n]
        """
        # If input is not the latent target size, resample gently (no pooling bottleneck)
        if x.size(-1) != self.latent_hw or x.size(-2) != self.latent_hw:
            x = F.interpolate(x, size=(self.latent_hw, self.latent_hw),
                              mode="bilinear", align_corners=False)
        z = self.stem(x)
        return z


class ConvDecoder(nn.Module):
    """
    Decoder from spatial latent back to critic planes.
    Input : z [B, C_latent, n, n]
    Output: x_hat [B, C_out, 10, 10]
    """
    def __init__(self, out_ch: int = 6, d_latent: int = 32, latent_hw: int = 10, target_hw: int = 10):
        super().__init__()
        self.out_ch = int(out_ch)
        self.latent_ch = int(d_latent)
        self.latent_hw = int(latent_hw)
        self.target_hw = int(target_hw)     # final reconstruction size (10)

        self.head = nn.Sequential(
            nn.Conv2d(self.latent_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C_latent, n, n]
        returns x_hat: [B, C_out, 10, 10]
        """
        h = z
        # If latent spatial size != target, upsample with mild interpolation
        if h.size(-1) != self.target_hw or h.size(-2) != self.target_hw:
            h = F.interpolate(h, size=(self.target_hw, self.target_hw),
                              mode="bilinear", align_corners=False)
        x_hat = self.head(h)
        return x_hat


@dataclass
class LatentAutoEncoder:
    encoder: ConvEncoder
    decoder: ConvDecoder


def build_autoencoder(in_ch: int = 6, d_latent: int = 32, latent_hw: int = 10) -> LatentAutoEncoder:
    """
    Backwards-compatible builder.
    - 'd_latent' now means #latent channels (C_latent).
    - 'latent_hw' controls spatial latent size n (default 10).
    """
    enc = ConvEncoder(in_ch=in_ch, d_latent=d_latent, latent_hw=latent_hw)
    dec = ConvDecoder(out_ch=in_ch, d_latent=d_latent, latent_hw=latent_hw, target_hw=10)
    return LatentAutoEncoder(encoder=enc, decoder=dec)