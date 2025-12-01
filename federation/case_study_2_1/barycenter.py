#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barycenter.py — Helpers for distributional priors in FedRL.

This module defines a per-round buffer that collects critic quantile outputs
and computes a CVaR-weighted barycentric prior for a single client.

Server-side aggregation across clients can be added here later.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class RoundDistBufferCfg:
    """
    Configuration for a single-client, single-round distributional buffer.

    alpha_tail: lower tail fraction used for per-state cvar.
    """
    alpha_tail: float = 0.10


class RoundDistBuffer:
    """
    Collects critic quantile outputs for one client in one communication round.

    Each call to add_batch adds a batch of quantile vectors q(x) in R^N for
    visited states. At the end of the round, build_prior computes a quantile-wise
    weighted median, where weights reflect both state frequency and per-state
    cvar magnitude.
    """

    def __init__(self, cfg: RoundDistBufferCfg):
        self.cfg = cfg
        self._z_list: List[torch.Tensor] = []
        self._cvar_list: List[torch.Tensor] = []
        self._num_samples: int = 0

    def reset(self) -> None:
        """
        Clear the buffer for a new communication round.
        """
        self._z_list.clear()
        self._cvar_list.clear()
        self._num_samples = 0

    def add_batch(self, z_batch: torch.Tensor, alpha_tail: Optional[float] = None) -> None:
        """
        Add a batch of critic quantile outputs for the current round.

        Args:
          z_batch: tensor of shape [B, Nq] with critic quantile values.
          alpha_tail: optional override for tail fraction used in cvar.
        """
        if z_batch is None or z_batch.numel() == 0:
            return

        if alpha_tail is None:
            alpha_tail = self.cfg.alpha_tail
        alpha_tail = float(alpha_tail)

        z_cpu = z_batch.detach().cpu()
        n_q = int(z_cpu.size(-1))
        k = max(1, int(alpha_tail * n_q))

        # per-state lower tail cvar (mean of the lowest k quantiles)
        cvar = z_cpu[:, :k].mean(dim=-1)  # [B]

        self._z_list.append(z_cpu)
        self._cvar_list.append(cvar)
        self._num_samples += int(z_cpu.size(0))

    def build_prior(
        self,
        use_cvar_weight: bool = True,
        cvar_scale: float = 1.0,
        eps: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        """
        Compute a per-round barycentric prior for this client.

        Returns:
          prior: tensor of shape [Nq] on cpu, or None if the buffer is empty.

        The prior is computed quantile-wise as a weighted median, with weights:
          w_i = 1 + cvar_scale * risk_i,   risk_i = max(-cvar_i, 0)
        where cvar_i is the lower tail cvar for state i. The constant "1"
        makes frequency explicit (each state contributes at least one unit).
        """
        if self._num_samples == 0:
            return None

        z_all = torch.cat(self._z_list, dim=0)     # [N, Nq]
        cvar_all = torch.cat(self._cvar_list, dim=0)  # [N]

        # base weight: one per state visit
        w = torch.ones_like(cvar_all)

        if use_cvar_weight:
            risk = (-cvar_all).clamp(min=0.0)  # larger for heavier lower tails
            w = w + float(cvar_scale) * risk

        w = (w + eps).clamp(min=eps)

        num_q = int(z_all.size(1))
        prior = torch.empty(num_q, dtype=z_all.dtype)

        for j in range(num_q):
            vals = z_all[:, j]
            vals_sorted, idx = torch.sort(vals)
            w_sorted = w[idx]
            cum_w = torch.cumsum(w_sorted, dim=0)
            cutoff = 0.5 * cum_w[-1]
            median_idx = torch.searchsorted(cum_w, cutoff)
            median_idx = int(torch.clamp(median_idx, 0, vals_sorted.numel() - 1))
            prior[j] = vals_sorted[median_idx]

        return prior
