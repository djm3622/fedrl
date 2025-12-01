"""
Barycenter utilities for per-round trust-region priors.

Unlike FedAvg over critic parameters, this module works directly with
critic output distributions. We use a simple weighted Euclidean barycenter,
which is sufficient because outputs are discrete quantiles / scalars.
"""

from __future__ import annotations

import torch


def weighted_barycenter(distributions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute a weighted barycenter over critic outputs.

    Args:
        distributions: [N, Q] or [N] tensor of critic outputs.
        weights: [N] non-negative weights.

    Returns:
        Tensor of shape [Q] (or scalar) representing the weighted barycenter.
    """
    if distributions.ndim == 1:
        distributions = distributions.unsqueeze(0)

    w = torch.as_tensor(weights, dtype=distributions.dtype, device=distributions.device)
    if w.numel() != distributions.size(0):
        raise ValueError("weights and distributions batch must align")

    w_clamped = torch.clamp(w, min=0.0)
    if float(w_clamped.sum().item()) <= 0.0:
        w_clamped = torch.ones_like(w_clamped)

    w_norm = w_clamped / w_clamped.sum()
    while w_norm.ndim < distributions.ndim:
        w_norm = w_norm.unsqueeze(-1)

    return (w_norm * distributions).sum(dim=0)

