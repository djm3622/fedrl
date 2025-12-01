#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl.py — Minimal FedAvg server for CRITIC only (with optional hazard-weighted aggregation).
- Aggregates critic parameters with weighted averaging and re-broadcasts.
- If hazard_metrics is provided to aggregate_and_refit, weights are computed as 1/(eps + hazard).
- Ignores any autoencoder flags and states.
- Comments are ascii-only.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional
import os
import torch
import wandb


def _weighted_avg_state_dict(
    state_dicts: List[Tuple[Dict[str, torch.Tensor], float]]
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Weighted average of state dicts. Returns None if nothing valid to average.
    - Expects a list of (state_dict, weight).
    - Ignores None or empty dicts and non-positive weights.
    - Validates that keys match across clients.
    """
    if not state_dicts:
        return None

    filtered = [
        (sd, float(max(0.0, w)))
        for sd, w in state_dicts
        if sd is not None and isinstance(sd, dict) and len(sd) > 0
    ]
    filtered = [(sd, w) for sd, w in filtered if w > 0.0]
    if not filtered:
        return None

    keys = list(filtered[0][0].keys())
    for sd, _ in filtered:
        if list(sd.keys()) != keys:
            raise RuntimeError("Mismatched critic state_dict keys across clients; cannot average.")

    acc = {k: torch.zeros_like(filtered[0][0][k], device="cpu") for k in keys}
    total_w = 0.0
    for sd, w in filtered:
        total_w += w
        for k in keys:
            acc[k] += sd[k].detach().cpu() * w

    if total_w <= 0.0:
        total_w = float(len(filtered))

    for k in keys:
        acc[k] /= total_w

    return acc


class FedRLServer:
    """
    Minimal server that aggregates CRITIC parameters via FedAvg and re-broadcasts them.

    New: If hazard_metrics is provided to aggregate_and_refit, the server computes
         hazard-weighted aggregation with weights w_i ∝ 1 / (eps + hazard_i).
    """
    def __init__(self, cfg: Any, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.round_idx = 0
        self.critic_state: Optional[Dict[str, torch.Tensor]] = None  # latest averaged critic state

        # hazard-weighting knobs (safe defaults if absent)
        self.agg_hazard_eps = float(getattr(cfg, "agg_hazard_eps", 1e-3))
        self.agg_w_max = float(getattr(cfg, "agg_w_max", 10.0))  # set <=0 to disable capping

        self.critic_state: Optional[Dict[str, torch.Tensor]] = None  # latest averaged critic state
        self.prior_quantiles: Optional[torch.Tensor] = None          # global distributional prior vector

        # Optional W&B run (server)
        self.run = None
        try:
            if getattr(cfg, "wandb_project", None):
                self.run = wandb.init(
                    project=cfg.wandb_project,
                    name=f"{getattr(cfg, 'wandb_run_name', 'run')}_fedrl_server",
                    id=f"{getattr(cfg, 'wandb_run_name', 'run')}_fedrl_server",
                    resume="allow",
                    config=asdict(cfg),
                    group=getattr(cfg, "wandb_group", None),
                    settings=wandb.Settings(start_method="thread"),
                )
        except Exception:
            self.run = None

    @torch.no_grad()
    def set_global_prior_quantiles(self, q_prior: Optional[torch.Tensor]) -> None:
        """
        Set the global distributional prior vector that will be broadcast to all clients.

        Args:
          q_prior: tensor [Nq] on any device, or None to clear.
        """
        if q_prior is None:
            self.prior_quantiles = None
            return
        if q_prior.dim() != 1:
            raise ValueError(f"set_global_prior_quantiles expected 1D tensor, got shape {tuple(q_prior.shape)}")
        self.prior_quantiles = q_prior.detach().cpu()


    def _wb_step(self) -> int:
        return int(self.round_idx)

    def package_broadcast(self) -> Dict[str, Any]:
        """
        What clients receive.

        For the new method, the important field is 'prior_quantiles', which is
        a global distributional prior vector. 'critic' is kept for backward
        compatibility and can be ignored by clients if they use the vector prior.
        """
        crit_sd = None if self.critic_state is None else {
            k: v.detach().cpu() for k, v in self.critic_state.items()
        }
        prior_q = None if self.prior_quantiles is None else self.prior_quantiles.detach().cpu()
        return {"critic": crit_sd, "prior_quantiles": prior_q}


    def _compute_hazard_weights(self, hazards: List[Optional[float]]) -> List[float]:
        """
        Convert per-client hazards into normalized weights via 1 / (eps + hazard),
        optionally capped by agg_w_max.
        """
        eps = float(self.agg_hazard_eps)
        w_cap = float(self.agg_w_max)
        ws: List[float] = []
        for h in hazards:
            h_val = 0.0 if (h is None) else float(h)
            w = 1.0 / (eps + max(0.0, h_val))
            if w_cap > 0.0:
                w = min(w, w_cap)
            ws.append(w)
        s = sum(ws)
        if s <= 0.0:
            n = max(1, len(ws))
            return [1.0 / n] * n
        return [w / s for w in ws]

    def aggregate_and_refit(
        self,
        critic_states: List[Tuple[Optional[Dict[str, torch.Tensor]], float]],
        *,
        hazard_metrics: Optional[List[Optional[float]]] = None,
        **_kwargs
    ) -> None:
        """
        Aggregate over the provided critic states.

        Args:
          critic_states: list of (state_dict or None, weight). If hazard_metrics is None,
                         these weights are used (FedAvg-compatible).
          hazard_metrics: optional list of per-client hazard rates aligned with critic_states.
                          If provided, server ignores the given weights and uses
                          hazard-weighted aggregation instead: w_i ∝ 1/(eps + hazard_i).

        Behavior:
          - If hazard_metrics is provided: use hazard-based weights.
          - Else: fallback to provided weights (original behavior).
        """
        # choose weights
        if hazard_metrics is not None:
            # compute hazard-based weights aligned with critic_states
            ws = self._compute_hazard_weights(hazard_metrics)
            typed: List[Tuple[Dict[str, torch.Tensor], float]] = []
            log_payload: Dict[str, float] = {}
            for idx, (sd, _) in enumerate(critic_states):
                if sd is None:
                    continue
                typed.append((sd, float(ws[idx])))
                # prepare logging
                log_payload[f"server/w_client_{idx}"] = float(ws[idx])
                h = hazard_metrics[idx] if hazard_metrics[idx] is not None else 0.0
                log_payload[f"server/hazard_client_{idx}"] = float(h)
        else:
            # use provided weights as-is (backward compatible)
            typed = [(sd, float(w)) for (sd, w) in critic_states if sd is not None]
            log_payload = {f"server/w_client_{i}": float(w) for i, (_, w) in enumerate(critic_states)}

        avg = _weighted_avg_state_dict(typed)
        if avg is not None:
            self.critic_state = avg

        if self.run is not None:
            log_data = {"server/round": int(self.round_idx)}
            log_data.update(log_payload)
            wandb.log(log_data, step=self._wb_step())

    def save_checkpoint(self, save_dir: str, tag: str):
        os.makedirs(save_dir, exist_ok=True)
        crit_path = None
        if self.critic_state is not None:
            crit_path = os.path.join(save_dir, f"critic_{tag}.pth")
            torch.save(self.critic_state, crit_path)
            if self.run is not None:
                art = wandb.Artifact(name=f"critic_{tag}", type="weights")
                art.add_file(crit_path)
                self.run.log_artifact(art)
        return crit_path

    def finish(self):
        if self.run is not None:
            self.run.finish()
