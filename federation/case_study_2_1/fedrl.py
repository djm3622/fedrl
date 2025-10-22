#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl.py — Minimal FedAvg server for CRITIC only.
- Aggregates critic parameters with weighted averaging and re-broadcasts.
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
    """
    def __init__(self, cfg: Any, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.round_idx = 0
        self.critic_state: Optional[Dict[str, torch.Tensor]] = None  # latest averaged critic state

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

    def _wb_step(self) -> int:
        return int(self.round_idx)

    def package_broadcast(self) -> Dict[str, Any]:
        """
        What clients receive. Only the critic state is sent (or None).
        """
        crit_sd = None if self.critic_state is None else {k: v.detach().cpu() for k, v in self.critic_state.items()}
        return {"critic": crit_sd}

    def aggregate_and_refit(
        self,
        critic_states: List[Tuple[Optional[Dict[str, torch.Tensor]], float]],
        **_kwargs
    ) -> None:
        """
        FedAvg over the provided critic states.
        critic_states: list of (state_dict or None, weight)
        """
        typed: List[Tuple[Dict[str, torch.Tensor], float]] = [
            (sd, w) for (sd, w) in critic_states if sd is not None
        ]
        avg = _weighted_avg_state_dict(typed)
        if avg is not None:
            self.critic_state = avg

        if self.run is not None:
            wandb.log({"server/round": int(self.round_idx)}, step=self._wb_step())

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

