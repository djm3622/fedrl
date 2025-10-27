#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl.py — FedAvg server for CRITIC only with relative-hazard-weighted aggregation.

Goal: penalize unsafe LEARNING rather than unsafe DYNAMICS.
We weight client critics inversely to their hazard *excess* over a deterministic
baseline hazard assigned by the environment heterogeneity schedule.

Definitions:
  - h_i:    empirical hazard rate reported by client i for the round
  - h_bi:   baseline hazard for client i computed on server via _spread_value
  - h_ti:   normalized hazard = h_i / (h_bi + eps_baseline)
  - weight: w_i ∝ 1 / (h_ti + eps_weight)^p

Config knobs (with safe defaults if missing in cfg):
  - hazard_prob:                  float in [0,1], center of baseline hazard schedule (required by your env)
  - hazard_delta (or env HAZARD_DELTA): float >= 0, spread for baseline schedule (default 0.05)
  - agg_hazard_eps_weight:        epsilon added in weight denominator (default 1e-3)
  - agg_hazard_eps_baseline:      epsilon added in baseline denominator (default 1e-6)
  - agg_weight_power:             exponent p for 1/(...)^p (default 1.0)
  - agg_weight_cap_ratio:         cap on max/min weight ratio (>0 enables; default 30.0)
  - wandb_project, wandb_run_name, wandb_group: optional W&B settings
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional
import os
import torch
import wandb


# ---------- core averaging helper (unchanged) ----------

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


# ---------- small helper to mirror client-side heterogeneity schedule ----------

def _spread_value(center: float, delta: float, rank: int, n_clients: int) -> float:
    """
    Deterministic linear spread in [center - delta, center + delta] across client ranks.
    rank in [0, n_clients-1].
    """
    if n_clients <= 1 or delta <= 0.0:
        return float(center)
    t = rank / float(max(1, n_clients - 1))
    return float(max(0.0, min(1.0, (center - delta) + (2.0 * delta) * t)))


class FedRLServer:
    """
    Server that aggregates CRITIC parameters.
    If hazard_metrics is provided to aggregate_and_refit, the server performs
    RELATIVE hazard-weighted averaging as documented above.
    """
    def __init__(self, cfg: Any, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.round_idx = 0
        self.critic_state: Optional[Dict[str, torch.Tensor]] = None  # latest averaged critic state

        # --- weighting knobs (with defaults) ---
        # epsilons
        self.eps_weight   = float(getattr(cfg, "agg_hazard_eps_weight", 1e-3))
        self.eps_baseline = float(getattr(cfg, "agg_hazard_eps_baseline", 1e-6))
        # power
        self.weight_power = float(getattr(cfg, "agg_weight_power", 1.0))
        # cap on ratio max/min to avoid dominance (<=0 disables)
        self.cap_ratio    = float(getattr(cfg, "agg_weight_cap_ratio", 30.0))
        # baseline schedule center and spread
        self.baseline_center = float(getattr(cfg, "hazard_prob", 0.05))
        # prefer cfg.hazard_delta; else fall back to env var HAZARD_DELTA; else default 0.05
        self.baseline_delta  = float(getattr(cfg, "hazard_delta", float(os.getenv("HAZARD_DELTA", "0.05"))))

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

    # ---------- relative-hazard weighting ----------

    def _compute_relative_hazard_weights(
        self,
        hazards_by_slot: List[Optional[float]],
        valid_slots: List[int],
        n_clients_total: int
    ) -> List[float]:
        """
        hazards_by_slot: raw list aligned with original client slots (len == n_clients_total)
        valid_slots: indices of slots that actually provided a state dict this round
        returns: normalized weights aligned with valid_slots (same length)
        """
        eps_w  = self.eps_weight
        eps_b  = self.eps_baseline
        p      = self.weight_power
        cap_r  = self.cap_ratio

        # build per-valid-slot baseline and normalized hazards
        raw_weights: List[float] = []
        self._last_weight_debug: List[Tuple[int, float, float, float, float]] = []  # (slot, h, h_b, h_t, w_raw)

        for slot in valid_slots:
            # empirical hazard (clip to [0,1])
            h = hazards_by_slot[slot]
            h = 0.0 if h is None else float(min(max(h, 0.0), 1.0))
            # deterministic baseline via the same spread schedule
            h_b = _spread_value(self.baseline_center, self.baseline_delta, slot, n_clients_total)
            # normalized hazard (penalize excess over baseline)
            h_t = h / (h_b + eps_b)
            # raw inverse weight
            w_raw = (1.0 / (h_t + eps_w)) ** p
            raw_weights.append(w_raw)
            self._last_weight_debug.append((slot, h, h_b, h_t, w_raw))

        # optional cap on weight ratio to avoid dominance
        if cap_r and cap_r > 0.0 and len(raw_weights) > 1:
            w_min = max(min(raw_weights), 1e-12)
            w_max_allowed = w_min * cap_r
            raw_weights = [min(w, w_max_allowed) for w in raw_weights]

        s = sum(raw_weights) if raw_weights else 1.0
        return [w / s for w in raw_weights] if s > 0.0 else [1.0 / max(1, len(raw_weights))] * len(raw_weights)

    # ---------- aggregation entrypoint ----------

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
          hazard_metrics: optional list of per-client empirical hazards aligned with client slots.

        Behavior:
          - If hazard_metrics is provided: use RELATIVE hazard-based weights.
            The server computes per-slot baseline hazards and penalizes excess hazard.
          - Else: fallback to provided weights (original behavior).
        """
        # collect valid entries and their original slot indices
        valid: List[Tuple[int, Dict[str, torch.Tensor], float]] = []
        for slot, (sd, w) in enumerate(critic_states):
            if sd is not None:
                valid.append((slot, sd, float(w)))

        if not valid:
            if self.run is not None:
                wandb.log({"server/round": int(self.round_idx), "server/no_payloads": 1}, step=self._wb_step())
            return

        # choose weights
        to_avg: List[Tuple[Dict[str, torch.Tensor], float]] = []
        log_payload: Dict[str, float] = {}

        if hazard_metrics is not None:
            n_total = len(critic_states)
            valid_slots = [slot for (slot, _, _) in valid]
            ws = self._compute_relative_hazard_weights(hazard_metrics, valid_slots, n_total)

            # pair back aligned by valid_slots order
            for (slot, sd, _), w in zip(valid, ws):
                to_avg.append((sd, float(w)))
                # log diagnostics
                log_payload[f"server/w_client_{slot}"] = float(w)

            # optional detailed logging of hazards and baselines
            if self.run is not None:
                for slot, h, h_b, h_t, w_raw in getattr(self, "_last_weight_debug", []):
                    log_payload[f"server/hazard_emp_client_{slot}"] = float(h)
                    log_payload[f"server/hazard_base_client_{slot}"] = float(h_b)
                    log_payload[f"server/hazard_norm_client_{slot}"] = float(h_t)
                    log_payload[f"server/w_raw_client_{slot}"] = float(w_raw)

                # also log the main knob values to make sweeps auditable
                log_payload["server/eps_weight"] = float(self.eps_weight)
                log_payload["server/eps_baseline"] = float(self.eps_baseline)
                log_payload["server/weight_power"] = float(self.weight_power)
                log_payload["server/cap_ratio"] = float(self.cap_ratio)

        else:
            # fallback: use provided weights
            for slot, sd, w in valid:
                to_avg.append((sd, float(w)))
                log_payload[f"server/w_client_{slot}"] = float(w)

        avg = _weighted_avg_state_dict(to_avg)
        if avg is not None:
            self.critic_state = avg

        if self.run is not None:
            log_data = {"server/round": int(self.round_idx)}
            log_data.update(log_payload)
            wandb.log(log_data, step=self._wb_step())

    # ---------- checkpointing ----------

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
