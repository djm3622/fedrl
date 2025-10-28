#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl.py — FedAvg server for CRITIC only with relative-hazard-weighted aggregation
and trust-region blending.

Goal: penalize unsafe LEARNING rather than unsafe DYNAMICS, while preserving
round-to-round stability of the server prior. We weight client critics inversely
to their hazard excess over a deterministic baseline, then apply a trust-region
blend between the previous server critic and the new weighted average.

Definitions:
  - h_i:    empirical hazard rate reported by client i for the round
  - h_bi:   baseline hazard for client i computed on server via _spread_value
  - h_ti:   normalized hazard = h_i / (h_bi + eps_baseline)
  - weight: w_i ∝ 1 / (h_ti + eps_weight)^p

Server trust region:
  - theta_next = (1 - beta_tr) * theta_prev + beta_tr * theta_avg
  - beta_tr = 1 / (1 + agg_trust_xi), agg_trust_xi >= 0
  - Setting agg_trust_xi = 0 disables blending (theta_next = theta_avg).

Config knobs (with safe defaults if missing in cfg):
  - hazard_prob:                  float in [0,1], center of baseline hazard schedule (required by env)
  - hazard_delta (or env HAZARD_DELTA): float >= 0, spread for baseline schedule (default 0.05)
  - agg_hazard_eps_weight:        epsilon added in weight denominator (default 1e-3)
  - agg_hazard_eps_baseline:      epsilon added in baseline denominator (default 1e-6)
  - agg_weight_power:             exponent p for 1/(...)^p (default 1.0)
  - agg_weight_cap_ratio:         cap on max/min weight ratio (>0 enables; default 30.0)
  - agg_trust_xi:                 server trust-region xi; larger -> more conservative (default 0.1)
  - agg_hazard_ema_rho:           EMA factor on hazards in [0,1); 0 disables EMA (default 0.9)
  - wandb_project, wandb_run_name, wandb_group: optional W&B settings
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional
import os
import torch
import wandb


# ---------- key helpers (subset detection / split) ----------

def _is_encoder_key(k: str) -> bool:
    # critic state_dict keys do NOT include "critic." prefix; examples:
    # "encoder.conv.0.weight", "encoder.head.0.weight", "_proj.0.weight"
    return k.startswith("encoder.") or k.startswith("_proj.")

def _is_head_key(k: str) -> bool:
    # expected critic: "v.weight", "v.bias"
    # distributional:  "head.weight", "head.bias"
    return (k == "head.weight") or (k == "head.bias") or (k == "v.weight") or (k == "v.bias")

def _split_critic(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
                                                        Dict[str, torch.Tensor],
                                                        Dict[str, torch.Tensor]]:
    enc, head, other = {}, {}, {}
    for k, v in sd.items():
        if _is_encoder_key(k):
            enc[k] = v
        elif _is_head_key(k):
            head[k] = v
        else:
            other[k] = v  # buffers: taus, etc.
    return enc, head, other


# ---------- core averaging helper ----------

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


def _blend_trust_region(
    prev: Optional[Dict[str, torch.Tensor]],
    avg: Dict[str, torch.Tensor],
    xi: float
) -> Dict[str, torch.Tensor]:
    """
    Trust-region blend:
      next = (1 - beta_tr) * prev + beta_tr * avg,  with beta_tr = 1 / (1 + xi).
    If prev is None or xi <= 0, returns avg.
    Operates on CPU tensors; caller should move to device as needed.
    """
    if prev is None or xi <= 0.0:
        return {k: v.detach().cpu() for k, v in avg.items()}
    beta_tr = 1.0 / (1.0 + float(xi))
    one_minus = 1.0 - beta_tr
    out: Dict[str, torch.Tensor] = {}
    for k in avg.keys():
        a = avg[k].detach().cpu()
        p = prev.get(k, None)
        if p is None:
            out[k] = a
        else:
            out[k] = one_minus * p.detach().cpu() + beta_tr * a
    return out


# ---------- head alignment helpers (row-wise sign + capped norm) ----------

def _find_head_keys(sd: Dict[str, torch.Tensor]) -> Optional[Tuple[str, str]]:
    if "head.weight" in sd and "head.bias" in sd:
        return ("head.weight", "head.bias")
    if "v.weight" in sd and "v.bias" in sd:
        return ("v.weight", "v.bias")
    return None

def _align_head_to_ref(
    sd: Dict[str, torch.Tensor],
    ref_sd: Optional[Dict[str, torch.Tensor]],
    *,
    do_scale: bool = True,
    sclip: Tuple[float, float] = (0.25, 4.0),
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Row-wise align a client's head to a reference head by sign (dot with ref row)
    and optional norm scaling. Returns a shallow-copied state dict with aligned head.
    - Works for DistValueCritic (head.*) and CentralCritic (v.*).
    - If ref_sd is None or keys missing, returns sd unchanged.
    """
    hk = _find_head_keys(sd)
    if hk is None or ref_sd is None:
        return sd
    kW, kb = hk
    if (kW not in ref_sd) or (kb not in ref_sd):
        return sd

    W  = sd[kW].detach().cpu().clone()
    b  = sd[kb].detach().cpu().clone()
    Wref = ref_sd[kW].detach().cpu()
    bref = ref_sd[kb].detach().cpu()

    # shape guard
    if W.shape != Wref.shape or b.shape != bref.shape:
        return sd  # safest: skip alignment on mismatch

    # row-wise sign via dot with reference row
    dots = (W * Wref).sum(dim=1, keepdim=True)  # [R,1]
    sgn  = torch.where(dots >= 0, 1.0, -1.0)    # [R,1]
    W = W * sgn
    b = b * sgn.squeeze(1)

    # optional norm scaling toward reference row norms
    if do_scale:
        nr = torch.linalg.norm(Wref, dim=1).clamp_min(eps)   # [R]
        nc = torch.linalg.norm(W,    dim=1).clamp_min(eps)   # [R]
        scale = (nr / nc).clamp(min=sclip[0], max=sclip[1])  # [R]
        W = W * scale[:, None]
        b = b * scale

    out = sd.copy()
    out[kW] = W
    out[kb] = b
    return out


# ---------- baseline schedule helper ----------

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
    RELATIVE hazard-weighted averaging as documented above, then applies a
    trust-region blend with the previous server critic.
    """
    def __init__(self, cfg: Any, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.round_idx = 0
        self.critic_state: Optional[Dict[str, torch.Tensor]] = None  # latest blended critic state

        # --- weighting knobs (with defaults) ---
        self.eps_weight   = float(getattr(cfg, "agg_hazard_eps_weight", 1e-3))
        self.eps_baseline = float(getattr(cfg, "agg_hazard_eps_baseline", 1e-6))
        self.weight_power = float(getattr(cfg, "agg_weight_power", 1.0))
        self.cap_ratio    = float(getattr(cfg, "agg_weight_cap_ratio", 30.0))

        # baseline schedule center and spread
        self.baseline_center = float(getattr(cfg, "hazard_prob", 0.05))
        self.baseline_delta  = float(getattr(cfg, "hazard_delta", float(os.getenv("HAZARD_DELTA", "0.05"))))

        # --- trust region knobs ---
        self.trust_xi = float(getattr(cfg, "agg_trust_xi", 0.1))  # 0.0 disables trust-region blend
        self._tr_beta_last: Optional[float] = None

        # --- optional hazard EMA smoothing ---
        self.hazard_ema_rho = float(getattr(cfg, "agg_hazard_ema_rho", 0.9))  # 0.0 disables EMA
        self._hazard_ema: Dict[int, float] = {}  # slot -> smoothed hazard

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
        What clients receive.
          - 'critic': full prior (encoder+proj+head+buffers) for FROZEN prior path on clients
          - 'critic_trunc': encoder(+proj) only for TRAINABLE overwrite on clients
        """
        crit_sd = None if self.critic_state is None else {k: v.detach().cpu() for k, v in self.critic_state.items()}
        enc_sd = None
        if crit_sd is not None:
            enc_sd = {k: v for k, v in crit_sd.items() if _is_encoder_key(k)}
        return {"critic": crit_sd, "critic_trunc": enc_sd}

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
        rho    = self.hazard_ema_rho

        raw_weights: List[float] = []
        self._last_weight_debug: List[Tuple[int, float, float, float, float]] = []  # (slot, h, h_b, h_t, w_raw)

        for slot in valid_slots:
            # empirical hazard (clip to [0,1])
            h_raw = hazards_by_slot[slot]
            h_raw = 0.0 if h_raw is None else float(min(max(h_raw, 0.0), 1.0))

            # EMA smoothing (optional)
            if rho > 0.0:
                h_prev = self._hazard_ema.get(slot, h_raw)
                h = float(rho * h_prev + (1.0 - rho) * h_raw)
                self._hazard_ema[slot] = h
            else:
                h = h_raw

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
          - Else: fallback to provided weights (original behavior).
          - Encoder is averaged directly; head is aligned to previous prior head before averaging.
          - After averaging, apply trust-region blend per-part, then rebuild full prior.
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

            for (slot, sd, _), w in zip(valid, ws):
                to_avg.append((sd, float(w)))
                log_payload[f"server/w_client_{slot}"] = float(w)

            if self.run is not None:
                for slot, h, h_b, h_t, w_raw in getattr(self, "_last_weight_debug", []):
                    log_payload[f"server/hazard_emp_client_{slot}"] = float(h)
                    log_payload[f"server/hazard_base_client_{slot}"] = float(h_b)
                    log_payload[f"server/hazard_norm_client_{slot}"] = float(h_t)
                    log_payload[f"server/w_raw_client_{slot}"] = float(w_raw)

                log_payload["server/eps_weight"] = float(self.eps_weight)
                log_payload["server/eps_baseline"] = float(self.eps_baseline)
                log_payload["server/weight_power"] = float(self.weight_power)
                log_payload["server/cap_ratio"] = float(self.cap_ratio)
                log_payload["server/hazard_ema_rho"] = float(self.hazard_ema_rho)

        else:
            # fallback: use provided weights
            ws = [w for (_, _, w) in valid]
            for slot, sd, w in valid:
                to_avg.append((sd, float(w)))
                log_payload[f"server/w_client_{slot}"] = float(w)

        # -------- split encoder/head/other for each client --------
        enc_list: List[Tuple[Dict[str, torch.Tensor], float]] = []
        head_list: List[Tuple[Dict[str, torch.Tensor], float]] = []
        other_template: Optional[Dict[str, torch.Tensor]] = None

        for (sd, w) in [(sd, w) for sd, w in [(sd, w) for (_, sd, _), w in zip(valid, ws)]]:
            enc_i, head_i, other_i = _split_critic(sd)
            if enc_i:
                enc_list.append((enc_i, w))
            if head_i:
                head_list.append((head_i, w))
            if other_template is None:
                other_template = other_i  # buffers

        # -------- (A) encoder average + trust region --------
        enc_avg = _weighted_avg_state_dict(enc_list) if enc_list else {}
        prev_enc = None
        if self.critic_state is not None:
            prev_enc = {k: v for k, v in self.critic_state.items() if _is_encoder_key(k)}
        enc_blend = _blend_trust_region(prev_enc, enc_avg, xi=max(0.0, float(self.trust_xi))) if enc_avg else (prev_enc or {})

        # -------- (B) head alignment to previous prior then average + TR --------
        # build reference head from previous prior
        ref_head: Dict[str, torch.Tensor] = {}
        if self.critic_state is not None:
            _, ref_head, _ = _split_critic(self.critic_state)

        # weighted average of aligned heads (only weight/bias)
        head_acc: Dict[str, torch.Tensor] = {}
        wtot = 0.0
        if head_list:
            # infer key set from first available head
            hk = _find_head_keys(head_list[0][0])
            if hk is not None:
                kW, kb = hk
                for head_i, w in head_list:
                    try:
                        aligned = _align_head_to_ref(head_i, ref_head, do_scale=True, sclip=(0.25, 4.0))
                    except Exception:
                        aligned = head_i
                    Wi = aligned.get(kW, None)
                    bi = aligned.get(kb, None)
                    if Wi is None or bi is None:
                        continue
                    Wi = Wi.detach().cpu()
                    bi = bi.detach().cpu()
                    head_acc[kW] = (Wi * w) if kW not in head_acc else (head_acc[kW] + Wi * w)
                    head_acc[kb] = (bi * w) if kb not in head_acc else (head_acc[kb] + bi * w)
                    wtot += w
                if wtot > 0.0:
                    head_acc[kW] = head_acc[kW] / wtot
                    head_acc[kb] = head_acc[kb] / wtot

        # trust-region blend for head
        head_blend: Dict[str, torch.Tensor] = {}
        if head_acc:
            prev_head = {k: v for k, v in (self.critic_state or {}).items() if _is_head_key(k)}
            if prev_head:
                beta_tr = 1.0 / (1.0 + max(0.0, float(self.trust_xi)))
                for k, a in head_acc.items():
                    p = prev_head.get(k, None)
                    head_blend[k] = ((1.0 - beta_tr) * p.detach().cpu() + beta_tr * a) if p is not None else a
            else:
                head_blend = {k: v.detach().cpu() for k, v in head_acc.items()}
        else:
            # no head in payload; keep previous head as-is
            head_blend = {k: v for k, v in (self.critic_state or {}).items() if _is_head_key(k)}

        # -------- rebuild full prior state --------
        new_state: Dict[str, torch.Tensor] = {}
        new_state.update(enc_blend)                                   # encoder(+proj)
        if other_template:
            new_state.update({k: v.detach().cpu() for k, v in other_template.items()})  # buffers passthrough
        new_state.update(head_blend)                                   # head

        self.critic_state = {k: v.detach().cpu() for k, v in new_state.items()}

        # -------- diagnostics (unchanged logging) --------
        if self.run is not None:
            log_data = {"server/round": int(self.round_idx)}
            log_data.update(log_payload)
            log_data["server/agg_trust_xi"] = float(self.trust_xi)
            try:
                v_new = torch.cat([p.flatten() for p in self.critic_state.values()])
                v_avg_e = torch.cat([p.flatten() for p in enc_avg.values()]) if enc_avg else torch.zeros(1)
                # simple magnitude probe using encoder part (closest to past behavior)
                step_norm = torch.norm(v_new[: v_avg_e.numel()] - v_avg_e).item() if v_avg_e.numel() > 0 else 0.0
                log_data["server/trust_step_l2_vs_avg"] = float(step_norm)
            except Exception:
                pass
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
