#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedrl.py — Federated Distributional RL server (critic-only in latent space).

Protocol each round:
  1) Encoder parameter averaging: omega_G <- sum_i beta_i * omega_i.
  2) Temporary global head: parameter-average latent heads for an initial q prior.
  3) Pair similar latents across clients; for each action and tau grid, compute
     risk-weighted Wasserstein barycenters; build a pseudo dataset (z, a) -> q_bar.
  4) Fine-tune the temporary global head on the pseudo dataset to fit barycenters.
  5) Broadcast updated encoder and head.

Clients keep actors local and never share actor parameters.

Pairing details (this implementation):
  - same-action kNN seeds with optional radius
  - require at least two distinct clients per cluster
  - weights combine distance kernel and risk dampening
  - optional percentile trimming for outliers

All comments are ascii-only.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from models.fedrl_nets import (
    LatentDistHead, barycenter_1d_quantiles
)

# try to use quantile-huber if available; fallback to mse
try:
    from agents.case_study_2_1.ppo_utils.helpers import quantile_huber_loss as _qr_loss
    _HAS_QR = True
except Exception:
    _HAS_QR = False


# ----------------- minimal W&B init for server -----------------

def _server_wandb_init(project: str, name: str, cfg_dict: dict, group: str | None):
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(
        project=project,
        name=name + "_fedrl_server",
        id=name + "_fedrl_server",
        resume="allow",
        config=cfg_dict,
        group=group,
    )


# ----------------- helper: parameter averaging -----------------

def _weighted_avg_state_dict(state_dicts: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    assert len(state_dicts) > 0
    keys = list(state_dicts[0][0].keys())
    acc = {k: torch.zeros_like(state_dicts[0][0][k], device="cpu") for k in keys}
    total_w = 0.0
    for sd, w in state_dicts:
        w = float(max(0.0, w))
        if w == 0.0:
            continue
        for k in keys:
            acc[k] += sd[k].detach().cpu() * w
        total_w += w
    if total_w == 0.0:
        total_w = float(len(state_dicts))
    for k in keys:
        acc[k] /= total_w
    return acc


# ----------------- FedRL server -----------------

class FedRLServer:
    def __init__(self, cfg: Any, d_latent: int, n_actions: int, n_quantiles: int, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.d_latent = int(d_latent)

        # server models
        self.encoder = None  # set by first payload's shapes if needed
        self.head = LatentDistHead(d_latent=d_latent, n_actions=n_actions, n_quantiles=n_quantiles).to(self.device)

        self.round_idx = 0
        self.run = _server_wandb_init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            cfg_dict=asdict(cfg),
            group=getattr(cfg, "wandb_group", None),
        )
        self.server_step_base = 0
        if self.run is not None and getattr(self.run, "resumed", False):
            self.server_step_base = int(self.run.summary.get("server/last_step", 0))

    def _wb_step(self) -> int:
        return int(self.server_step_base + self.round_idx)

    # ---- broadcast package ----
    def package_broadcast(self) -> Dict[str, Any]:
        enc_sd = None
        if self.encoder is not None:
            enc_sd = {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
        head_sd = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        return {"encoder": enc_sd, "head": head_sd}

    # ---- load from averaged states ----
    def load_encoder_from_state(self, enc_state: Dict[str, torch.Tensor], encoder_builder):
        if self.encoder is None:
            self.encoder = encoder_builder().to(self.device)
        self.encoder.load_state_dict(enc_state, strict=True)
        self.encoder.to(self.device)

    def load_head_from_state(self, head_state: Dict[str, torch.Tensor]):
        self.head.load_state_dict(head_state, strict=True)
        self.head.to(self.device)

    # ---- one federated aggregation round ----
    def aggregate_and_refit(self,
                            encoder_states: List[Tuple[Dict[str, torch.Tensor], float]],
                            head_states: List[Tuple[Dict[str, torch.Tensor], float]],
                            latent_payloads: List[Dict[str, Any]],
                            encoder_builder) -> None:
        """
        encoder_states: list of (state_dict, weight)
        head_states   : list of (state_dict, weight) for latent head
        latent_payloads: list of dicts with:
           {
             "z": FloatTensor [K_i, d_latent],
             "a": LongTensor  [K_i],
             "q": FloatTensor [K_i, n_quantiles],
             "risk": float in [0,1]
           }
        encoder_builder: callable returning an encoder module with same arch as clients
        """

        # 1) encoder param average
        enc_avg = _weighted_avg_state_dict(encoder_states)
        self.load_encoder_from_state(enc_avg, encoder_builder)

        # 2) temporary global head by param average
        head_avg = _weighted_avg_state_dict(head_states)
        self.load_head_from_state(head_avg)

        # 3) risk-aware pairwise barycenters across clients
        # gather all (z, a, q, client_id, risk)
        zs, as_, qs, gids, risks = [], [], [], [], []
        for gid, pay in enumerate(latent_payloads):
            if pay is None:
                continue
            z_i: torch.Tensor = pay["z"].detach().cpu()
            a_i: torch.Tensor = pay["a"].detach().cpu().long()
            q_i: torch.Tensor = pay["q"].detach().cpu()
            r_i: float = float(max(0.0, min(1.0, pay.get("risk", 0.0))))
            if z_i.numel() == 0:
                continue
            zs.append(z_i); as_.append(a_i); qs.append(q_i)
            gids.append(torch.full((z_i.size(0),), gid, dtype=torch.long))
            risks.append(torch.full((z_i.size(0),), r_i, dtype=torch.float32))

        if len(zs) == 0:
            return

        Z = torch.cat(zs, dim=0)                  # [N_all, d_latent]
        A = torch.cat(as_, dim=0)                 # [N_all]
        Q = torch.cat(qs, dim=0)                  # [N_all, n_quantiles]
        G = torch.cat(gids, dim=0)                # [N_all]
        R = torch.cat(risks, dim=0)               # [N_all]

        # config
        k_neighbors = int(getattr(self.cfg, "fedrl_knn_k", 4))
        radius = float(getattr(self.cfg, "fedrl_pair_radius", 0.75))     # used for cosine in [0,2], l2 else
        metric = str(getattr(self.cfg, "fedrl_pair_metric", "cosine"))   # "cosine" or "l2"
        lam_dist = float(getattr(self.cfg, "fedrl_pair_lambda", 2.0))    # distance kernel rate
        kappa = float(getattr(self.cfg, "fedrl_risk_kappa", 0.0))
        trim_pct = float(getattr(self.cfg, "fedrl_pair_trim_pct", 0.0))  # percentile trimming in [0,100)

        Zn = F.normalize(Z, dim=-1) if metric == "cosine" else Z
        N = Z.size(0)

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        used = torch.zeros(N, dtype=torch.bool)
        for i in range(N):
            if used[i]:
                continue
            ai = A[i].item()
            zi = Zn[i:i+1]
            # same action and not used
            mask = (A == ai) & (~used)
            if mask.sum().item() <= 1:
                continue
            Zcand = Zn[mask]
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            # distances
            if metric == "cosine":
                sim = torch.matmul(Zcand, zi.t()).squeeze(-1)  # [M]
                dist = 1.0 - sim
            else:
                dist = torch.norm(Zcand - Z[i:i+1], dim=-1)

            # take k nearest
            order = torch.argsort(dist)
            order = order[:k_neighbors]
            sel = idxs[order]
            sel_dist = dist[order]

            # optional radius cutoff
            if radius > 0:
                keep = sel_dist <= radius
                sel = sel[keep]
                sel_dist = sel_dist[keep]

            if sel.numel() <= 1:
                continue

            # optional trimming by percentile to drop outliers
            if trim_pct > 0.0 and sel_dist.numel() > 3:
                thr = torch.quantile(sel_dist, q=min(0.999, max(0.0, trim_pct / 100.0)))
                keep = sel_dist <= thr
                sel = sel[keep]
                sel_dist = sel_dist[keep]
                if sel.numel() <= 1:
                    continue

            # require at least two distinct clients in the cluster
            if torch.unique(G[sel]).numel() < 2:
                continue

            # distance weights (exponential kernel) and risk dampening
            # scale distance by its mean for numerical stability
            dscale = sel_dist.mean().item() if sel_dist.numel() > 0 else 1.0
            dscale = max(dscale, 1e-8)
            w_dist = torch.exp(-lam_dist * (sel_dist / dscale))
            w_dist = w_dist / (w_dist.sum() + 1e-12)

            cl_risk = R[sel]  # [m]
            w = w_dist / (1.0 + kappa * cl_risk)
            w = w / (w.sum() + 1e-12)

            pairs.append((sel, w))
            used[sel] = True

        # compute barycenters and build pseudo dataset
        zb_list, ab_list, qb_list = [], [], []
        for sel, w in pairs:
            q_list = [Q[j] for j in sel]                 # each [n_quantiles]
            q_bar = barycenter_1d_quantiles(q_list, w.tolist())  # [n_quantiles]
            z_bar = Z[sel].mean(dim=0)
            a_bar = A[sel][0]
            zb_list.append(z_bar.unsqueeze(0))
            ab_list.append(a_bar.view(1))
            qb_list.append(q_bar.unsqueeze(0))

        if len(zb_list) == 0:
            return

        Z_bar = torch.cat(zb_list, dim=0).to(self.device)        # [B, d_latent]
        A_bar = torch.cat(ab_list, dim=0).to(self.device).long() # [B]
        Q_bar = torch.cat(qb_list, dim=0).to(self.device)        # [B, n_quantiles]

        # 4) fine-tune the head on (Z_bar, A_bar) -> Q_bar
        self.head.train()
        opt = torch.optim.Adam(self.head.parameters(), lr=float(getattr(self.cfg, "fedrl_head_ft_lr", 1e-3)))
        ft_steps = int(getattr(self.cfg, "fedrl_head_ft_steps", 200))
        bs = int(getattr(self.cfg, "fedrl_head_ft_bs", 256))

        if _HAS_QR:
            taus = torch.linspace(0.0, 1.0, steps=self.n_quantiles, device=self.device)
        for t in range(ft_steps):
            idx = torch.randint(0, Z_bar.size(0), (min(bs, Z_bar.size(0)),), device=Z_bar.device)
            z_mb = Z_bar[idx]
            a_mb = A_bar[idx]
            q_tgt = Q_bar[idx]
            q_hat = self.head(z_mb, a_mb)
            if _HAS_QR:
                loss = _qr_loss(q_hat, q_tgt, taus=taus, kappa=float(getattr(self.cfg, "qr_kappa", 1.0)))
            else:
                loss = F.mse_loss(q_hat, q_tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (t + 1) % 50 == 0 and self.run is not None:
                wandb.log({ "server/fedrl/ft_loss": float(loss.item()) }, step=self._wb_step())

        if self.run is not None:
            wandb.log({
                "server/fedrl/n_pairs": len(pairs),
                "server/round": int(self.round_idx),
            }, step=self._wb_step())
            self.run.summary["server/last_step"] = self._wb_step()

    # ---- checkpointing ----
    def save_checkpoint(self, save_dir: str, tag: str):
        os.makedirs(save_dir, exist_ok=True)
        enc_path = None
        if self.encoder is not None:
            enc_path = os.path.join(save_dir, f"encoder_{tag}.pth")
            torch.save(self.encoder.state_dict(), enc_path)
        head_path = os.path.join(save_dir, f"latent_head_{tag}.pth")
        torch.save(self.head.state_dict(), head_path)
        if self.run is not None:
            if enc_path is not None:
                art_e = wandb.Artifact(name=f"encoder_{tag}", type="weights")
                art_e.add_file(enc_path)
                self.run.log_artifact(art_e)
            art_h = wandb.Artifact(name=f"latent_head_{tag}", type="weights")
            art_h.add_file(head_path)
            self.run.log_artifact(art_h)
        return enc_path, head_path

    def finish(self):
        if self.run is not None:
            self.run.finish()
