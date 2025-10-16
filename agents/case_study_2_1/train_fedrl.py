#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fedrl.py — Multi-process single-GPU Federated Distributional RL with latent aggregation.

High-level:
  - Long-lived clients. Each has: env, MAPPO actor, external AE, latent dist head.
  - Local training uses either "lipschitz" operators (soft feasibility + shrinkage) or "map_w1" trust-region.
  - After local epochs, each client returns:
      * encoder state_dict (for parameter averaging)
      * head state_dict      (for temp prior)
      * latent triples (z, a, q) subsampled from recent steps (per-action balanced)
      * risk summary (cvar of per-batch loss, normalized)
  - Server averages encoder, builds temp global head, computes barycenters on paired latents,
    fine-tunes the global head on barycenters (quantile-huber if available), and broadcasts new encoder+head.
"""

from __future__ import annotations
import os
import time
import math
import copy
from dataclasses import asdict
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb

from utils.general import normalize_weights
from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel
from models.fedrl_nets import build_autoencoder, LatentDistHead
from federation.case_study_2_1.fedrl import FedRLServer

# helpers from your PPO utils (quantile huber and optim)
from agents.case_study_2_1.ppo_utils.helpers import (
    quantile_huber_loss, setup_optimizers
)

# ---------------- CPU threading ----------------
def _set_threads_per_proc(intra: int, inter: int = 1, blas_threads: int = 1):
    os.environ["OMP_NUM_THREADS"] = str(intra)
    os.environ["MKL_NUM_THREADS"] = str(intra)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    torch.set_num_threads(intra)
    torch.set_num_interop_threads(inter)


def _client_wandb_safe_init(project: str, name: str, cfg_dict: dict, group: str | None, rank: int):
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")
    base_name = f"{name}_fedrl_client{rank}"
    run_id = base_name
    run_dir = os.path.join(
        os.getenv("SLURM_TMPDIR", os.path.join(os.getcwd(), "wandb_cache")),
        f"client_{rank}"
    )
    os.makedirs(run_dir, exist_ok=True)
    settings = wandb.Settings(start_method="thread")
    try:
        return wandb.init(
            project=project,
            name=base_name,
            id=run_id,
            resume="allow",
            config=cfg_dict,
            group=(group if group is not None else f"{name}_fedrl"),
            dir=run_dir,
            settings=settings,
        )
    except Exception as e:
        print(f"[wandb] client {rank} init failed: {e}  (continuing without W&B)")
        return None


# ---------------- Local trainer (minimal) ----------------

class FedRLClientTrainer:
    """
    Minimal on-policy training loop using MAPPO actor and a latent-space dist critic:
      - AE: encoder+decoder
      - Head: maps (z, a) to quantiles
      - Value estimate uses policy-weighted expected quantile mean across actions

    Two regularization modes:
      1) lipschitz: feasibility clamp via tanh and affine shrink toward server prior
      2) map_w1   : quantile-level W1 trust-region penalty to prior
    """
    def __init__(self, cfg: Any, env: Any, actor_model: Any, device: str, client_id: int):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        self.client_id = int(client_id)

        # models
        self.actor = actor_model.actor.to(self.device)
        # AE
        self.ae = build_autoencoder(in_ch=6, d_latent=cfg.fedrl_d_latent)
        self.encoder = self.ae.encoder.to(self.device)
        self.decoder = self.ae.decoder.to(self.device)
        # latent dist head
        self.head = LatentDistHead(d_latent=cfg.fedrl_d_latent, n_actions=5, n_quantiles=cfg.n_quantiles).to(self.device)

        # optim
        self.opt_actor, _ = setup_optimizers(actor_model, cfg)  # keep actor opt from your helper
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=float(getattr(cfg, "fedrl_enc_lr", 1e-3)))
        self.opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=float(getattr(cfg, "fedrl_dec_lr", 5e-4)))
        self.opt_head = torch.optim.Adam(self.head.parameters(), lr=float(getattr(cfg, "fedrl_head_lr", 1e-3)))

        # state
        (actor_obs, critic_obs), _ = self.env.reset(seed=cfg.seed)
        self.actor_obs = actor_obs
        self.critic_obs = critic_obs
        self.n_agents = self.env.cfg.n_agents
        self.h_actor = self.actor.init_hidden(self.n_agents, self.device)
        self.total_env_steps = 0

        self.prior_head_state = None  # will be set on each broadcast

        # buffer of latent triples for server payload
        self.z_buf: List[torch.Tensor] = []
        self.a_buf: List[torch.Tensor] = []
        self.q_buf: List[torch.Tensor] = []
        self.loss_recent: List[float] = []

    # ----- broadcast loaders -----
    def load_broadcast(self, pkg: Dict[str, Any]):
        enc_sd = pkg.get("encoder", None)
        if enc_sd is not None:
            self.encoder.load_state_dict(enc_sd, strict=True)
            self.encoder.to(self.device)
        head_sd = pkg.get("head", None)
        if head_sd is not None:
            self.prior_head_state = {k: v.clone() for k, v in head_sd.items()}
            self.head.load_state_dict(head_sd, strict=True)

    # ----- policy forward -----
    @torch.no_grad()
    def _forward_policy(self, ego_bna: torch.Tensor, agent_ids: torch.Tensor, h_in: torch.Tensor):
        dists, h_out = self.actor(ego_bna, agent_ids, h_in=h_in)
        actions = dists.sample()
        logprobs = dists.log_prob(actions)
        return actions, logprobs, dists, h_out

    def _encode_global(self, glob_b: torch.Tensor) -> torch.Tensor:
        return self.encoder(glob_b)

    def _critic_quantiles(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.head(z, a)

    @torch.no_grad()
    def _value_from_quantiles(self, z: torch.Tensor, actor_logits_dist) -> torch.Tensor:
        A = 5
        B = z.size(0)
        qs = []
        for a in range(A):
            a_t = torch.full((B,), a, dtype=torch.long, device=z.device)
            q = self.head(z, a_t)            # [B, M]
            qs.append(q.mean(dim=-1, keepdim=True))
        q_mean_per_a = torch.cat(qs, dim=-1)  # [B, A]
        probs = actor_logits_dist.probs if hasattr(actor_logits_dist, "probs") else torch.softmax(actor_logits_dist.logits, dim=-1)
        v = (q_mean_per_a * probs).sum(dim=-1)
        return v

    # ----- local regularization operators -----

    def _apply_lipschitz_ops(self, q_pred: torch.Tensor, q_prior: torch.Tensor) -> torch.Tensor:
        """
        Soft feasibility clamp and affine shrink toward prior.
        """
        delta = float(getattr(self.cfg, "fedrl_lip_delta", 10.0))
        beta = float(getattr(self.cfg, "fedrl_lip_beta", 1.0))
        alpha = float(getattr(self.cfg, "fedrl_lip_alpha", 0.1))
        x = (q_pred - q_prior) / (max(delta, 1e-6) ** beta)
        q_feas = q_prior + delta * torch.tanh(x)
        q_out = (1.0 - alpha) * q_feas + alpha * q_prior
        return q_out

    def _map_w1_penalty(self, q_pred: torch.Tensor, q_prior: torch.Tensor) -> torch.Tensor:
        """
        Quantile-huber distance between q_pred and q_prior as trust-region penalty.
        """
        taus = torch.linspace(0.0, 1.0, steps=q_pred.size(-1), device=q_pred.device, dtype=q_pred.dtype)
        kappa = float(getattr(self.cfg, "qr_kappa", 1.0))
        return float(getattr(self.cfg, "fedrl_trust_kappa", 1.0)) * quantile_huber_loss(q_pred, q_prior, taus=taus, kappa=kappa)

    # ----- collection + update -----

    def collect_step(self):
        # actor ego view
        from agents.case_study_2_1.ppo_utils.helpers import ego_list_to_tchw, to_tchw
        ego_bna = ego_list_to_tchw(self.actor_obs).to(self.device)
        glob_b = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)  # [1, C, H, W]
        agent_ids = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

        actions, logps, dists, h_out = self._forward_policy(ego_bna, agent_ids, self.h_actor)
        self.h_actor = h_out.detach()

        act_dict = {i: int(actions[i].item()) for i in range(self.n_agents)}
        (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = self.env.step(act_dict)

        # encode and log one latent triple for server payload (random agent index)
        with torch.no_grad():
            z = self._encode_global(glob_b)  # [1, d]
            j = int(torch.randint(low=0, high=self.n_agents, size=(1,)).item())
            a_j = actions[j:j+1]             # [1]
            q_j = self._critic_quantiles(z, a_j)  # [1, M]
            self.z_buf.append(z.squeeze(0).detach().cpu())
            self.a_buf.append(a_j.squeeze(0).detach().cpu().long())
            self.q_buf.append(q_j.squeeze(0).detach().cpu())

        # commit
        self.actor_obs, self.critic_obs = actor_obs_next, critic_obs_next
        self.total_env_steps += 1

        if terminated or truncated:
            self.h_actor = self.actor.init_hidden(self.n_agents, self.device)
            (self.actor_obs, self.critic_obs), _ = self.env.reset()

    def update_epoch(self):
        """
        One on-policy epoch over a short window:
          - AE reconstruction on current critic_obs batch
          - Distributional Bellman update in latent space
          - Regularization against server prior per mode
        """
        from agents.case_study_2_1.ppo_utils.helpers import to_tchw
        B = int(getattr(self.cfg, "fedrl_mb", 64))
        gamma = float(self.cfg.gamma)

        # current obs
        glob_b = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
        z = self._encode_global(glob_b)  # [1, d]

        # pick a random action for bootstrap target
        a = torch.randint(low=0, high=5, size=(1,), device=self.device).long()
        q_pred = self._critic_quantiles(z, a)  # [1, M]

        # server prior if available
        if self.prior_head_state is not None:
            head_prior = LatentDistHead(self.encoder.d_latent, 5, self.cfg.n_quantiles).to(self.device)
            head_prior.load_state_dict(self.prior_head_state, strict=True)
            q_prior = head_prior(z, a).detach()
        else:
            q_prior = q_pred.detach()

        # simple one-step bootstrap; if you have a rollout buffer, replace with proper next state
        a_next = torch.randint(low=0, high=5, size=(1,), device=self.device).long()
        q_next = self._critic_quantiles(z, a_next).detach()
        r = torch.tensor([float(getattr(self.cfg, "fedrl_bootstrap_r", 0.0))], device=self.device, dtype=q_pred.dtype)
        q_tgt = r.view(-1, 1) + gamma * q_next  # [1, M]

        # apply chosen regularization
        mode = str(getattr(self.cfg, "fedrl_local_reg", "lipschitz")).lower()
        if mode == "lipschitz":
            q_train = self._apply_lipschitz_ops(q_pred, q_prior)
            reg_loss = torch.zeros((), device=self.device)
        else:
            q_train = q_pred
            reg_loss = self._map_w1_penalty(q_pred, q_prior)

        # quantile-huber loss for distributional critic
        taus = torch.linspace(0.0, 1.0, steps=q_pred.size(-1), device=self.device, dtype=q_pred.dtype)
        loss_q = quantile_huber_loss(q_train, q_tgt, taus=taus, kappa=float(getattr(self.cfg, "qr_kappa", 1.0)))

        # autoencoder reconstruction (no detach, so encoder gets recon gradients)
        x_hat = self.decoder(z)
        rec_loss = F.mse_loss(x_hat, glob_b)

        # total
        lam_rec = float(getattr(self.cfg, "fedrl_rec_lambda", 1e-2))
        loss = loss_q + lam_rec * rec_loss + reg_loss

        self.opt_head.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        self.opt_dec.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.opt_head.step()
        self.opt_enc.step()
        self.opt_dec.step()

        self.loss_recent.append(float(loss.item()))
        if wandb.run is not None:
            wandb.log({
                f"client{self.client_id}/fedrl/loss_total": float(loss.item()),
                f"client{self.client_id}/fedrl/loss_q": float(loss_q.item()),
                f"client{self.client_id}/fedrl/loss_rec": float(rec_loss.item()),
                f"client{self.client_id}/env/steps": self.total_env_steps,
            }, step=self.total_env_steps)

    # ----- payload for server -----

    def build_payload(self, max_samples: int = 1024) -> Dict[str, Any]:
        """
        Subsample latent triples. To improve pairing, perform per-action balanced sampling.
        """
        n = len(self.z_buf)
        if n == 0:
            Z = torch.empty(0, self.encoder.d_latent)
            A = torch.empty(0, dtype=torch.long)
            Q = torch.empty(0, self.head.n_quantiles)
        else:
            # indices per action
            A_all = torch.stack(self.a_buf, dim=0).long()  # [n]
            by_action = {a: torch.nonzero(A_all == a, as_tuple=False).squeeze(-1).tolist() for a in range(5)}
            # quota per action
            per_a = max(1, max_samples // 5)
            sel_idx: List[int] = []
            rng = np.random.default_rng()
            for a in range(5):
                idxs = by_action.get(a, [])
                if len(idxs) == 0:
                    continue
                take = min(len(idxs), per_a)
                choice = rng.choice(idxs, size=take, replace=False).tolist()
                sel_idx.extend(choice)
            # if underfull, fill with any remaining
            remaining = max_samples - len(sel_idx)
            if remaining > 0:
                all_idx = list(set(range(n)) - set(sel_idx))
                if len(all_idx) > 0:
                    fill = rng.choice(all_idx, size=min(remaining, len(all_idx)), replace=False).tolist()
                    sel_idx.extend(fill)

            Z = torch.stack([self.z_buf[i] for i in sel_idx], dim=0) if len(sel_idx) > 0 else torch.empty(0, self.encoder.d_latent)
            A = torch.stack([self.a_buf[i] for i in sel_idx], dim=0).long() if len(sel_idx) > 0 else torch.empty(0, dtype=torch.long)
            Q = torch.stack([self.q_buf[i] for i in sel_idx], dim=0) if len(sel_idx) > 0 else torch.empty(0, self.head.n_quantiles)

        # risk proxy: cvar of recent loss tail
        risk_alpha = float(getattr(self.cfg, "fedrl_risk_alpha", 0.9))
        if len(self.loss_recent) >= 16:
            vals = np.array(self.loss_recent[-512:])
            var = np.quantile(vals, risk_alpha)
            cvar = vals[vals >= var].mean() if (vals >= var).any() else float(vals.max())
            base = max(vals.mean(), 1e-6)
            risk = float(np.clip(cvar / base - 1.0, 0.0, 1.0))
        else:
            risk = 0.0

        # state dicts
        enc_sd = {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
        head_sd = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}

        # clear buffers
        self.z_buf.clear(); self.a_buf.clear(); self.q_buf.clear()

        return {
            "encoder_state": enc_sd,
            "head_state": head_sd,
            "latents": {"z": Z, "a": A, "q": Q, "risk": risk},
        }


# ---------------- Client loop ----------------

def _client_loop(rank: int, base_cfg: Any, n_clients: int, threads_per_client: int,
                 cmd_q: mp.Queue, rep_q: mp.Queue, gpu_device: str):
    try:
        _set_threads_per_proc(intra=threads_per_client, inter=1, blas_threads=1)
        time.sleep(0.05 * rank)

        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed) + int(rank)
        cfg.client_id = rank
        cfg.n_clients = n_clients
        cfg.device = gpu_device

        run = _client_wandb_safe_init(
            project=base_cfg.wandb_project,
            name=base_cfg.wandb_run_name,
            cfg_dict=asdict(cfg),
            group=getattr(base_cfg, "wandb_group", None),
            rank=rank,
        )

        env = MultiAgentGridWorld(cfg)
        model = MAPPOModel.build(
            n_actions=5,
            ego_k=cfg.ego_k,
            n_agents=cfg.n_agents,
            critic_type="expected",  # critic unused; we provide latent head instead
            n_quantiles=cfg.n_quantiles,
        )
        tr = FedRLClientTrainer(cfg, env, model, device=cfg.device, client_id=rank)

        # receive first broadcast
        pkg0 = cmd_q.get()
        assert isinstance(pkg0, dict) and pkg0.get("cmd") == "broadcast"
        tr.load_broadcast(pkg0["payload"])

        # main loop
        while True:
            msg = cmd_q.get()
            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd", "")
            if cmd == "shutdown":
                break
            elif cmd == "broadcast":
                tr.load_broadcast(msg["payload"])
            elif cmd == "train_round":
                epochs = int(msg.get("epochs", 1))
                steps_target = int(getattr(cfg, "total_steps", 10_000))
                for _ in range(epochs):
                    rollout_len = int(getattr(cfg, "rollout_len", 64))
                    for _s in range(rollout_len):
                        tr.collect_step()
                        if tr.total_env_steps >= steps_target:
                            break
                    tr.update_epoch()
                    if tr.total_env_steps >= steps_target:
                        break

                payload = tr.build_payload(max_samples=int(getattr(cfg, "fedrl_payload_max", 1024)))
                rep_q.put({
                    "rank": rank,
                    "encoder_state": payload["encoder_state"],
                    "head_state": payload["head_state"],
                    "latents": payload["latents"],
                })
            else:
                continue

        if run is not None:
            run.finish()

    except Exception as e:
        rep_q.put({"rank": rank, "error": str(e)})


# ---------------- Runner ----------------

def run_fedrl(cfg: Any) -> Tuple[str, str]:
    """
    Entry point similar to run_fedavg.
    Returns (encoder_ckpt_path, latent_head_ckpt_path)
    """
    device = str(cfg.device)
    n_clients = int(cfg.n_clients)
    total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
    threads_per_client = max(1, total_cpus // max(1, n_clients))

    # server
    d_latent = int(getattr(cfg, "fedrl_d_latent", 128))
    n_quantiles = int(cfg.n_quantiles)
    server = FedRLServer(cfg, d_latent=d_latent, n_actions=5, n_quantiles=n_quantiles, device=device)
    save_dir = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_dir, exist_ok=True)

    # epoch budgets from weights
    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    def _epochs(local_epochs_per_round: int, n_clients: int, weights) -> List[int]:
        out = []
        for i in range(n_clients):
            e_i = max(1, math.ceil(local_epochs_per_round * float(weights[i]) * n_clients))
            out.append(int(e_i))
        return out
    epochs_per_client = _epochs(cfg.local_epochs_per_round, cfg.n_clients, w)

    # spawn clients
    ctx = mp.get_context("spawn")
    cmd_queues: List[mp.Queue] = []
    rep_queues: List[mp.Queue] = []
    procs: List[mp.Process] = []
    for rank in range(n_clients):
        cq = ctx.Queue()
        rq = ctx.Queue()
        p = ctx.Process(
            target=_client_loop,
            args=(rank, cfg, n_clients, threads_per_client, cq, rq, device),
        )
        p.daemon = False
        p.start()
        cmd_queues.append(cq)
        rep_queues.append(rq)
        procs.append(p)

    # initial broadcast (encoder/head randomly init on server)
    def _encoder_builder():
        from models.fedrl_nets import ConvEncoder
        return ConvEncoder(in_ch=6, d_latent=d_latent)

    init_pkg = server.package_broadcast()
    for cq in cmd_queues:
        cq.put({"cmd": "broadcast", "payload": init_pkg})

    # rounds
    for rnd in range(int(cfg.num_communication_rounds)):
        server.round_idx = rnd

        # trigger client training with budgets
        for rank in range(n_clients):
            cmd_queues[rank].put({"cmd": "train_round", "epochs": int(epochs_per_client[rank])})

        # gather
        enc_states: List[Tuple[Dict[str, torch.Tensor], float]] = []
        head_states: List[Tuple[Dict[str, torch.Tensor], float]] = []
        latents_all: List[Dict[str, Any]] = []
        for rank in range(n_clients):
            msg = rep_queues[rank].get()
            if "error" in msg:
                raise RuntimeError(f"client {rank} failed: {msg['error']}")
            enc_states.append((msg["encoder_state"], float(w[rank])))
            head_states.append((msg["head_state"], float(w[rank])))
            latents_all.append(msg["latents"])

        # aggregate + refit
        server.aggregate_and_refit(
            encoder_states=enc_states,
            head_states=head_states,
            latent_payloads=latents_all,
            encoder_builder=_encoder_builder
        )

        # broadcast updated encoder + head
        pkg = server.package_broadcast()
        for cq in cmd_queues:
            cq.put({"cmd": "broadcast", "payload": pkg})

        # checkpoint periodically
        if (rnd + 1) % max(1, int(getattr(cfg, "ckpt_every_rounds", 5))) == 0 or (rnd + 1) == int(cfg.num_communication_rounds):
            server.save_checkpoint(save_dir, tag=f"round{rnd+1}")

    for cq in cmd_queues:
        cq.put({"cmd": "shutdown"})
    for p in procs:
        p.join()

    enc_path, head_path = server.save_checkpoint(save_dir, tag="final")
    server.finish()
    return enc_path, head_path
