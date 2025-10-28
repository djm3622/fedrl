#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fedrl.py — Multi-process single-GPU FedAvg over the CRITIC (MAPPO), used as a forward-time prior.
- Each client runs standard PPO (MAPPO).
- After each round, clients send their critic state to server.
- Server performs weighted parameter averaging and broadcasts the averaged critic back.
- Clients DO NOT load the averaged weights into their critic. Instead:
    * they update a frozen prior head inside the critic
    * they enable affine shrink + soft tanh tube in forward
- Autoencoder flags are ignored; no barycenter logic.

Notes:
- Trust-region blending and hazard EMA are implemented server-side in FedRLServer.
- No changes needed here beyond config values the user may set:
    * agg_trust_xi (float, default 0.1)
    * agg_hazard_ema_rho (float, default 0.9; set 0.0 to disable)
"""

from __future__ import annotations
import os
import time
import math
import copy
from dataclasses import asdict
from typing import Any, Tuple, List, Dict, Optional

import torch
import torch.multiprocessing as mp
import wandb

from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel
from agents.case_study_2_1.ppo_utils.trainer import PPOTrainer
from federation.case_study_2_1.fedrl import FedRLServer
from utils.general import normalize_weights


# ---------------- CPU threading ----------------
def _set_threads_per_proc(intra: int, inter: int = 1, blas_threads: int = 1):
    os.environ["OMP_NUM_THREADS"] = str(intra)
    os.environ["MKL_NUM_THREADS"] = str(intra)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    torch.set_num_threads(intra)
    torch.set_num_interop_threads(inter)

def _spread_value(center: float, delta: float, rank: int, n_clients: int) -> float:
    if n_clients <= 1 or delta <= 0:
        return float(center)
    t = rank / (n_clients - 1)
    return float(max(0.0, min(1.0, (center - delta) + (2.0 * delta) * t)))

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


def _compute_epochs_per_client(local_epochs_per_round: int, n_clients: int, weights) -> list[int]:
    """
    Allocate local epochs proportional to normalized client_weights.
    Sum_i epochs_i ~= local_epochs_per_round * n_clients.
    Guarantees at least 1 epoch per client.
    """
    out = []
    for i in range(n_clients):
        e_i = max(1, math.ceil(local_epochs_per_round * float(weights[i]) * n_clients))
        out.append(int(e_i))
    return out


# ---------------- Client loop ----------------
def _client_loop(
    rank: int,
    base_cfg: Any,
    n_clients: int,
    threads_per_client: int,
    gpu_device: str,
    cmd_q: mp.Queue,
    rep_q: mp.Queue
):
    """
    Simple command loop:
      - 'broadcast': receive averaged critic sd and set it as the prior (no weight overwrite)
      - 'train_round': run local PPO for given epochs, then reply with local critic sd
      - 'shutdown': exit
    """
    try:
        _set_threads_per_proc(intra=threads_per_client, inter=1, blas_threads=1)
        time.sleep(0.05 * rank)

        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed) + int(rank)
        cfg.client_id = rank
        cfg.device = gpu_device
        cfg.n_clients = n_clients

        # per-client deterministic variations
        hazard_delta = float(os.getenv("HAZARD_DELTA", "0.05"))
        slip_delta   = float(os.getenv("SLIP_DELTA",   "0.02"))
        cfg.hazard_prob = _spread_value(base_cfg.hazard_prob, hazard_delta, rank, n_clients)
        cfg.slip_prob   = _spread_value(base_cfg.slip_prob,   slip_delta,   rank, n_clients)

        # force ae off on clients
        setattr(cfg, "enable_ae_aux", False)

        run = _client_wandb_safe_init(
            project=base_cfg.wandb_project,
            name=base_cfg.wandb_run_name,
            cfg_dict=asdict(cfg),
            group=getattr(base_cfg, "wandb_group", None),
            rank=rank,
        )

        # build env + model + trainer
        env = MultiAgentGridWorld(cfg)
        model = MAPPOModel.build(
            n_actions=5,
            ego_k=cfg.ego_k,
            n_agents=cfg.n_agents,
            critic_type=getattr(cfg, "param_type", "expected"),
            n_quantiles=cfg.n_quantiles,
        )
        trainer = PPOTrainer(cfg, env, model, device=cfg.device, client_id=rank, wb_step_base=0)

        def _load_critic_from_pkg(pkg: Dict[str, Any]):
            crit_sd = pkg.get("critic", None)
            if crit_sd is not None and getattr(trainer.model, "critic", None) is not None:
                try:
                    # DO NOT overwrite local critic weights; update frozen prior head only
                    if hasattr(trainer.model, "update_critic_prior"):
                        trainer.model.update_critic_prior(crit_sd)
                    # enable forward-time prior regularization
                    if hasattr(trainer.model, "set_prior_regularization"):
                        trainer.model.set_prior_regularization(
                            enabled=getattr(trainer.cfg, "prior_enabled", True),
                            alpha=float(getattr(trainer.cfg, "prior_alpha", 0.5)),
                            beta=float(getattr(trainer.cfg, "prior_beta", 1.0)),
                            radius_abs=float(getattr(trainer.cfg, "prior_radius_abs", 0.0)),
                            radius_rel=float(getattr(trainer.cfg, "prior_radius_rel", 0.10)),
                        )
                    trainer.model.critic.to(trainer.device)
                except Exception as e:
                    print(f"[client {rank}] failed to set critic prior: {e}")

        while True:
            msg = cmd_q.get()
            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd", "")
            if cmd == "shutdown":
                break
            elif cmd == "broadcast":
                _load_critic_from_pkg(msg.get("payload", {}))
            elif cmd == "train_round":
                epochs = int(msg.get("epochs", 1))
                for _ in range(max(1, epochs)):
                    trainer.train_for_epochs(n_epochs=1)
                    if trainer.total_env_steps >= cfg.total_steps:
                        break

                # reply with local critic state (to be averaged on server)
                crit_state = None
                if getattr(trainer.model, "critic", None) is not None:
                    try:
                        crit_state = {k: v.detach().cpu() for k, v in trainer.model.critic.state_dict().items()}
                    except Exception:
                        crit_state = None

                rep_q.put({
                    "rank": rank,
                    "critic_state": crit_state,
                    "metrics": trainer.get_round_metrics(reset=True),  # hazard_rate etc.
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
    Multi-process FedAvg over the critic only (server-side), used as forward-time prior on clients.
    Returns (critic_ckpt_path or "", "").
    """
    device = str(cfg.device)
    n_clients = int(getattr(cfg, "n_clients", 1))
    total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
    threads_per_client = max(1, total_cpus // max(1, n_clients))

    # Server
    server = FedRLServer(cfg, device=device)

    # Normalize weights and allocate per-client epochs
    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    epochs_per_client = _compute_epochs_per_client(cfg.local_epochs_per_round, cfg.n_clients, w)

    # Spawn clients
    ctx = mp.get_context("spawn")
    cmd_queues: List[mp.Queue] = []
    rep_queues: List[mp.Queue] = []
    procs: List[mp.Process] = []

    for rank in range(n_clients):
        cq = ctx.Queue()
        rq = ctx.Queue()
        p = ctx.Process(target=_client_loop, args=(rank, cfg, n_clients, threads_per_client, device, cq, rq))
        p.daemon = False
        p.start()
        cmd_queues.append(cq)
        rep_queues.append(rq)
        procs.append(p)

    # Initial broadcast (may be empty on round 0)
    pkg0 = server.package_broadcast()
    for cq in cmd_queues:
        cq.put({"cmd": "broadcast", "payload": pkg0})

    # Rounds
    R = int(getattr(cfg, "num_communication_rounds", 1))
    for r in range(R):
        server.round_idx = r

        # Ask clients to train with their budget
        for rank in range(n_clients):
            cmd_queues[rank].put({"cmd": "train_round", "epochs": int(epochs_per_client[rank])})

        # Gather critics + metrics from clients
        raw_msgs: List[Dict[str, Any]] = []
        for rank in range(n_clients):
            msg = rep_queues[rank].get()
            if "error" in msg:
                raise RuntimeError(f"client {rank} failed: {msg['error']}")
            raw_msgs.append(msg)

        # Build hazard_metrics aligned with client messages
        hazard_metrics: List[Optional[float]] = []
        for msg in raw_msgs:
            m = msg.get("metrics", {}) or {}
            hazard_metrics.append(float(m.get("hazard_rate", 0.0)))

        # Package critic states (weights here are dummies; server will ignore when hazard_metrics is provided)
        crit_states: List[Tuple[Optional[Dict[str, torch.Tensor]], float]] = []
        for msg in raw_msgs:
            crit_states.append((msg.get("critic_state", None), 1.0))

        # Aggregate with hazard-weighted prior (critic only) + trust-region blending on server
        server.aggregate_and_refit(
            critic_states=crit_states,
            hazard_metrics=hazard_metrics,
        )

        # Broadcast the averaged critic back out (as prior only)
        pkg = server.package_broadcast()
        for cq in cmd_queues:
            cq.put({"cmd": "broadcast", "payload": pkg})

        # Optional checkpoint cadence
        if (r + 1) % max(1, int(getattr(cfg, "ckpt_every_rounds", R))) == 0:
            save_dir = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
            os.makedirs(save_dir, exist_ok=True)
            server.save_checkpoint(save_dir, tag=f"round{r+1}")

    # Shutdown clients
    for cq in cmd_queues:
        cq.put({"cmd": "shutdown"})
    for p in procs:
        p.join()

    # Final save on server
    save_dir = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_dir, exist_ok=True)
    crit_path = server.save_checkpoint(save_dir, tag="final")
    server.finish()

    return (crit_path or "", "")
