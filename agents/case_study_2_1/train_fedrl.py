#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fedrl.py — Multi-process single-GPU FedRL with critic prior.

- Each client runs standard PPO (MAPPO).
- During a round, each client collects distributional critic outputs into a
  per-round buffer (if the critic is distributional and the buffer is enabled).
- At the end of the round, each client summarizes its visited states into a
  single barycentric prior over critic quantiles (per-client, per-round).
- Clients still send their full critic state to the server (FedAvg-compatible)
  and additionally send the per-round prior quantiles. The server can use one
  or both to construct the forward-time prior for the next round.

Notes:
- Clients DO NOT overwrite local critic weights with the server aggregate.
  Instead, they update a frozen prior head inside the critic and enable
  affine shrink + soft tanh tube in the forward pass.
- Autoencoder flags are ignored on clients.
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
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
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
      - 'broadcast': receive server payload and configure critic prior
      - 'train_round': run local PPO for given epochs, then reply with:
            * local critic state (FedAvg-compatible)
            * per-round distributional prior quantiles (if enabled)
            * round-level metrics (hazard_rate, etc.)
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
        slip_delta = float(os.getenv("SLIP_DELTA", "0.02"))
        cfg.hazard_prob = _spread_value(base_cfg.hazard_prob, hazard_delta, rank, n_clients)
        cfg.slip_prob = _spread_value(base_cfg.slip_prob, slip_delta, rank, n_clients)

        # force ae off on clients
        setattr(cfg, "enable_ae_aux", False)

        # enable distributional prior buffer on FedRL runs (trainer still gates on critic type)
        setattr(cfg, "enable_dist_prior_buffer", True)

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
            """
            Configure the client critic from the server payload.

            New behavior:
              - If a global prior quantile vector is provided and the critic is
                distributional, use it as a state-independent distributional prior.
              - Enable prior regularization in the forward pass.
              - Optionally fall back to the old FedAvg-style prior if no vector
                is provided or the critic is not distributional.
            """
            prior_q = pkg.get("prior_quantiles", None)
            crit_sd = pkg.get("critic", None)

            is_dist_critic = getattr(trainer, "is_dist", hasattr(trainer.model.critic, "taus"))

            try:
                if prior_q is not None and is_dist_critic and hasattr(trainer.model, "set_global_prior_quantiles"):
                    print(f"[client {rank}] setting critic prior from quantile vector")
                    # Set global vector prior on distributional critic
                    trainer.model.set_global_prior_quantiles(prior_q)

                    # Enable forward-time prior regularization
                    if hasattr(trainer.model, "set_prior_regularization"):
                        trainer.model.set_prior_regularization(
                            enabled=getattr(trainer.cfg, "enabled", True),
                            alpha=float(getattr(trainer.cfg, "alpha", 0.9)),
                            beta=float(getattr(trainer.cfg, "beta", 1.0)),
                            radius_abs=float(getattr(trainer.cfg, "radius_abs", 0.0)),
                            radius_rel=float(getattr(trainer.cfg, "radius_rel", 1.0)),
                        )
                elif crit_sd is not None and getattr(trainer.model, "critic", None) is not None:
                    # Fallback: old behavior using a full critic prior
                    print(f"[client {rank}] setting critic prior from full state dict")
                    if hasattr(trainer.model, "update_critic_prior"):
                        trainer.model.update_critic_prior(crit_sd)
                    if hasattr(trainer.model, "set_prior_regularization"):
                        trainer.model.set_prior_regularization(
                            enabled=getattr(trainer.cfg, "enabled", True),
                            alpha=float(getattr(trainer.cfg, "alpha", 0.9)),
                            beta=float(getattr(trainer.cfg, "beta", 1.0)),
                            radius_abs=float(getattr(trainer.cfg, "radius_abs", 0.0)),
                            radius_rel=float(getattr(trainer.cfg, "radius_rel", 1.0)),
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

                # Build per-round distributional prior (barycenter) if enabled
                prior_quantiles = None
                if getattr(trainer, "enable_dist_prior_buffer", False):
                    try:
                        q_prior = trainer.build_round_distributional_prior()
                        if q_prior is not None:
                            prior_quantiles = q_prior.detach().cpu()

                            # --- NEW: log barycenter using wandb.Table + wandb.plot.histogram ---
                            if run is not None:
                                try:
                                    arr = prior_quantiles.numpy().astype("float32")

                                    # x-axis: quantile index; y-axis: quantile value
                                    xs = list(range(len(arr)))
                                    ys = arr.tolist()

                                    line = wandb.plot.line_series(
                                        xs=[xs],
                                        ys=[ys],
                                        keys=[f"client{rank}"],
                                        title=f"Client {rank} prior barycenter (quantiles)",
                                        xname="quantile_index",
                                    )

                                    run.log(
                                        {
                                            "fedrl/prior_barycenter_mean": float(arr.mean()),
                                            "fedrl/prior_barycenter_min": float(arr.min()),
                                            "fedrl/prior_barycenter_max": float(arr.max()),
                                            "fedrl/prior_barycenter_line": line,
                                        },
                                        step=int(trainer.total_env_steps),
                                    )

                                except Exception as e:
                                    print(f"[client {rank}] failed to log barycenter: {e}")
                            # --- end NEW ---

                    except Exception as e:
                        print(f"[client {rank}] failed to build dist prior: {e}")

                # reply with local critic state + optional prior quantiles
                crit_state = None
                if getattr(trainer.model, "critic", None) is not None:
                    try:
                        crit_state = {
                            k: v.detach().cpu()
                            for k, v in trainer.model.critic.state_dict().items()
                        }
                    except Exception:
                        crit_state = None

                rep_q.put({
                    "rank": rank,
                    "critic_state": crit_state,
                    "prior_quantiles": prior_quantiles,
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
    Multi-process FedRL runner.

    For now, the server still aggregates critic states via FedAvg (or a
    hazard-weighted variant if enabled in FedRLServer). Clients also send
    per-round distributional priors (prior_quantiles), which can be used
    to replace or augment the FedAvg logic in a subsequent step.

    Returns:
      (critic_ckpt_path or "", "").
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

        # Gather critics + priors + metrics from clients
        raw_msgs: List[Dict[str, Any]] = []
        for rank in range(n_clients):
            msg = rep_queues[rank].get()
            if "error" in msg:
                raise RuntimeError(f"client {rank} failed: {msg['error']}")
            raw_msgs.append(msg)

        # Build hazard_metrics aligned with client messages (kept for logging)
        hazard_metrics: List[Optional[float]] = []
        for msg in raw_msgs:
            m = msg.get("metrics", {}) or {}
            hazard_metrics.append(float(m.get("hazard_rate", 0.0)))

        # Collect per-client distributional priors
        client_priors: List[torch.Tensor] = []
        for msg in raw_msgs:
            q_i = msg.get("prior_quantiles", None)
            if q_i is not None:
                client_priors.append(q_i.detach().cpu())

        # Arithmetic average of all client priors (if any)
        q_global: Optional[torch.Tensor]
        if client_priors:
            stacked = torch.stack(client_priors, dim=0)  # [K, Nq]
            q_global = stacked.mean(dim=0)               # [Nq]
        else:
            q_global = None

        # Set the global distributional prior on the server
        server.set_global_prior_quantiles(q_global)

        # Note: we do not need FedAvg over critic states for this method.
        # critic_states and aggregate_and_refit are not used here.

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
