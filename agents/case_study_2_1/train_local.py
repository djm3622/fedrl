#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_local.py — Multi-process single-GPU, N independent local clients (no federation).

- Spawns N long-lived client processes (one per client).
- Each client:
    * sets a unique seed (base + rank) and mild env heterogeneity (hazard/slip).
    * creates its own W&B run with a stable id: <run_name>_local_client{rank}.
    * trains for a fixed local epoch budget (no aggregation or parameter exchange).
    * saves actor/critic weights under outputs/<project>/<run_name>/client_{rank}/
      and logs them as artifacts to that client's W&B run.
"""

from __future__ import annotations
import os
import time
import math
import copy
from dataclasses import asdict
from typing import Any, List, Tuple

import torch
import torch.multiprocessing as mp
import wandb

from agents.case_study_2_1.ppo_utils.trainer import PPOTrainer
from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel
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


def _compute_epochs_per_client(local_epochs_per_round: int, n_clients: int, weights) -> list[int]:
    """
    Allocate local epochs proportional to client_weights.
    Sum_i epochs_i ~= local_epochs_per_round * n_clients.
    """
    out = []
    for i in range(n_clients):
        e_i = max(1, math.ceil(local_epochs_per_round * float(weights[i]) * n_clients))
        out.append(int(e_i))
    return out


def _model_builder_from_cfg(cfg) -> MAPPOModel:
    return MAPPOModel.build(
        n_actions=5,
        ego_k=cfg.ego_k,
        n_agents=cfg.n_agents,
        critic_type=cfg.param_type,
        n_quantiles=cfg.n_quantiles,
    )


def _client_wandb_safe_init(project: str, name: str, cfg_dict: dict, group: str | None, rank: int):
    """
    One stable W&B run per client. Uses id=name_local_client{rank} so resumes are clean.
    """
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")

    base_name = f"{name}_local_client{rank}"
    run_id = base_name

    run_dir = os.path.join(
        os.getenv("SLURM_TMPDIR", os.path.join(os.getcwd(), "wandb_cache")),
        f"client_{rank}"
    )
    os.makedirs(run_dir, exist_ok=True)

    try:
        return wandb.init(
            project=project,
            name=base_name,
            id=run_id,
            resume="allow",
            config=cfg_dict,
            group=(group if group is not None else f"{name}_local"),
            dir=run_dir,
        )
    except Exception as e:
        print(f"[wandb] client {rank} init failed: {e}  (continuing without W&B)")
        return None


def _client_worker(
    rank: int,
    base_cfg: Any,
    n_clients: int,
    threads_per_client: int,
    total_epochs_for_client: int,
    gpu_device: str,
    out_dir: str,
):
    try:
        _set_threads_per_proc(intra=threads_per_client, inter=1, blas_threads=1)
        time.sleep(0.05 * rank)

        # --- per-client config (seed + heterogeneity) ---
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed) + int(rank)
        cfg.client_id = rank
        cfg.n_clients = n_clients
        cfg.device = gpu_device

        hazard_delta = float(os.getenv("HAZARD_DELTA", "0.05"))
        slip_delta   = float(os.getenv("SLIP_DELTA",   "0.02"))
        cfg.hazard_prob = _spread_value(base_cfg.hazard_prob, hazard_delta, rank, n_clients)
        cfg.slip_prob   = _spread_value(base_cfg.slip_prob,   slip_delta,   rank, n_clients)

        run = _client_wandb_safe_init(
            project=base_cfg.wandb_project,
            name=base_cfg.wandb_run_name,
            cfg_dict=asdict(cfg),
            group=getattr(base_cfg, "wandb_group", None),
            rank=rank,
        )

        # --- build env/model/trainer and train for the allocated epochs ---
        env = MultiAgentGridWorld(cfg)
        model = _model_builder_from_cfg(cfg)
        model.actor.to(cfg.device)
        model.critic.to(cfg.device)

        tr = PPOTrainer(cfg, env, model, device=cfg.device, client_id=rank)

        # Train for the total client budget (as epochs)
        _epochs_remaining = int(total_epochs_for_client)
        while _epochs_remaining > 0 and tr.total_env_steps < cfg.total_steps:
            step_before = tr.total_env_steps
            tr.train_for_epochs(1)
            _epochs_remaining -= 1
            # safety: if rollout_len is huge and total_steps is tight
            if tr.total_env_steps == step_before:
                break

        # --- save per-client weights ---
        client_dir = os.path.join(out_dir, f"client_{rank}")
        os.makedirs(client_dir, exist_ok=True)
        actor_path = os.path.join(client_dir, "actor.pth")
        critic_path = os.path.join(client_dir, "critic.pth")
        torch.save(model.actor.state_dict(), actor_path)
        torch.save(model.critic.state_dict(), critic_path)

        if run is not None:
            art_a = wandb.Artifact(name=f"actor_client{rank}", type="weights")
            art_a.add_file(actor_path)
            run.log_artifact(art_a)
            art_c = wandb.Artifact(name=f"critic_client{rank}", type="weights")
            art_c.add_file(critic_path)
            run.log_artifact(art_c)
            run.finish()

    except Exception as e:
        # Print error to stderr for visibility; no central queue in local baseline
        import sys, traceback
        print(f"[local client {rank}] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()


def run_local_multi(cfg: Any) -> List[Tuple[str, str]]:
    """
    Launch N independent local clients, each training for an epoch budget proportional to client_weights.
    Returns list of (actor_path, critic_path) per client.
    """
    device = str(cfg.device)
    n_clients = int(cfg.n_clients)
    total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
    threads_per_client = max(1, total_cpus // max(1, n_clients))

    save_root = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_root, exist_ok=True)

    # total local epochs per client (proportional to weights), matched to federated budget
    epochs_per_client = _compute_epochs_per_client(
        local_epochs_per_round=cfg.local_epochs_per_round,
        num_rounds=cfg.num_communication_rounds,
        n_clients=n_clients,
        weights=cfg.client_weights,
    )

    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []
    for rank in range(n_clients):
        p = ctx.Process(
            target=_client_worker,
            args=(rank, cfg, n_clients, threads_per_client, int(epochs_per_client[rank]), device, save_root),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # collect paths deterministically
    out: List[Tuple[str, str]] = []
    for rank in range(n_clients):
        client_dir = os.path.join(save_root, f"client_{rank}")
        out.append((
            os.path.join(client_dir, "actor.pth"),
            os.path.join(client_dir, "critic.pth"),
        ))
    return out
