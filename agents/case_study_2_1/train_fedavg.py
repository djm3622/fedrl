#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fedavg.py — Multi-process single-GPU FedAvg (critic-only).

Broadcasts/aggregates ONLY the critic parameters.
Actors remain local; no actor parameters are sent or received.
"""

from __future__ import annotations
import os
import time
import math
import copy
from dataclasses import asdict
from typing import Any, Dict, Tuple, List

import torch
import torch.multiprocessing as mp
import wandb

from federation.case_study_2_1.fedavg import (
    FedAvgServer,
    pack_critic_state,
    unpack_critic_state_into,
)
from utils.general import normalize_weights
from agents.case_study_2_1.ppo_utils.trainer import PPOTrainer
from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel


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


def _compute_client_epochs(local_epochs_per_round: int, n_clients: int, weights) -> int:
    return max(1, math.ceil(local_epochs_per_round * (weights[0] * n_clients)))


def _model_builder_from_cfg(cfg) -> MAPPOModel:
    return MAPPOModel.build(
        n_actions=5,
        ego_k=cfg.ego_k,
        n_agents=cfg.n_agents,
        critic_type=cfg.param_type,
        n_quantiles=cfg.n_quantiles,
    )


def _client_wandb_safe_init(project: str, name: str, cfg_dict: dict, group: str | None, rank: int):
    # Offline mode if no API key to avoid auth/network errors (LibreSSL, etc.)
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")

    # Per-client isolated directory (prevents file contention on clusters)
    run_dir = os.path.join(
        os.getenv("SLURM_TMPDIR", os.path.join(os.getcwd(), "wandb_cache")),
        f"client_{rank}"
    )
    os.makedirs(run_dir, exist_ok=True)

    settings = wandb.Settings(start_method="thread")
    try:
        return wandb.init(
            project=project,
            name=name,
            config=cfg_dict,
            group=group,          # e.g., group by run_name so server + clients are grouped
            dir=run_dir,
            settings=settings,
            reinit=True,
        )
    except Exception as e:
        print(f"[wandb] client {rank} init failed: {e}  (continuing without W&B)")
        return None


def _client_worker(
    rank: int,
    base_cfg: Any,
    n_clients: int,
    threads_per_client: int,
    global_critic_flat: Dict[str, torch.Tensor],
    client_weight: float,
    epochs_per_round: int,
    queue: mp.Queue,
    gpu_device: str,
):
    try:
        _set_threads_per_proc(intra=threads_per_client, inter=1, blas_threads=1)
        time.sleep(0.05 * rank)

        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed) + int(rank)
        cfg.client_id = rank
        cfg.n_clients = n_clients
        cfg.device = gpu_device

        # --- INIT A CLIENT RUN BEFORE ANY wandb.log IS CALLED ---
        run = _client_wandb_safe_init(
            project=base_cfg.wandb_project,
            name=f"{base_cfg.wandb_run_name}_client{rank}",
            cfg_dict=asdict(base_cfg),
            group=getattr(base_cfg, "wandb_group", base_cfg.wandb_run_name),
            rank=rank,
        )

        # deterministic variations
        hazard_delta = float(os.getenv("HAZARD_DELTA", "0.05"))
        slip_delta   = float(os.getenv("SLIP_DELTA",   "0.02"))
        cfg.hazard_prob = _spread_value(base_cfg.hazard_prob, hazard_delta, rank, n_clients)
        cfg.slip_prob   = _spread_value(base_cfg.slip_prob,   slip_delta,   rank, n_clients)

        # build env + model; load ONLY critic from server; actor stays local
        env = MultiAgentGridWorld(cfg)
        model = _model_builder_from_cfg(cfg)
        unpack_critic_state_into(model, global_critic_flat)
        model.actor.to(cfg.device)
        model.critic.to(cfg.device)

        tr = PPOTrainer(cfg, env, model, device=cfg.device, client_id=rank)

        total_steps_start = tr.total_env_steps
        for _ in range(int(epochs_per_round)):
            while not tr.roll.full():
                tr.collect_rollout_step()
                if tr.total_env_steps >= cfg.total_steps:
                    break
            if tr.roll.ptr > 0:
                tr.update_epoch()
            tr.maybe_eval()
            if tr.total_env_steps >= cfg.total_steps:
                break

        num_samples = int(tr.total_env_steps - total_steps_start)
        critic_flat = pack_critic_state(tr.model)

        if run is not None:
            run.finish()

        queue.put({
            "rank": rank,
            "state_dict": critic_flat,
            "weight": float(client_weight),
            "num_samples": num_samples,
        })
    except Exception as e:
        queue.put({"rank": rank, "error": str(e)})


def run_fedavg(cfg: Any) -> Tuple[str, str]:
    """
    Orchestrates FedAvg rounds (critic-only). Returns (actor_path, critic_path) of server snapshot.
    """
    device = str(cfg.device)
    n_clients = int(cfg.n_clients)
    total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
    threads_per_client = max(1, total_cpus // max(1, n_clients))

    server = FedAvgServer(cfg, lambda: _model_builder_from_cfg(cfg), device=device)
    save_dir = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_dir, exist_ok=True)

    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    epochs_per_round = _compute_client_epochs(cfg.local_epochs_per_round, cfg.n_clients, w)

    ctx = mp.get_context("spawn")
    for rnd in range(int(cfg.num_communication_rounds)):
        server.round_idx = rnd
        global_critic_flat = server.get_global_critic_state()  # critic only

        queue: mp.Queue = ctx.Queue()
        procs: List[mp.Process] = []
        for rank in range(n_clients):
            p = ctx.Process(
                target=_client_worker,
                args=(
                    rank,
                    cfg,
                    n_clients,
                    threads_per_client,
                    global_critic_flat,
                    float(w[rank]),
                    int(epochs_per_round),
                    queue,
                    device,
                ),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        payloads: List[Tuple[Dict[str, torch.Tensor], float, int]] = []
        for _ in range(n_clients):
            msg = queue.get()
            if "error" in msg:
                raise RuntimeError(f"client {msg.get('rank')} failed: {msg['error']}")
            payloads.append((msg["state_dict"], float(msg["weight"]), int(msg["num_samples"])))

        for p in procs:
            p.join()

        server.aggregate(payloads)
        if (rnd + 1) % max(1, int(getattr(cfg, "ckpt_every_rounds", 5))) == 0 or (rnd + 1) == int(cfg.num_communication_rounds):
            server.save_checkpoint(save_dir, tag=f"round{rnd+1}")

    actor_path, critic_path = server.save_checkpoint(save_dir, tag="final")
    server.finish()
    return actor_path, critic_path
