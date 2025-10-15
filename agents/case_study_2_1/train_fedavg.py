#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fedavg.py — Multi-process single-GPU FedAvg (critic-only), long-lived clients.

Protocol:
  - Parent spawns N client processes once.
  - For each round:
      * parent sends {"cmd":"train_round", "critic": <flat critic state>, "epochs": <int>}
      * client loads critic, trains for 'epochs' via PPOTrainer.train_for_epochs, replies with
        {"cmd":"round_done", "critic": <flat critic state>, "num_samples": <int>}
      * parent aggregates critic via FedAvg and proceeds to next round.
  - On completion: parent sends {"cmd":"shutdown"}; clients finish and exit.
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
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")
    run_id = f"{name}_client{rank}"  # stable id per client
    run_dir = os.path.join(
        os.getenv("SLURM_TMPDIR", os.path.join(os.getcwd(), "wandb_cache")),
        f"client_{rank}"
    )
    os.makedirs(run_dir, exist_ok=True)
    settings = wandb.Settings(start_method="thread")
    try:
        # keep a single long-lived run; no finish() until shutdown
        return wandb.init(
            project=project,
            name=run_id,
            id=run_id,
            resume="allow",
            config=cfg_dict,
            group=group,
            dir=run_dir,
            settings=settings,
        )
    except Exception as e:
        print(f"[wandb] client {rank} init failed: {e}  (continuing without W&B)")
        return None


def _client_loop(
    rank: int,
    base_cfg: Any,
    n_clients: int,
    threads_per_client: int,
    cmd_q: mp.Queue,
    rep_q: mp.Queue,
    gpu_device: str,
):
    """
    Long-lived client process. Builds env, model, PPOTrainer once.
    Receives 'train_round' commands with broadcast critic, trains, replies with updated critic.
    """
    try:
        _set_threads_per_proc(intra=threads_per_client, inter=1, blas_threads=1)
        time.sleep(0.05 * rank)

        cfg = copy.deepcopy(base_cfg)
        cfg.seed = int(base_cfg.seed) + int(rank)
        cfg.client_id = rank
        cfg.n_clients = n_clients
        cfg.device = gpu_device

        # per-client deterministic variations
        hazard_delta = float(os.getenv("HAZARD_DELTA", "0.05"))
        slip_delta   = float(os.getenv("SLIP_DELTA",   "0.02"))
        cfg.hazard_prob = _spread_value(base_cfg.hazard_prob, hazard_delta, rank, n_clients)
        cfg.slip_prob   = _spread_value(base_cfg.slip_prob,   slip_delta,   rank, n_clients)

        # one W&B run per client
        run = _client_wandb_safe_init(
            project=base_cfg.wandb_project,
            name=base_cfg.wandb_run_name,
            cfg_dict=asdict(base_cfg),
            group=getattr(base_cfg, "wandb_group", base_cfg.wandb_run_name),
            rank=rank,
        )

        # build env + model + trainer once
        env = MultiAgentGridWorld(cfg)
        model = _model_builder_from_cfg(cfg)
        model.actor.to(cfg.device)
        model.critic.to(cfg.device)

        tr = PPOTrainer(cfg, env, model, device=cfg.device, client_id=rank)

        # main command loop
        while True:
            msg = cmd_q.get()  # blocking
            if not isinstance(msg, dict) or "cmd" not in msg:
                continue
            cmd = msg["cmd"]
            if cmd == "shutdown":
                break
            elif cmd == "train_round":
                critic_flat = msg["critic"]
                epochs = int(msg.get("epochs", 1))

                # load broadcast critic and reset critic optimizer moments
                tr.load_critic_flat(critic_flat, reset_opt=True)
                # train for requested epochs
                num_samples = tr.train_for_epochs(epochs)

                # return new critic snapshot
                rep_q.put({
                    "rank": rank,
                    "state_dict": pack_critic_state(tr.model),
                    "num_samples": int(num_samples),
                })
            else:
                # ignore unknown
                continue

        if run is not None:
            run.finish()

    except Exception as e:
        rep_q.put({"rank": rank, "error": str(e)})


def run_fedavg(cfg: Any) -> Tuple[str, str]:
    """
    Orchestrates FedAvg rounds (critic-only). Returns (actor_path, critic_path) of server snapshot.
    """
    device = str(cfg.device)
    n_clients = int(cfg.n_clients)
    total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
    threads_per_client = max(1, total_cpus // max(1, n_clients))

    # server with initial global model
    server = FedAvgServer(cfg, lambda: _model_builder_from_cfg(cfg), device=device)
    save_dir = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_dir, exist_ok=True)

    # FedAvg weights and per-round local epochs
    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    epochs_per_round = _compute_client_epochs(cfg.local_epochs_per_round, cfg.n_clients, w)

    ctx = mp.get_context("spawn")

    # spawn long-lived clients with dedicated queues
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

    # main federated rounds
    for rnd in range(int(cfg.num_communication_rounds)):
        server.round_idx = rnd

        # broadcast current global critic to all clients and request local training
        critic_flat = server.get_global_critic_state()
        for rank in range(n_clients):
            cmd_queues[rank].put({"cmd": "train_round", "critic": critic_flat, "epochs": int(epochs_per_round)})

        # gather client updates
        payloads: List[Tuple[Dict[str, torch.Tensor], float, int]] = []
        for rank in range(n_clients):
            msg = rep_queues[rank].get()
            if "error" in msg:
                raise RuntimeError(f"client {rank} failed: {msg['error']}")
            sd = msg["state_dict"]
            ns = int(msg["num_samples"])
            payloads.append((sd, float(w[rank]), ns))

        # aggregate and checkpoint periodically
        server.aggregate(payloads)
        if (rnd + 1) % max(1, int(getattr(cfg, "ckpt_every_rounds", 5))) == 0 or (rnd + 1) == int(cfg.num_communication_rounds):
            server.save_checkpoint(save_dir, tag=f"round{rnd+1}")

    # tell clients to shutdown and join
    for cq in cmd_queues:
        cq.put({"cmd": "shutdown"})
    for p in procs:
        p.join()

    actor_path, critic_path = server.save_checkpoint(save_dir, tag="final")
    server.finish()
    return actor_path, critic_path
