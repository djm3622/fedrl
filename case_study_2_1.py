#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
case_study_2_1.py — Entry point for Case Study 2.1

Methods supported:
  - local   : single-client local training (uses your existing train_single)
  - fedavg  : multi-client FedAvg with parallel clients on one GPU

Config assumptions (case_2):
  - cfg.n_clients
  - cfg.local_epochs_per_round
  - cfg.num_communication_rounds
  - cfg.client_weights (list or scalar)
  - cfg.total_steps, rollout_len, etc. for PPOTrainer
  - cfg.wandb_project, cfg.wandb_run_name, device, seed, etc.
"""

from __future__ import annotations
import argparse
import os
import torch
from dataclasses import asdict

from utils.general import load_config
from utils import wandb_helper

from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel
from agents.case_study_2_1 import train_single

# fedavg runner
from agents.case_study_2_1.train_fedavg import run_fedavg
import wandb


def _model_builder_from_cfg(cfg):
    return MAPPOModel.build(
        n_actions=5,
        ego_k=cfg.ego_k,
        n_agents=cfg.n_agents,
        critic_type=cfg.param_type,
        n_quantiles=cfg.n_quantiles,
    )


def _ensure_archives(cfg, model):
    save_path = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_path, exist_ok=True)
    actor_arch_path = os.path.join(save_path, "actor_arch.txt")
    critic_arch_path = os.path.join(save_path, "critic_arch.txt")

    # always write the architecture snapshots
    with open(actor_arch_path, "w") as f:
        f.write(str(model.actor))
    with open(critic_arch_path, "w") as f:
        f.write(str(model.critic))

    # log to W&B ONLY if a run is already active
    # (fedavg path initializes W&B inside FedAvgServer later)
    if wandb.run is not None:
        from utils import wandb_helper as wb
        wb.log_artifact(actor_arch_path, name="actor_architecture", type="architecture")
        wb.log_artifact(critic_arch_path, name="critic_architecture", type="architecture")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Case Study 2.1 entry point")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--method", type=str, default="fedavg", choices=["local", "fedavg"], help="Training method.")
    args = parser.parse_args()

    cfg = load_config(args.config, config_type="case_2")
    cfg.validate()

    save_path = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_path, exist_ok=True)

    if args.method == "local":
        # single-client baseline (your existing loop)
        wandb_helper.init_wandb(cfg.wandb_project, cfg.wandb_run_name, asdict(cfg))
        env = MultiAgentGridWorld(cfg)
        model = _model_builder_from_cfg(cfg)
        _ensure_archives(cfg, model)
        model = train_single.train(cfg, env, model, device=cfg.device)

        actor_weights_path = os.path.join(save_path, "actor.pth")
        critic_weights_path = os.path.join(save_path, "critic.pth")
        torch.save(model.actor.state_dict(), actor_weights_path)
        torch.save(model.critic.state_dict(), critic_weights_path)
        wandb_helper.log_artifact(actor_weights_path, name="actor_weights", type="weights")
        wandb_helper.log_artifact(critic_weights_path, name="critic_weights", type="weights")
        wandb_helper.finish_run()
    else:
        # federated averaging with parallel clients
        # first archive the architectures for reference
        tmp_model = _model_builder_from_cfg(cfg)
        _ensure_archives(cfg, tmp_model)
        # run fedavg
        actor_path, critic_path = run_fedavg(cfg)
        print(f"[fedavg] final checkpoints: {actor_path}, {critic_path}")


if __name__ == "__main__":
    main()
