#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# runners
from agents.case_study_2_1.train_fedavg import run_fedavg
from agents.case_study_2_1.train_fedtrunc import run_fedtrunc
from agents.case_study_2_1.train_local import run_local_multi
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

    with open(actor_arch_path, "w") as f:
        f.write(str(model.actor))
    with open(critic_arch_path, "w") as f:
        f.write(str(model.critic))

    if wandb.run is not None:
        from utils import wandb_helper as wb
        wb.log_artifact(actor_arch_path, name="actor_architecture", type="architecture")
        wb.log_artifact(critic_arch_path, name="critic_architecture", type="architecture")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Case Study 2.1 entry point")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--method", type=str, default="local",
        choices=["local", "fedavg", "fedtrunc"],
        help="Training method."
    )
    args = parser.parse_args()

    cfg = load_config(args.config, config_type="case_2")
    cfg.validate()

    save_path = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_path, exist_ok=True)

    # local currently trains only one model, update this to train multiple clients
    if args.method == "local":
        tmp_model = _model_builder_from_cfg(cfg)
        _ensure_archives(cfg, tmp_model)
        ckpts = run_local_multi(cfg)
        for i, (a, c) in enumerate(ckpts):
            print(f"[local] client {i} checkpoints: {a}, {c}")

    elif args.method == "fedavg":
        tmp_model = _model_builder_from_cfg(cfg)
        _ensure_archives(cfg, tmp_model)
        actor_path, critic_path = run_fedavg(cfg)
        print(f"[fedavg] final checkpoints: {actor_path}, {critic_path}")

    elif args.method == "fedtrunc":
        tmp_model = _model_builder_from_cfg(cfg)
        _ensure_archives(cfg, tmp_model)
        actor_path, critic_path = run_fedtrunc(cfg)
        print(f"[fedtrunc] final checkpoints: {actor_path}, {critic_path}")


if __name__ == "__main__":
    main()
