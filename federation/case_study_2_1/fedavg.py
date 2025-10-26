#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fedavg.py — FedAvg server aggregating ONLY the critic for MAPPO.

We exchange and average a flat dict with keys "critic.<k>" only.
Actors/policies remain local to each client and are never aggregated.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import os
import torch
import wandb

# ---------- helpers to pack/unpack ONLY the critic ----------

def pack_critic_state(model) -> Dict[str, torch.Tensor]:
    """
    Returns a cpu state dict: { "critic.<k>": tensor }
    """
    critic_sd = model.critic.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k, v in critic_sd.items():
        out[f"critic.{k}"] = v.detach().cpu().clone()
    return out


def unpack_critic_state_into(model, flat_state: Dict[str, torch.Tensor]) -> None:
    """
    Loads the critic substate from a flat dict { "critic.<k>": tensor }.
    """
    critic_sd: Dict[str, torch.Tensor] = {}
    for k, v in flat_state.items():
        if not k.startswith("critic."):
            raise KeyError(f"unexpected non-critic key in flat_state: {k}")
        critic_sd[k[len("critic."):]] = v
    model.critic.load_state_dict(critic_sd, strict=True)


def _weighted_average_state(states: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    """
    states: list of (flat critic state, weight)
    returns: weighted average flat critic state on CPU
    """
    assert len(states) > 0
    keys = list(states[0][0].keys())
    acc = {k: torch.zeros_like(states[0][0][k], device="cpu") for k in keys}
    total_w = 0.0
    for sd, w in states:
        w = float(max(0.0, w))
        if w == 0.0:
            continue
        for k in acc.keys():
            acc[k] += sd[k].detach().cpu() * w
        total_w += w
    if total_w == 0.0:
        total_w = float(len(states))
    for k in acc.keys():
        acc[k] /= total_w
    return acc


# ---------- minimal, safe W&B init for server ----------

def _server_wandb_safe_init(project: str, name: str, cfg_dict: dict, group: str | None):
    if not os.getenv("WANDB_API_KEY", ""):
        os.environ.setdefault("WANDB_MODE", "offline")
    # Use stable id + resume to avoid spawning extra runs
    return wandb.init(
        project=project,
        name=name + "_fedavg_server",
        id=name + "_fedavg_server",
        resume="allow",
        config=cfg_dict,
        group=group,
    )


class FedAvgServer:
    def __init__(self, cfg: Any, model_builder, device: str = "cuda:0"):
        """
        model_builder: () -> MAPPOModel (with .actor and .critic)
        """
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model_builder()
        self.model.actor.to(self.device)
        self.model.critic.to(self.device)
        self.round_idx = 0

        self.run = _server_wandb_safe_init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            cfg_dict=asdict(cfg),
            group=getattr(cfg, "wandb_group", None),
        )

        # server absolute step base (for monotonic logging across resumes)
        self.server_step_base = 0
        if self.run is not None and getattr(self.run, "resumed", False):
            self.server_step_base = int(self.run.summary.get("server/last_step", 0))

    def _wb_step(self) -> int:
        return int(self.server_step_base + self.round_idx)

    # -------- global critic weights io --------
    def get_global_critic_state(self) -> Dict[str, torch.Tensor]:
        return pack_critic_state(self.model)

    def load_global_critic_state(self, flat_state: Dict[str, torch.Tensor]) -> None:
        unpack_critic_state_into(self.model, flat_state)
        self.model.critic.to(self.device)

    # -------- aggregation (critic only) --------
    def aggregate(self, client_payloads: List[Tuple[Dict[str, torch.Tensor], float, int]]) -> None:
        """
        client_payloads: list of (critic_flat_state, weight, n_samples)
        """
        avg_state = _weighted_average_state([(sd, float(w)) for (sd, w, _) in client_payloads])
        self.load_global_critic_state(avg_state)

        total_samples = int(sum(max(0, int(ns)) for (_, _, ns) in client_payloads))
        if self.run is not None:
            wandb.log({
                "server/round": int(self.round_idx),
                "server/total_samples_round": total_samples,
                "server/num_clients_round": len(client_payloads),
            }, step=self._wb_step())
            # persist last step to support resume
            self.run.summary["server/last_step"] = self._wb_step()

    # -------- checkpointing (save both modules for convenience) --------
    def save_checkpoint(self, save_dir: str, tag: str):
        os.makedirs(save_dir, exist_ok=True)
        actor_path = os.path.join(save_dir, f"actor_{tag}.pth")
        critic_path = os.path.join(save_dir, f"critic_{tag}.pth")
        torch.save(self.model.actor.state_dict(), actor_path)
        torch.save(self.model.critic.state_dict(), critic_path)

        if self.run is not None:
            art_a = wandb.Artifact(name=f"actor_{tag}", type="weights")
            art_a.add_file(actor_path)
            self.run.log_artifact(art_a)
            art_c = wandb.Artifact(name=f"critic_{tag}", type="weights")
            art_c.add_file(critic_path)
            self.run.log_artifact(art_c)
        return actor_path, critic_path

    def finish(self):
        if self.run is not None:
            self.run.finish()
