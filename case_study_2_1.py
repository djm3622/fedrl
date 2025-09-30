import argparse
from dataclasses import asdict
from utils.general import load_config
from envs.case_study_2_1.magridworld import MultiAgentGridWorld
from models.mappo_nets import MAPPOModel
from agents.case_study_2_1 import train_single
import os
import torch
from utils import wandb_helper

import matplotlib
matplotlib.use("Agg") 


methods = ['local', 'fedavg', 'fedtrunc', 'fedrl']


def main(config_path: str):
    cfg = load_config(config_path, config_type='case_2')
    cfg.validate()
    save_path = os.path.join("outputs", cfg.wandb_project, cfg.wandb_run_name)
    os.makedirs(save_path, exist_ok=True)

    wandb_helper.init_wandb(cfg.wandb_project, cfg.wandb_run_name, asdict(cfg))

    env = MultiAgentGridWorld(cfg)
    model = MAPPOModel.build(
        n_actions=5, ego_k=cfg.ego_k, n_agents=cfg.n_agents, 
        critic_type=cfg.param_type, n_quantiles=cfg.n_quantiles
    )

    actor_arch_path = os.path.join(save_path, "actor_arch.txt")
    critic_arch_path = os.path.join(save_path, "critic_arch.txt")
    
    with open(actor_arch_path, "w") as f:
        f.write(str(model.actor))
    with open(critic_arch_path, "w") as f:
        f.write(str(model.critic))

    wandb_helper.log_artifact(actor_arch_path, name="actor_architecture", type="architecture")
    wandb_helper.log_artifact(critic_arch_path, name="critic_architecture", type="architecture")

    model = train_single.train(cfg, env, model, device=cfg.device)

    actor_weights_path = os.path.join(save_path, "actor.pth")
    critic_weights_path = os.path.join(save_path, "critic.pth")

    torch.save(model.actor.state_dict(), actor_weights_path)
    torch.save(model.critic.state_dict(), critic_weights_path)

    wandb_helper.log_artifact(actor_weights_path, name="actor_weights", type="weights")
    wandb_helper.log_artifact(critic_weights_path, name="critic_weights", type="weights")

    wandb_helper.finish_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for training environment.")
    parser.add_argument("config", type=str, help="Path to the config file.")
    
    args = parser.parse_args()
    main(args.config)