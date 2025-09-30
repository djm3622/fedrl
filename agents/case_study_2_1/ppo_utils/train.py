
# this files should support training a single PPO agent
# changes should be made to accomodate the federated server setup later
# it should also support the fedrl training additon to pull on the reward distribution
# the trainn should be broken into a step and each multi-run file should handle their own training loops

import numpy as np
import torch
import wandb
from torch import nn

from agents.case_study_2_1.ppo_utils.helpers import (
    compute_gae, to_tchw, ego_list_to_tchw, quantile_huber_loss,
    setup_optimizers, centralized_value_mean, ppo_epoch_update
)
from agents.case_study_2_1.ppo_utils.buffer import RolloutBuffer
from eval.distributions import log_distributional_visuals_wandb
from eval.metric_losses import run_eval_rollout


def train(cfg, env, model, device):
    model.actor.to(device)
    model.critic.to(device)
    model.actor.set_exploration_eps(cfg.exploration_eps)

    hidden_dim = model.actor.hidden_dim
    h_actor = model.actor.init_hidden(env.cfg.n_agents, device)

    opt_actor, opt_critic = setup_optimizers(model, cfg)

    # reset env and infer shapes
    (actor_obs, critic_obs), info = env.reset(seed=cfg.seed)
    n_agents = env.cfg.n_agents
    ego0 = ego_list_to_tchw(actor_obs)  # [A, C, k, k]
    glob0 = to_tchw(critic_obs)         # [C, H, W]

    roll = RolloutBuffer(
        n_agents=n_agents,
        ego_shape=(ego0.shape[1], ego0.shape[2], ego0.shape[3]),
        glob_shape=(glob0.shape[0], glob0.shape[1], glob0.shape[2]),
        rollout_len=cfg.rollout_len,
        device=cfg.device,
        hidden_dim=hidden_dim,
    )

    total_env_steps = 0
    ep_returns = []
    ep_len = 0
    ep_return = 0.0
    next_eval = cfg.eval_every_steps

    is_dist = hasattr(model.critic, "taus")

    while total_env_steps < cfg.total_steps:
        roll.clear()

        for _ in range(cfg.rollout_len):
            ego_bna = ego_list_to_tchw(actor_obs).to(device)        # [A, C, k, k]
            glob_b = to_tchw(critic_obs).unsqueeze(0).to(device)    # [1, C, H, W]
            agent_ids = torch.arange(n_agents, dtype=torch.long, device=device)

            h_in = h_actor.detach()

            with torch.no_grad():
                dists, h_out = model.actor(ego_bna, agent_ids, h_in=h_in)
                actions = dists.sample()
                logprobs = dists.log_prob(actions)

            with torch.no_grad():
                v = centralized_value_mean(model, glob_b)

            act_dict = {i: int(actions[i].item()) for i in range(n_agents)}
            (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = env.step(act_dict)
            h_actor = h_out.detach()

            roll.add(
                ego_bna=ego_bna,
                agent_ids=agent_ids,
                glob_b=glob_b.squeeze(0),
                act_bna=actions,
                logp_bna=logprobs,
                v_b=v.detach(),
                rew_b=torch.tensor(team_rew, dtype=torch.float32, device=device),
                done_b=torch.tensor(float(terminated or truncated), device=device),
                h_actor_in=h_in,
            )

            total_env_steps += 1
            ep_len += 1
            ep_return += float(team_rew)

            actor_obs, critic_obs = actor_obs_next, critic_obs_next

            if terminated or truncated:
                term_reason = info.get("terminated_by", "unknown")
                wandb.log({
                    "ep/len": ep_len,
                    "ep/return_team": ep_return,
                    "ep/objects_delivered_total": int(info.get("objects_delivered_total", 0)),
                    "ep/term/time_limit": 1.0 if term_reason == "time_limit" else 0.0,
                    "ep/term/catastrophe": 1.0 if term_reason == "catastrophe" else 0.0,
                    "ep/term/full_success": 1.0 if term_reason == "all_goals_and_objects" else 0.0,
                }, step=total_env_steps)

                ep_returns.append(ep_return)
                h_actor = model.actor.init_hidden(n_agents, device)
                (actor_obs, critic_obs), _ = env.reset()
                ep_return = 0.0
                ep_len    = 0

            if roll.full():
                break

        with torch.no_grad():
            glob_last = to_tchw(critic_obs).unsqueeze(0).to(device)
            last_v = centralized_value_mean(model, glob_last)

        adv, ret = compute_gae(
            roll, last_value=float(last_v.item()),
            gamma=cfg.gamma, lam=cfg.gae_lambda, device=cfg.device
        )
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # flatten per time for actor
        T = roll.ptr
        h_flat = roll.h_actor[:T].reshape(T * n_agents, hidden_dim)
        logp_flat = roll.logprobs[:T].reshape(T * n_agents)
        act_flat = roll.actions[:T].reshape(T * n_agents)
        agent_ids_flat = roll.agent_ids[:T].reshape(T * n_agents)
        ego_flat = roll.ego[:T].reshape(T * n_agents, *roll.ego.shape[2:])
        old_logp_flat = logp_flat.detach()
        glob_batch = roll.glob[:T]
        values_old = roll.values[:T].detach()

        metrics = ppo_epoch_update(
            cfg=cfg,
            model=model,
            opt_actor=opt_actor,
            opt_critic=opt_critic,
            roll=roll,
            adv=adv,
            ret=ret,
            ego_flat=ego_flat,
            agent_ids_flat=agent_ids_flat,
            h_flat=h_flat,
            act_flat=act_flat,
            old_logp_flat=old_logp_flat,
            glob_batch=glob_batch,
            values_old=values_old,
            device=device,
            quantile_huber_loss=quantile_huber_loss,
        )

        # extra diagnostics for distributional critic
        if is_dist:
            with torch.no_grad():
                z_dbg = model.critic(glob_batch[: min(64, T)].to(device))
                v_mean_dbg = z_dbg.mean(dim=-1).mean().item()
            metrics["critic/v_mean_mb"] = v_mean_dbg

        # lr logging
        metrics.update({
            "optim/lr_actor": opt_actor.param_groups[0]["lr"],
            "optim/lr_critic": opt_critic.param_groups[0]["lr"],
            "env/steps": total_env_steps,
        })
        wandb.log(metrics, step=total_env_steps)

        if total_env_steps % cfg.log_interval == 0:
            mean_ret = np.mean(ep_returns[-10:]) if ep_returns else 0.0
            wandb.log({
                "train/mean_ep_ret_10": mean_ret,
                "train/last_v": float(last_v.item()),
                "env/episodes": len(ep_returns),
            }, step=total_env_steps)
            if is_dist:
                glob_dbg = glob_batch[: min(128, glob_batch.size(0))]
                log_distributional_visuals_wandb(
                    glob_batch=glob_dbg,
                    model=model,
                    device=device,
                    wb_step=total_env_steps,
                    split="train",
                    max_batch=128,
                )

        if total_env_steps >= next_eval:
            eval_gif = f"outputs/eval/seed_{cfg.seed}_step_{total_env_steps}.gif"
            _ = run_eval_rollout(
                env, model, device,
                deterministic=False,
                record=True,
                log_wandb=True,
                gif_path=eval_gif,
                wb_step=total_env_steps,
            )
            next_eval += cfg.eval_every_steps

    return model
