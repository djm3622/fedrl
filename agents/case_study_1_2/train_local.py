# agents/case_study_1_2/train_local.py

from agents.case_study_1_2.train_helper import (
    RolloutBuffer, compute_gae, run_eval_rollout,
    to_tchw, ego_list_to_tchw, quantile_huber_loss,
    log_distributional_visuals_wandb
)
import torch
import wandb
import numpy as np
from torch import nn, optim
import copy
import torch.nn.functional as F

def train(cfg, env, model, device):
    model.actor.to(device)
    model.critic.to(device)
    model.actor.set_exploration_eps(0.05)

    hidden_dim = model.actor.hidden_dim
    h_actor = model.actor.init_hidden(env.cfg.n_agents, device)

    # allow separate critic lr and weight decay without breaking old configs
    lr_actor = getattr(cfg, "lr_actor", cfg.lr)
    lr_critic = getattr(cfg, "lr_critic", cfg.lr)
    wd_critic = getattr(cfg, "critic_weight_decay", 0.0)

    opt_actor = optim.Adam(model.actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(model.critic.parameters(), lr=lr_critic, weight_decay=wd_critic)

    (actor_obs, critic_obs), info = env.reset(seed=cfg.seed)
    n_agents = env.cfg.n_agents
    ego0 = ego_list_to_tchw(actor_obs)
    glob0 = to_tchw(critic_obs)

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

    # optional EMA target critic for distributional stability
    critic_target = None
    ema_tau = float(getattr(cfg, "critic_target_ema", 0.01))
    if is_dist:
        critic_target = copy.deepcopy(model.critic).to(device)
        for p in critic_target.parameters():
            p.requires_grad_(False)

    while total_env_steps < cfg.total_steps:
        roll.clear()

        for _ in range(cfg.rollout_len):
            ego_bna = ego_list_to_tchw(actor_obs).to(device)
            glob_b = to_tchw(critic_obs).unsqueeze(0).to(device)
            agent_ids = torch.arange(n_agents, dtype=torch.long, device=device)
            h_in = h_actor.detach()

            with torch.no_grad():
                dists, h_out = model.actor(ego_bna, agent_ids, h_in=h_in)
                actions = dists.sample()
                logprobs = dists.log_prob(actions)

            with torch.no_grad():
                v = model.critic.mean_value(glob_b).squeeze(0)

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
            last_v = model.critic.mean_value(glob_last).squeeze(0)

        adv, ret = compute_gae(
            roll, last_value=float(last_v.item()),
            gamma=cfg.gamma, lam=cfg.gae_lambda, device=cfg.device
        )
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        total_v_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        n_mb = 0

        B = roll.ptr
        h_flat = roll.h_actor[:B].reshape(B * n_agents, hidden_dim)
        logp_flat = roll.logprobs[:B].reshape(B * n_agents)
        act_flat = roll.actions[:B].reshape(B * n_agents)
        agent_ids_flat = roll.agent_ids[:B].reshape(B * n_agents)
        ego_flat = roll.ego[:B].reshape(B * n_agents, *roll.ego.shape[2:])
        old_logp_flat = logp_flat.detach()
        values_old = roll.values[:B].detach()
        adv_batch = adv.detach()
        ret_batch = ret.detach()
        glob_batch = roll.glob[:B]

        glob_next_all = torch.empty_like(glob_batch)
        glob_next_all[:-1] = glob_batch[1:]                      # s_{t+1} for t < B-1
        glob_next_all[-1]  = glob_last.squeeze(0)      # s_B from the env after rollout

        idxs = np.arange(B)
        mb_size = B // cfg.minibatches if cfg.minibatches > 0 else B
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, mb_size):
                end = start + mb_size
                mb = idxs[start:end]
                if len(mb) == 0:
                    continue

                if is_dist:
                    z_pred = model.critic(glob_batch[mb].to(device))  # [mb, N]

                    glob_next_mb = glob_next_all[mb].to(device)

                    with torch.no_grad():
                        z_next = (critic_target or model.critic)(glob_next_mb)
                        rew_mb = roll.rewards[:B][mb].to(device)
                        done_mb = roll.dones[:B][mb].to(device)
                        z_tgt = rew_mb.unsqueeze(-1) + cfg.gamma * (1.0 - done_mb.unsqueeze(-1)) * z_next
                        z_tgt = torch.clamp(z_tgt, model.critic.v_min, model.critic.v_max)


                    kappa = float(getattr(cfg, "qr_kappa", 1.0))
                    v_loss = quantile_huber_loss(z_pred, z_tgt, model.critic.taus, kappa=kappa)
                    anchor_coef = float(getattr(cfg, "mean_anchor_coef", 0.1))  # 0.05–0.2 works well
                    m_pred = z_pred.mean(dim=-1)                                # [mb]
                    v_loss = v_loss + anchor_coef * F.mse_loss(m_pred, ret_batch[mb].to(device))


                else:
                    v_pred = model.critic(glob_batch[mb].to(device))     # [mb]
                    v_loss_unclipped = (v_pred - ret_batch[mb].to(device)) ** 2
                    v_clipped = values_old[mb].to(device) + torch.clamp(
                        v_pred - values_old[mb].to(device), -cfg.clip_eps, cfg.clip_eps
                    )
                    v_loss_clipped = (v_clipped - ret_batch[mb].to(device)) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                if not torch.isfinite(v_loss):
                    wandb.log({"warn/v_loss_not_finite": 1.0}, step=total_env_steps)
                    continue

                opt_critic.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
                opt_critic.step()

                if is_dist and critic_target is not None:
                    with torch.no_grad():
                        for tp, sp in zip(critic_target.parameters(), model.critic.parameters()):
                            tp.data.lerp_(sp.data, ema_tau)

                # actor update
                mask = []
                for t in mb:
                    base = t * n_agents
                    mask.extend(range(base, base + n_agents))
                mask = torch.as_tensor(mask, dtype=torch.long, device=device)

                dists_new, _ = model.actor(
                    ego_flat[mask].to(device),
                    agent_ids_flat[mask].to(device),
                    h_in=h_flat[mask].to(device)
                )
                new_logp = dists_new.log_prob(act_flat[mask].to(device))
                ratio = torch.exp(new_logp - old_logp_flat[mask])
                adv_per_agent = adv_batch[mb].to(device).repeat_interleave(n_agents)
                pg1 = ratio * adv_per_agent
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_per_agent
                policy_loss = -torch.min(pg1, pg2).mean()
                entropy = dists_new.entropy().mean()
                loss = policy_loss - cfg.ent_coef * entropy

                total_v_loss += v_loss.detach().item()
                total_policy_loss += policy_loss.detach().item()
                total_entropy += entropy.detach().item()
                n_mb += 1

                opt_actor.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), cfg.max_grad_norm)
                opt_actor.step()

        mb_div = max(n_mb, 1)
        log_metrics = {
            "loss/value": total_v_loss / mb_div,
            "loss/policy": total_policy_loss / mb_div,
            "policy/entropy": total_entropy / mb_div,
            "optim/lr_actor": opt_actor.param_groups[0]["lr"],
            "optim/lr_critic": opt_critic.param_groups[0]["lr"],
            "rollout/len": B,
            "critic/is_distributional": float(is_dist),
        }

        if is_dist:
            with torch.no_grad():
                z_dbg = model.critic(glob_batch[: min(64, B)].to(device))
                v_mean_dbg = z_dbg.mean(dim=-1).mean().item()
                z_abs_mean = z_dbg.abs().mean().item()
            log_metrics.update({
                "critic/v_mean_mb": v_mean_dbg,
                "critic/z_abs_mean_mb": z_abs_mean,
            })

        wandb.log(log_metrics, step=total_env_steps)

        if total_env_steps % cfg.log_interval == 0:
            mean_ret = np.mean(ep_returns[-10:]) if ep_returns else 0.0
            wandb.log({
                "train/mean_ep_ret_10": mean_ret,
                "train/last_v": float(last_v.item()),
                "env/episodes": len(ep_returns),
                "env/steps": total_env_steps,
            }, step=total_env_steps)

            if is_dist:
                glob_dbg = glob_batch[:min(128, glob_batch.size(0))]
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
