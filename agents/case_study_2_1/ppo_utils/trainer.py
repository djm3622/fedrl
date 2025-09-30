import numpy as np
import torch
import wandb
from typing import Dict, Any

from .helpers import (
    compute_gae, to_tchw, ego_list_to_tchw, quantile_huber_loss,
    setup_optimizers, centralized_value_mean, ppo_epoch_update
)
from .buffer import RolloutBuffer
from eval.distributions import log_distributional_visuals_wandb
from eval.metric_losses import run_eval_rollout


class PPOTrainer:
    """
    Manages one client (env, model, buffer, optimizers).
    Exposes small steps so a multi-client runner can interleave collection and updates.
    """

    def __init__(self, cfg: Any, env: Any, model: Any, device: str | torch.device, client_id: int = 0):
        # runtime guards only
        assert hasattr(model, "actor") and hasattr(model, "critic"), "model must have actor and critic"
        assert hasattr(model.actor, "init_hidden"), "actor must define init_hidden(n_agents, device)"
        assert hasattr(env, "reset") and hasattr(env, "step") and hasattr(env, "cfg"), "env must expose reset/step and cfg"

        self.cfg = cfg
        self.env = env
        self.model = model
        self.device = torch.device(device)
        self.client_id = client_id

        self.model.actor.to(self.device)
        self.model.critic.to(self.device)
        if hasattr(self.model.actor, "set_exploration_eps"):
            self.model.actor.set_exploration_eps(cfg.exploration_eps)

        self.hidden_dim = self.model.actor.hidden_dim
        self.h_actor = self.model.actor.init_hidden(self.env.cfg.n_agents, self.device)

        self.opt_actor, self.opt_critic = setup_optimizers(self.model, cfg)

        # reset and infer shapes
        (actor_obs, critic_obs), _ = self.env.reset(seed=cfg.seed)
        self.n_agents = self.env.cfg.n_agents
        ego0 = ego_list_to_tchw(actor_obs)
        glob0 = to_tchw(critic_obs)

        self.roll = RolloutBuffer(
            n_agents=self.n_agents,
            ego_shape=(ego0.shape[1], ego0.shape[2], ego0.shape[3]),
            glob_shape=(glob0.shape[0], glob0.shape[1], glob0.shape[2]),
            rollout_len=cfg.rollout_len,
            device=str(self.device),
            hidden_dim=self.hidden_dim,
        )

        self.actor_obs = actor_obs
        self.critic_obs = critic_obs

        self.total_env_steps = 0
        self.ep_returns: list[float] = []
        self.ep_len = 0
        self.ep_return = 0.0
        self.next_eval = cfg.eval_every_steps

        self.is_dist = hasattr(self.model.critic, "taus")

    # ---------- collection ----------

    def _forward_policy(self, ego_bna: torch.Tensor, agent_ids: torch.Tensor, h_in: torch.Tensor):
        with torch.no_grad():
            dists, h_out = self.model.actor(ego_bna, agent_ids, h_in=h_in)
            actions = dists.sample()
            logprobs = dists.log_prob(actions)
        return actions, logprobs, h_out

    def _forward_value(self, glob_b: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return centralized_value_mean(self.model, glob_b)

    def collect_rollout_step(self) -> bool:
        """
        Single env step; push into buffer.
        Returns True if buffer is now full.
        """
        ego_bna = ego_list_to_tchw(self.actor_obs).to(self.device)
        glob_b = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
        agent_ids = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

        h_in = self.h_actor.detach()

        actions, logprobs, h_out = self._forward_policy(ego_bna, agent_ids, h_in=h_in)
        v = self._forward_value(glob_b)

        act_dict = {i: int(actions[i].item()) for i in range(self.n_agents)}
        (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = self.env.step(act_dict)
        self.h_actor = h_out.detach()

        self.roll.add(
            ego_bna=ego_bna,
            agent_ids=agent_ids,
            glob_b=glob_b.squeeze(0),
            act_bna=actions,
            logp_bna=logprobs,
            v_b=v.detach(),
            rew_b=torch.tensor(team_rew, dtype=torch.float32, device=self.device),
            done_b=torch.tensor(float(terminated or truncated), device=self.device),
            h_actor_in=h_in,
        )

        self.total_env_steps += 1
        self.ep_len += 1
        self.ep_return += float(team_rew)

        self.actor_obs, self.critic_obs = actor_obs_next, critic_obs_next

        if terminated or truncated:
            term_reason = info.get("terminated_by", "unknown")
            wandb.log({
                f"client{self.client_id}/ep/len": self.ep_len,
                f"client{self.client_id}/ep/return_team": self.ep_return,
                f"client{self.client_id}/ep/objects_delivered_total": int(info.get("objects_delivered_total", 0)),
                f"client{self.client_id}/ep/term/time_limit": 1.0 if term_reason == "time_limit" else 0.0,
                f"client{self.client_id}/ep/term/catastrophe": 1.0 if term_reason == "catastrophe" else 0.0,
                f"client{self.client_id}/ep/term/full_success": 1.0 if term_reason == "all_goals_and_objects" else 0.0,
            }, step=self.total_env_steps)

            self.ep_returns.append(self.ep_return)
            self.h_actor = self.model.actor.init_hidden(self.n_agents, self.device)
            (self.actor_obs, self.critic_obs), _ = self.env.reset()
            self.ep_return = 0.0
            self.ep_len = 0

        return self.roll.full()

    # ---------- value bootstrap, adv, ppo update ----------

    def _bootstrap_last_value(self) -> float:
        glob_last = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
        last_v = self._forward_value(glob_last)
        return float(last_v.item())

    def update_epoch(self) -> Dict[str, float]:
        last_v = self._bootstrap_last_value()
        adv, ret = compute_gae(
            self.roll, last_value=last_v,
            gamma=self.cfg.gamma, lam=self.cfg.gae_lambda, device=str(self.device)
        )
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        T = self.roll.ptr
        h_flat = self.roll.h_actor[:T].reshape(T * self.n_agents, self.hidden_dim)
        logp_flat = self.roll.logprobs[:T].reshape(T * self.n_agents)
        act_flat = self.roll.actions[:T].reshape(T * self.n_agents)
        agent_ids_flat = self.roll.agent_ids[:T].reshape(T * self.n_agents)
        ego_flat = self.roll.ego[:T].reshape(T * self.n_agents, *self.roll.ego.shape[2:])
        old_logp_flat = logp_flat.detach()
        glob_batch = self.roll.glob[:T]
        values_old = self.roll.values[:T].detach()

        metrics = ppo_epoch_update(
            cfg=self.cfg,
            model=self.model,
            opt_actor=self.opt_actor,
            opt_critic=self.opt_critic,
            roll=self.roll,
            adv=adv,
            ret=ret,
            ego_flat=ego_flat,
            agent_ids_flat=agent_ids_flat,
            h_flat=h_flat,
            act_flat=act_flat,
            old_logp_flat=old_logp_flat,
            glob_batch=glob_batch,
            values_old=values_old,
            device=str(self.device),
            quantile_huber_loss=quantile_huber_loss,
        )

        if self.is_dist:
            with torch.no_grad():
                z_dbg = self.model.critic(glob_batch[: min(64, T)].to(self.device))
                v_mean_dbg = z_dbg.mean(dim=-1).mean().item()
            metrics[f"client{self.client_id}/critic/v_mean_mb"] = v_mean_dbg

        metrics.update({
            f"client{self.client_id}/optim/lr_actor": self.opt_actor.param_groups[0]["lr"],
            f"client{self.client_id}/optim/lr_critic": self.opt_critic.param_groups[0]["lr"],
            f"client{self.client_id}/env/steps": self.total_env_steps,
        })

        wandb.log(metrics, step=self.total_env_steps)

        if self.total_env_steps % self.cfg.log_interval == 0:
            mean_ret = float(np.mean(self.ep_returns[-10:])) if self.ep_returns else 0.0
            wandb.log({
                f"client{self.client_id}/train/mean_ep_ret_10": mean_ret,
                f"client{self.client_id}/train/last_v": float(last_v),
                f"client{self.client_id}/env/episodes": len(self.ep_returns),
            }, step=self.total_env_steps)

            if self.is_dist:
                glob_dbg = glob_batch[: min(128, glob_batch.size(0))]
                log_distributional_visuals_wandb(
                    glob_batch=glob_dbg,
                    model=self.model,
                    device=str(self.device),
                    wb_step=self.total_env_steps,
                    split=f"train_client{self.client_id}",
                    max_batch=128,
                )

        self.roll.clear()
        return metrics


    def maybe_eval(self) -> None:
        if self.total_env_steps >= self.next_eval:
            eval_gif = f"outputs/eval/client_{self.client_id}_seed_{self.cfg.seed}_step_{self.total_env_steps}.gif"
            _ = run_eval_rollout(
                self.env, self.model, self.device,
                deterministic=False,
                record=True,
                log_wandb=True,
                gif_path=eval_gif,
                wb_step=self.total_env_steps,
            )
            self.next_eval += self.cfg.eval_every_steps
