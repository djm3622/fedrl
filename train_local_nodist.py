# agents/case_study_1_2/train_local.py
# Minimal MAPPO trainer for MultiAgentGridWorld. Single env, on-policy PPO+GAE.
# Keeps things simple so you can verify learning and plotting quickly.
# Later, add vectorized envs and a distributional critic swap.

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Tuple, List
import imageio.v2 as imageio  # for GIF writing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, disables GUI output


from envs.case_study_1_2.magridworld import MultiAgentGridWorld, MultiAgentGridConfig, make_aligned_client_cfg
from models.mappo_nets import MAPPOModel
from utils.general import pick_device

def capture_frame(env):
    try:
        frame = env.render(mode="rgb_array")
        if isinstance(frame, np.ndarray):
            return frame
    except Exception:
        pass
    try:
        frame = env.render(return_rgb_array=True)
        if isinstance(frame, np.ndarray):
            return frame
    except Exception:
        pass
    try:
        frame = env.render()
        if isinstance(frame, np.ndarray):
            return frame
    except Exception:
        pass
    return None

def save_gif(frames, path, fps=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0/max(fps,1))



# -------------------------
# Config
# -------------------------
@dataclass
class TrainCfg:
    device = pick_device()
    total_steps: int = 100_000
    rollout_len: int = 512
    update_epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 42
    log_interval: int = 1


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tchw(x_hw_c: np.ndarray) -> torch.Tensor:
    # input HxWxC float32 -> torch [C,H,W]
    assert x_hw_c.ndim == 3
    x = torch.from_numpy(x_hw_c).permute(2, 0, 1).contiguous().float()
    return x


def ego_list_to_tchw(ego_list: List[np.ndarray]) -> torch.Tensor:
    # list of kxkxC -> [B, C, k, k]
    arr = np.stack(ego_list, axis=0)  # [B, k, k, C]
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float()
    return x


# -------------------------
# Rollout storage (team advantage applied to all agents)
# -------------------------
class RolloutBuffer:
    def __init__(self, n_agents: int, ego_shape: Tuple[int, int, int], glob_shape: Tuple[int, int, int],
                 rollout_len: int, device: str):
        self.n_agents = n_agents
        self.rollout_len = rollout_len
        self.device = device

        B = rollout_len
        C_ego, H_ego, W_ego = ego_shape
        C_glb, H_glb, W_glb = glob_shape

        self.ego = torch.zeros(B, n_agents, C_ego, H_ego, W_ego, device=device)
        self.agent_ids = torch.zeros(B, n_agents, dtype=torch.long, device=device)
        self.glob = torch.zeros(B, C_glb, H_glb, W_glb, device=device)
        self.actions = torch.zeros(B, n_agents, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(B, n_agents, device=device)
        self.values = torch.zeros(B, device=device)
        self.rewards = torch.zeros(B, device=device)       # team reward
        self.dones = torch.zeros(B, device=device)

        self.ptr = 0

    def add(self, ego_bna, agent_ids, glob_b, act_bna, logp_bna, v_b, rew_b, done_b):
        t = self.ptr
        self.ego[t] = ego_bna
        self.agent_ids[t] = agent_ids
        self.glob[t] = glob_b
        self.actions[t] = act_bna
        self.logprobs[t] = logp_bna
        self.values[t] = v_b
        self.rewards[t] = rew_b
        self.dones[t] = done_b
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.rollout_len

    def clear(self):
        self.ptr = 0


def compute_gae(roll: RolloutBuffer, last_value: float, gamma: float, lam: float, device: str):
    T = roll.ptr
    adv = torch.zeros(T, device=device)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - roll.dones[t]
        next_value = last_value if t == T - 1 else roll.values[t + 1]
        delta = roll.rewards[t] + gamma * next_value * nonterminal - roll.values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + roll.values[:T]
    return adv, ret


# -------------------------
# Training
# -------------------------
def make_env_and_model(seed: int) -> Tuple[MultiAgentGridWorld, MAPPOModel]:
    cfg = make_aligned_client_cfg(n_agents=3, H=10, W=10, seed=seed)
    env = MultiAgentGridWorld(cfg)
    model = MAPPOModel.build(n_actions=4, ego_k=cfg.ego_k, n_agents=cfg.n_agents)
    return env, model


def train():
    cfg = TrainCfg()
    set_seed(cfg.seed)

    env, model = make_env_and_model(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    model.actor.to(device)
    model.critic.to(device)

    opt_actor = optim.Adam(model.actor.parameters(), lr=cfg.lr)
    opt_critic = optim.Adam(model.critic.parameters(), lr=cfg.lr)

    # reset env and infer shapes
    (actor_obs, critic_obs), info = env.reset(seed=cfg.seed)
    n_agents = env.cfg.n_agents
    ego0 = ego_list_to_tchw(actor_obs)  # [n_agents, C, k, k]
    glob0 = to_tchw(critic_obs)         # [C, H, W]

    roll = RolloutBuffer(
        n_agents=n_agents,
        ego_shape=(ego0.shape[1], ego0.shape[2], ego0.shape[3]),
        glob_shape=(glob0.shape[0], glob0.shape[1], glob0.shape[2]),
        rollout_len=cfg.rollout_len,
        device=cfg.device,
    )

    total_env_steps = 0
    ep_returns = []
    ep_len = 0
    ep_return = 0.0

    print("Starting training...")

    gif_outdir = os.path.join("outputs", "rollouts")
    next_log = cfg.log_interval  # optional threshold if you prefer threshold-based logging

    while total_env_steps < cfg.total_steps:
        # ----- collect rollout -----
        roll.clear()
        frames: List[np.ndarray] = []  # collect frames for this rollout

        for _ in range(cfg.rollout_len):
            ego_bna = ego_list_to_tchw(actor_obs).to(device)        # [A, C, k, k]
            glob_b = to_tchw(critic_obs).unsqueeze(0).to(device)    # [1, C, H, W]
            agent_ids = torch.arange(n_agents, dtype=torch.long, device=device)

            # policy for each agent
            dists = model.actor(ego_bna, agent_ids)
            actions = dists.sample()
            logprobs = dists.log_prob(actions)

            # centralized value once per state
            with torch.no_grad():
                v = model.critic(glob_b).squeeze(0)

            # env step
            act_dict = {i: int(actions[i].item()) for i in range(n_agents)}
            (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = env.step(act_dict)

            # capture frame AFTER the step so you see the result of the actions
            frame = capture_frame(env)
            frames.append(frame)

            # store
            roll.add(
                ego_bna=ego_bna,
                agent_ids=agent_ids,
                glob_b=glob_b.squeeze(0),
                act_bna=actions,
                logp_bna=logprobs,
                v_b=v.detach(),
                rew_b=torch.tensor(team_rew, dtype=torch.float32, device=device),
                done_b=torch.tensor(float(terminated or truncated), device=device),
            )

            # episode bookkeeping
            total_env_steps += 1
            ep_len += 1
            ep_return += float(team_rew)

            # advance obs
            actor_obs, critic_obs = actor_obs_next, critic_obs_next

            if terminated or truncated:
                (actor_obs, critic_obs), _ = env.reset()
                ep_returns.append(ep_return)
                ep_return = 0.0
                ep_len = 0

                # also capture a frame at reset if available (optional)
                frame = capture_frame(env)
                if frame is not None:
                    frames.append(frame)

            if roll.full():
                break

        # bootstrap value for last state
        with torch.no_grad():
            glob_last = to_tchw(critic_obs).unsqueeze(0).to(device)
            last_v = model.critic(glob_last).squeeze(0)

        # ----- compute GAE -----
        adv, ret = compute_gae(roll, last_value=float(last_v.item()), gamma=cfg.gamma, lam=cfg.gae_lambda, device=cfg.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # ----- flatten per-agent items for actor updates -----
        B = roll.ptr
        logp_flat = roll.logprobs[:B].reshape(B * n_agents)
        act_flat = roll.actions[:B].reshape(B * n_agents)
        agent_ids_flat = roll.agent_ids[:B].reshape(B * n_agents)
        ego_flat = roll.ego[:B].reshape(B * n_agents, *roll.ego.shape[2:])
        old_logp_flat = logp_flat.detach()
        glob_batch = roll.glob[:B]
        values_old = roll.values[:B].detach()
        adv_batch = adv.detach()
        ret_batch = ret.detach()

        # minibatch indexing
        idxs = np.arange(B)
        mb_size = B // cfg.minibatches if cfg.minibatches > 0 else B
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, mb_size):
                end = start + mb_size
                mb = idxs[start:end]
                if len(mb) == 0:
                    continue

                # critic update
                v_pred = model.critic(glob_batch[mb].to(device))
                v_loss_unclipped = (v_pred - ret_batch[mb].to(device)) ** 2
                v_clipped = values_old[mb].to(device) + torch.clamp(v_pred - values_old[mb].to(device), -cfg.clip_eps, cfg.clip_eps)
                v_loss_clipped = (v_clipped - ret_batch[mb].to(device)) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                opt_critic.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
                opt_critic.step()

                # actor update
                mask = []
                for t in mb:
                    base = t * n_agents
                    mask.extend(range(base, base + n_agents))
                mask = torch.as_tensor(mask, dtype=torch.long, device=device)

                dists_new = model.actor(ego_flat[mask].to(device), agent_ids_flat[mask].to(device))
                new_logp = dists_new.log_prob(act_flat[mask].to(device))
                ratio = torch.exp(new_logp - old_logp_flat[mask])
                adv_per_agent = adv_batch[mb].to(device).repeat_interleave(n_agents)
                pg1 = ratio * adv_per_agent
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_per_agent
                policy_loss = -torch.min(pg1, pg2).mean()
                entropy = dists_new.entropy().mean()
                loss = policy_loss - cfg.ent_coef * entropy

                opt_actor.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), cfg.max_grad_norm)
                opt_actor.step()

        # logging + video dump
        if total_env_steps % cfg.log_interval == 0:
            mean_ret = np.mean(ep_returns[-10:]) if ep_returns else 0.0
            print(f"steps={total_env_steps}  mean_ep_ret_10={mean_ret:.2f}  last_v={last_v.item():.2f}", flush=True)

            # save gif for the last rollout
            gif_path = os.path.join(gif_outdir, f"rollout_steps_{total_env_steps}.gif")
            save_gif(frames, gif_path, fps=10)
            print(f"[saved rollout video -> {gif_path}]", flush=True)



    print("Training complete.")


if __name__ == "__main__":
    train()
