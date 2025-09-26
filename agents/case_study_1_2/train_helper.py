import torch
from typing import Tuple, List
import numpy as np
from typing import Optional
from utils.general import capture_frame, save_gif
import wandb
import os

class RolloutBuffer:
    def __init__(
        self, n_agents: int, ego_shape: Tuple[int, int, int], 
        glob_shape: Tuple[int, int, int],
        rollout_len: int, device: str, hidden_dim: int = 128
    ):
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
        self.rewards = torch.zeros(B, device=device)   
        self.dones = torch.zeros(B, device=device)
        self.h_actor = torch.zeros(B, n_agents, hidden_dim, device=device)

        self.ptr = 0

    def add(self, ego_bna, agent_ids, glob_b, act_bna, logp_bna, v_b, rew_b, done_b, h_actor_in):
        t = self.ptr
        self.ego[t] = ego_bna
        self.agent_ids[t] = agent_ids
        self.glob[t] = glob_b
        self.actions[t] = act_bna
        self.logprobs[t] = logp_bna
        self.values[t] = v_b
        self.rewards[t] = rew_b
        self.dones[t] = done_b
        self.h_actor[t] = h_actor_in
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


@torch.no_grad()
def run_eval_rollout(
    env,
    model,
    device: torch.device,
    *,
    deterministic: bool = False,
    max_steps: Optional[int] = None,
    record: bool = True,
    gif_path: Optional[str] = None,
    log_wandb: bool = True,
    wb_step: Optional[int] = None,   # pass total_env_steps when calling if you want nice W&B step alignment
) -> dict:
    """
    Run ONE evaluation episode (no gradient). Compatible with FF or GRU actor.
    - Turns off epsilon-mixing during eval and restores it after.
    - Works for 4 actions (UDLR) or 5 (UDLR + stay).
    - If `record=True`, captures an initial frame so the GIF is never empty.
    - If `log_wandb=True`, logs both metrics and the GIF (if written).
    """
    model.actor.eval()
    model.critic.eval()

    (actor_obs, critic_obs), _ = env.reset()
    n_agents = env.cfg.n_agents

    # GRU support
    has_gru = hasattr(model.actor, "init_hidden") and hasattr(model.actor, "hidden_dim")
    h = model.actor.init_hidden(n_agents, device) if has_gru else None

    # Turn off exploration eps during eval (restore later)
    eps_restore = None
    if hasattr(model.actor, "set_exploration_eps"):
        eps_restore = float(getattr(model.actor, "_eps_explore", 0.0))
        model.actor.set_exploration_eps(0.0)

    # Accumulators
    team_return = 0.0
    per_agent_return = np.zeros(n_agents, dtype=np.float64)
    steps = 0
    first_delivery_t = None
    deliveries = 0
    object_move_steps = 0
    last_obj_positions = None

    probs_running_sum = None
    probs_count = 0

    frames = [] if record else None
    # Capture frame at reset so GIF is never empty
    if record:
        f0 = capture_frame(env)
        if f0 is not None:
            frames.append(f0)

    while True:
        ego_bna = ego_list_to_tchw(actor_obs).to(device)       # [A,C,k,k]
        glob_b = to_tchw(critic_obs).unsqueeze(0).to(device)   # [1,C,H,W]
        agent_ids = torch.arange(n_agents, dtype=torch.long, device=device)

        # Policy forward (FF or GRU)
        if has_gru:
            if "eps_override" in model.actor.forward.__code__.co_varnames:
                dists, h = model.actor(ego_bna, agent_ids, h_in=h, eps_override=0.0)
            else:
                dists, h = model.actor(ego_bna, agent_ids, h_in=h)
        else:
            out = model.actor(ego_bna, agent_ids)
            dists = out[0] if isinstance(out, tuple) else out

        # Choose actions
        if deterministic:
            actions = torch.argmax(dists.probs, dim=-1)  # [A]
        else:
            actions = dists.sample()

        # Track action probs (support 4 or 5 actions)
        probs = dists.probs.detach().cpu().numpy()       # [A, n_actions]
        if probs_running_sum is None:
            probs_running_sum = np.zeros(probs.shape[1], dtype=np.float64)
        probs_running_sum += probs.mean(axis=0)
        probs_count += 1

        # Env step
        act_dict = {i: int(actions[i].item()) for i in range(n_agents)}
        (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = env.step(act_dict)

        # Record frame after step
        if record:
            f = capture_frame(env)
            if f is not None:
                frames.append(f)

        # Returns
        team_return += float(team_rew)
        if "rewards_per_agent" in info and isinstance(info["rewards_per_agent"], (list, tuple)):
            per_agent_return += np.array(info["rewards_per_agent"], dtype=np.float64)

        # Deliveries & object motion
        if "objects_delivered_new" in info:
            deliveries += int(info.get("objects_delivered_new", 0))
            if first_delivery_t is None and info.get("objects_delivered_total", 0) > 0:
                first_delivery_t = steps + 1

        if "object_pos_list" in info:
            cur_obj_positions = tuple(info["object_pos_list"])
            if last_obj_positions is not None and cur_obj_positions != last_obj_positions:
                object_move_steps += 1
            last_obj_positions = cur_obj_positions

        steps += 1
        actor_obs, critic_obs = actor_obs_next, critic_obs_next

        # Stop
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    term_reason = info.get("terminated_by", "unknown")
    success_full = (term_reason == "all_goals_and_objects")
    success_any_delivery = (info.get("objects_delivered_total", 0) > 0)
    mean_action_probs = (probs_running_sum / max(probs_count, 1)) if probs_running_sum is not None else np.zeros(4)

    # Base metrics
    metrics = {
        "eval/steps": steps,
        "eval/team_return": float(team_return),
        "eval/per_agent_return_mean": float(per_agent_return.mean()),
        "eval/per_agent_return_sum": per_agent_return.tolist(),
        "eval/objects_delivered_total": int(info.get("objects_delivered_total", deliveries)),
        "eval/objects_delivered_first_t": int(first_delivery_t) if first_delivery_t is not None else -1,
        "eval/object_move_steps": int(object_move_steps),
    }

    # Action prob metrics (dynamic: 4 or 5 actions)
    action_names = ["up", "right", "down", "left", "stay"][: len(mean_action_probs)]
    for idx, name in enumerate(action_names):
        metrics[f"eval/action_prob_{name}"] = float(mean_action_probs[idx])

    # Save GIF (if any frames)
    if record and gif_path and frames and len(frames) > 0:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        try:
            save_gif(frames, gif_path, fps=10)
            metrics["eval/gif_path"] = gif_path
        except Exception as e:
            metrics["eval/gif_error"] = str(e)

    # Log to W&B (metrics + video)
    if log_wandb:
        term_flags = {
            "eval/term/time_limit": 1.0 if term_reason == "time_limit" else 0.0,
            "eval/term/catastrophe": 1.0 if term_reason == "catastrophe" else 0.0,
            "eval/term/full_success": 1.0 if term_reason == "all_goals_and_objects" else 0.0,
            "eval/term/other": 1.0 if term_reason not in ("time_limit", "catastrophe", "all_goals_and_objects") else 0.0,
        }
        wandb.log({**metrics, **term_flags}, step=wb_step)
        if record and gif_path and os.path.exists(gif_path):
            try:
                wandb.log({"eval/gif": wandb.Video(gif_path, fps=10, format="gif")}, step=wb_step)
            except Exception as e:
                wandb.log({"eval/gif_upload_error": str(e)}, step=wb_step)

    # Restore exploration epsilon
    if eps_restore is not None and hasattr(model.actor, "set_exploration_eps"):
        model.actor.set_exploration_eps(eps_restore)

    return metrics