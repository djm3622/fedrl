import torch
from typing import Tuple, List, Optional, Dict
import numpy as np
from typing import Optional
from utils.general import capture_frame, save_gif, _ensure_dir
import wandb
import os
import matplotlib.pyplot as plt


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
        actor_obs, _ = actor_obs_next, critic_obs_next

        # Stop
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    term_reason = info.get("terminated_by", "unknown")
    mean_action_probs = (probs_running_sum / max(probs_count, 1)) if probs_running_sum is not None else np.zeros(4)

    # Base metrics
    metrics = {
        "eval/steps": steps,
        "eval/team_return": float(team_return),
        "eval/per_agent_return_mean": float(per_agent_return.mean()),
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
        except Exception as e:
            print(f"WARNING: failed to save eval GIF to {gif_path}: {e}")

    # Log to W&B (metrics + video)
    if log_wandb:
        term_flags = {
            "eval/term/time_limit": 1.0 if term_reason == "time_limit" else 0.0,
            "eval/term/catastrophe": 1.0 if term_reason == "catastrophe" else 0.0,
            "eval/term/full_success": 1.0 if term_reason == "all_goals_and_objects" else 0.0,
            "eval/term/other": 1.0 if term_reason not in ("time_limit", "catastrophe", "all_goals_and_objects") else 0.0,
        }
        wandb.log({**metrics, **term_flags}, step=wb_step)
        try:
            art = wandb.Artifact("eval_gifs", type="evaluation")
            # store under your chosen name
            desired_name = os.path.basename(gif_path)
            art.add_file(gif_path, name=desired_name)
            wandb.log_artifact(art)
        except Exception as e:
            wandb.log({"eval/gif_artifact_error": str(e)}, step=wb_step)

    # Restore exploration epsilon
    if eps_restore is not None and hasattr(model.actor, "set_exploration_eps"):
        model.actor.set_exploration_eps(eps_restore)

    return metrics


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    pred:   [B, N] predicted quantiles (per fixed taus)
    target: [B, N] target samples at same quantile positions
    taus:   [N]    fixed quantile levels in (0,1)

    Returns scalar loss (mean over batch and quantiles).
    """
    # pairwise TD errors at corresponding quantiles
    u = target.detach() - pred  # [B, N]
    abs_u = u.abs()
    # Huber
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    # Asymmetric quantile weighting
    tau = taus.view(1, -1)  # [1, N]
    loss = torch.where(u < 0.0, (1.0 - tau) * huber, tau * huber)
    return loss.mean()


@torch.no_grad()
def dist_critic_diagnostics(
    z: torch.Tensor,             # [B, N] quantiles from critic(glob_batch)
    taus: torch.Tensor,          # [N]
) -> Dict[str, float]:
    """
    Returns scalar diagnostics computed from z.
    Assumes z is NOT necessarily sorted; will sort for stats only (no grads).
    """
    # sort along quantile axis to make the function monotone for reporting
    z_sorted, _ = torch.sort(z, dim=-1)
    # batch mean across states of each quantile
    z_mean = z_sorted.mean(dim=0)                   # [N]
    # percentiles over states at each tau for ribbon
    z_q25 = z_sorted.quantile(0.25, dim=0)
    z_q75 = z_sorted.quantile(0.75, dim=0)

    # aggregate scalar moments on the induced distribution per state, then mean
    v_mean = z_sorted.mean(dim=-1)                  # [B]
    v_std = z_sorted.std(dim=-1, unbiased=False)    # [B]

    def interp_at(p: float) -> torch.Tensor:
        """
        Linear interpolation of the batch of quantile functions at probability p in (0,1).
        Returns a tensor of shape [B].
        """
        z_sorted_local = z_sorted  # [B, N]
        nq = taus.numel()

        # fractional position along the quantile axis
        pos = p * (nq - 1)

        # make it a tensor on the correct device/dtype, then clamp
        pos_t = torch.tensor(pos, dtype=z_sorted_local.dtype, device=z_sorted_local.device)
        pos_t = torch.clamp(pos_t, 0.0, float(nq - 1))

        # integer neighbors and linear weight
        lo = pos_t.floor().long()                         # 0D long tensor
        hi = torch.clamp(lo + 1, max=nq - 1)              # 0D long tensor
        w = (pos_t - lo.to(z_sorted_local.dtype))         # 0D float tensor in [0,1]

        # gather needs 1D indices per batch; expand to [B,1]
        B = z_sorted_local.size(0)
        lo_idx = lo.expand(B).unsqueeze(1)                # [B,1]
        hi_idx = hi.expand(B).unsqueeze(1)                # [B,1]

        z_lo = torch.gather(z_sorted_local, dim=-1, index=lo_idx).squeeze(1)  # [B]
        z_hi = torch.gather(z_sorted_local, dim=-1, index=hi_idx).squeeze(1)  # [B]

        return (1.0 - w) * z_lo + w * z_hi                # [B]


    p05 = interp_at(0.05).mean().item()
    p50 = interp_at(0.50).mean().item()
    p95 = interp_at(0.95).mean().item()

    # cvar at alpha: mean of lower tail up to quantile alpha
    def cvar_alpha(alpha: float) -> float:
        k = max(1, int(alpha * taus.numel()))
        return z_sorted[:, :k].mean().item()

    return {
        "v_mean_mb": v_mean.mean().item(),
        "v_std_mb": v_std.mean().item(),
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "cvar10": cvar_alpha(0.10),
        # for plotting
        "_z_mean": z_mean.cpu().numpy(),
        "_z_q25": z_q25.cpu().numpy(),
        "_z_q75": z_q75.cpu().numpy(),
    }

@torch.no_grad()
def plot_quantile_ribbon(
    taus: torch.Tensor,
    z_mean_np: np.ndarray,
    z_q25_np: np.ndarray,
    z_q75_np: np.ndarray,
    out_path: str,
    title: str,
):
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(6,4), dpi=150)
    t = taus.detach().cpu().numpy()
    plt.plot(t, z_mean_np, label="batch mean quantile curve")
    plt.fill_between(t, z_q25_np, z_q75_np, alpha=0.25, label="IQR across states")
    plt.xlabel("tau")
    plt.ylabel("quantile value")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@torch.no_grad()
def plot_example_states(
    taus: torch.Tensor,
    z: torch.Tensor,            # [B, N], NOT necessarily sorted
    out_path: str,
    title: str,
):
    _ensure_dir(os.path.dirname(out_path))
    # choose three representative states by mean value: low, median, high
    z_sorted, _ = torch.sort(z, dim=-1)
    v_mean = z_sorted.mean(dim=-1)
    order = torch.argsort(v_mean)
    idxs = [order[0].item(), order[len(order)//2].item(), order[-1].item()] if z.size(0) >= 3 else list(range(z.size(0)))

    t = taus.detach().cpu().numpy()
    plt.figure(figsize=(6,4), dpi=150)
    for i in idxs:
        zi = z_sorted[i].detach().cpu().numpy()
        plt.plot(t, zi, label=f"state idx {i}")
    plt.xlabel("tau")
    plt.ylabel("quantile value")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@torch.no_grad()
def log_distributional_visuals_wandb(
    glob_batch: torch.Tensor,   # [B, C, H, W]
    model,
    device: torch.device,
    wb_step: Optional[int],
    split: str = "train",       # or "eval"
    max_batch: int = 128,
):
    """
    Creates and logs:
      1) Quantile ribbon over batch (mean and IQR).
      2) Example state quantile curves (low/median/high).
      3) Scalar risk diagnostics.
    No-op for expected critic.
    """
    if not hasattr(model.critic, "taus"):
        return  # expected critic: nothing to visualize

    # slice a manageable batch from whatever you give us
    B = glob_batch.size(0)
    mb = min(B, max_batch)
    z = model.critic(glob_batch[:mb].to(device))  # [mb, N]
    taus = model.critic.taus

    diags = dist_critic_diagnostics(z, taus)
    # plots
    out_dir = os.path.join("outputs", "vis")
    fig1 = os.path.join(out_dir, f"{split}_quantile_ribbon_step_{wb_step}.png")
    fig2 = os.path.join(out_dir, f"{split}_example_states_step_{wb_step}.png")

    plot_quantile_ribbon(taus, diags["_z_mean"], diags["_z_q25"], diags["_z_q75"], out_path=fig1,
                         title=f"{split.upper()} critic: batch quantile ribbon")
    plot_example_states(taus, z, out_path=fig2,
                        title=f"{split.upper()} critic: example state quantiles")

    # log images + diagnostics
    try:
        wandb.log({
            f"{split}/critic_quantile_ribbon": wandb.Image(fig1),
            f"{split}/critic_example_states": wandb.Image(fig2),
            f"{split}/critic_v_mean_mb": diags["v_mean_mb"],
            f"{split}/critic_v_std_mb": diags["v_std_mb"],
            f"{split}/critic_p05": diags["p05"],
            f"{split}/critic_p50": diags["p50"],
            f"{split}/critic_p95": diags["p95"],
            f"{split}/critic_cvar10": diags["cvar10"],
        }, step=wb_step)
    except Exception as e:
        wandb.log({f"{split}/critic_vis_error": str(e)}, step=wb_step)
