from typing import Dict, Optional
import matplotlib.pyplot as plt
from utils.general import _ensure_dir, save_gif, capture_frame
import torch
import numpy as np
from agents.case_study_2_1.ppo_utils.helpers import to_tchw, ego_list_to_tchw
import wandb
import os


def output_metric_comparison(results: Dict):
    methods = results.keys()

    first_method_key = next(iter(methods))
    if 'losses' in results[first_method_key] and 'mean' in results[first_method_key]['losses']:
         num_rounds = len(results[first_method_key]['losses']['mean'])
         rounds = range(1, num_rounds + 1)

         plt.figure(figsize=(10, 6))
         for method in methods:
             if 'losses' in results[method] and 'mean' in results[method]['losses'] and 'std' in results[method]['losses']:
                 mean_losses = results[method]['losses']['mean']
                 std_losses = results[method]['losses']['std']
                 plt.plot(rounds, mean_losses, label=f'{method} (Mean)')
                 plt.fill_between(rounds, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2)

         plt.xlabel("Communication Rounds")
         plt.ylabel("Average Training Loss")
         plt.title("Average Training Loss over Communication Rounds (Mean ± Std)")
         plt.legend()
         plt.grid(True)
         plt.show()

         metrics_to_plot = []
         if 'metrics' in results[first_method_key]:
              metrics_to_plot = results[first_method_key]['metrics'].keys()

         for metric_name in metrics_to_plot:
             plt.figure(figsize=(10, 6))
             for method in methods:
                 if metric_name in results[method]['metrics'] and 'mean' in results[method]['metrics'][metric_name] and 'std' in results[method]['metrics'][metric_name]:
                      mean_metric = results[method]['metrics'][metric_name]['mean']
                      std_metric = results[method]['metrics'][metric_name]['std']
                      plt.plot(rounds, mean_metric, label=method)
                      plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)

             plt.xlabel("Communication Rounds")
             plt.ylabel(metric_name)
             plt.title(f"{metric_name} over Communication Rounds (Mean ± Std)")
             plt.legend()
             plt.grid(True)
             plt.savefig(f"{metric_name}_over_rounds.png")
             plt.show()

         print("\nPlotting finished.")


@torch.no_grad()
def dist_critic_diagnostics(
    z: torch.Tensor,             # [B, N] 
    taus: torch.Tensor,          # [N]
) -> Dict[str, float]:
    z_sorted, _ = torch.sort(z, dim=-1)
    z_mean = z_sorted.mean(dim=0)                   # [N]
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
        "_z_mean": z_mean.cpu().numpy(),
        "_z_q25": z_q25.cpu().numpy(),
        "_z_q75": z_q75.cpu().numpy(),
    }


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
    wb_step: Optional[int] = None,  
) -> dict:
    model.actor.eval()
    model.critic.eval()

    (actor_obs, _), _ = env.reset()
    n_agents = env.cfg.n_agents

    has_gru = hasattr(model.actor, "init_hidden") and hasattr(model.actor, "hidden_dim")
    h = model.actor.init_hidden(n_agents, device) if has_gru else None

    eps_restore = None
    if hasattr(model.actor, "set_exploration_eps"):
        eps_restore = float(getattr(model.actor, "_eps_explore", 0.0))
        model.actor.set_exploration_eps(0.0)

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
    if record:
        f0 = capture_frame(env)
        if f0 is not None:
            frames.append(f0)

    while True:
        ego_bna = ego_list_to_tchw(actor_obs).to(device)       # [A,C,k,k]
        agent_ids = torch.arange(n_agents, dtype=torch.long, device=device)

        if has_gru:
            if "eps_override" in model.actor.forward.__code__.co_varnames:
                dists, h = model.actor(ego_bna, agent_ids, h_in=h, eps_override=0.0)
            else:
                dists, h = model.actor(ego_bna, agent_ids, h_in=h)
        else:
            out = model.actor(ego_bna, agent_ids)
            dists = out[0] if isinstance(out, tuple) else out

        if deterministic:
            actions = torch.argmax(dists.probs, dim=-1)  # [A]
        else:
            actions = dists.sample()

        # track action probs (support 4 or 5 actions)
        probs = dists.probs.detach().cpu().numpy()       # [A, n_actions]
        if probs_running_sum is None:
            probs_running_sum = np.zeros(probs.shape[1], dtype=np.float64)
        probs_running_sum += probs.mean(axis=0)
        probs_count += 1

        act_dict = {i: int(actions[i].item()) for i in range(n_agents)}
        (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = env.step(act_dict)

        # record frame after step
        if record:
            f = capture_frame(env)
            if f is not None:
                frames.append(f)

        team_return += float(team_rew)
        if "rewards_per_agent" in info and isinstance(info["rewards_per_agent"], (list, tuple)):
            per_agent_return += np.array(info["rewards_per_agent"], dtype=np.float64)

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

        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    term_reason = info.get("terminated_by", "unknown")
    mean_action_probs = (probs_running_sum / max(probs_count, 1)) if probs_running_sum is not None else np.zeros(4)

    metrics = {
        "eval/steps": steps,
        "eval/team_return": float(team_return),
        "eval/per_agent_return_mean": float(per_agent_return.mean()),
        "eval/objects_delivered_total": int(info.get("objects_delivered_total", deliveries)),
        "eval/objects_delivered_first_t": int(first_delivery_t) if first_delivery_t is not None else -1,
        "eval/object_move_steps": int(object_move_steps),
    }

    action_names = ["up", "right", "down", "left", "stay"][: len(mean_action_probs)]
    for idx, name in enumerate(action_names):
        metrics[f"eval/action_prob_{name}"] = float(mean_action_probs[idx])

    if record and gif_path and frames and len(frames) > 0:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        try:
            save_gif(frames, gif_path, fps=10)
        except Exception as e:
            print(f"WARNING: failed to save eval GIF to {gif_path}: {e}")

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
            desired_name = os.path.basename(gif_path)
            art.add_file(gif_path, name=desired_name)
            wandb.log_artifact(art)
        except Exception as e:
            wandb.log({"eval/gif_artifact_error": str(e)}, step=wb_step)

    if eps_restore is not None and hasattr(model.actor, "set_exploration_eps"):
        model.actor.set_exploration_eps(eps_restore)

    return metrics