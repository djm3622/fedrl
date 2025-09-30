from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import imageio.v2 as iio # pyright: ignore[reportMissingImports]

from utils.general import _ensure_dir

import matplotlib
matplotlib.use("Agg")

import torch
import wandb
import os
from eval.metric_losses import dist_critic_diagnostics


@torch.no_grad()
def save_round_plots(
    *,
    algo: str,
    round_idx: int,
    client_models: Sequence[torch.nn.Module],
    env,
    z: np.ndarray,                         # shape [A]
    out_root: str = "plots",
    truth_nsamp: int = 30_000,
    global_pbars: Optional[np.ndarray] = None,  # shape [K, A]
    loo_pbars: Optional[np.ndarray] = None,     # shape [N, K, A]
    save_npz: bool = False,
) -> None:

    out_dir = Path(out_root) / algo / f"round_{round_idx:03d}"
    _ensure_dir(out_dir)

    device = next(client_models[0].parameters()).device
    K = int(client_models[0].n_arms)

    z = np.asarray(z, dtype=float).ravel()

    dz = float(z[1] - z[0]) if z.size > 1 else 1.0

    for i, model in enumerate(client_models):
        for a in range(K):
            p_pred = model(torch.tensor([[a]], device=device)).squeeze(0).detach().cpu().numpy()
            p_true = env.estimate_truth_hist(z, i, a, nsamp=truth_nsamp)

            p_pred = np.asarray(p_pred, dtype=float).ravel()
            p_true = np.asarray(p_true, dtype=float).ravel()
            assert z.ndim == p_pred.ndim == 1 and z.size == p_pred.size, \
                f"shape mismatch: z {z.shape} vs p_pred {p_pred.shape}"

            # Optional teachers
            p_global = global_pbars[a] if global_pbars is not None else None
            p_loo    = loo_pbars[i, a]  if loo_pbars is not None else None

            ymax = 1.1 * float(
                max(
                    np.max(p_pred),
                    np.max(p_true),
                    np.max(p_global) if p_global is not None else 0.0,
                    np.max(p_loo)    if p_loo    is not None else 0.0,
                )
            ) or 1.0

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.bar(z, p_pred, width=0.9 * dz, alpha=0.5, label="Model")
            ax.plot(z, p_true, linewidth=2.0, label="Truth")
            if p_global is not None:
                ax.plot(z, p_global, linestyle="--", linewidth=2.0, label="Global (bary)")
            if p_loo is not None:
                ax.plot(z, p_loo, linestyle=":", linewidth=1.5, label="LOO (bary)")

            ax.set_xlabel("Atom support (z)")
            ax.set_ylabel("Probability")
            ax.set_ylim(0.0, ymax)
            ax.legend(loc="best", frameon=True, fontsize=9)
            fig.tight_layout()

            png_path = out_dir / f"client_{i:02d}_arm_{a:02d}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)

            if save_npz:
                npz_path = out_dir / f"client_{i:02d}_arm_{a:02d}.npz"
                np.savez_compressed(
                    npz_path,
                    z=z,
                    p_pred=p_pred,
                    p_true=p_true,
                    p_global=(p_global if p_global is not None else np.array([])),
                    p_loo=(p_loo if p_loo is not None else np.array([])),
                )

def compile_videos(
    *,
    algo: str,
    out_root: str,
    n_clients: int,
    n_arms: int,
    fps: int = 100,
    pattern: str = "round_*",
    ext: str = "mp4",               # "mp4" (H.264) or "webm"
    codec: str = "libx264",         # use "libvpx-vp9" for webm
    crf: int = 23,                  # lower = higher quality; 18–28 is typical
) -> None:
    base = Path(out_root) / algo
    vid_dir = base / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    rounds = sorted(base.glob(pattern))
    for i in range(n_clients):
        for a in range(n_arms):
            frames = []
            for r in rounds:
                f = r / f"client_{i:02d}_arm_{a:02d}.png"
                if f.exists():
                    frames.append(iio.imread(f))
            if not frames:
                continue

            out = vid_dir / f"client_{i:02d}_arm_{a:02d}.{ext}"
            # Requires ffmpeg in PATH
            with iio.get_writer(
                out, fps=fps, codec=codec, quality=None, macro_block_size=None
            ) as w:
                for fr in frames:
                    w.append_data(fr)


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
    if not hasattr(model.critic, "taus"):
        return  # expected critic: nothing to visualize

    # slice a manageable batch from whatever you give us
    B = glob_batch.size(0)
    mb = min(B, max_batch)
    z = model.critic(glob_batch[:mb].to(device))  # [mb, N]
    taus = model.critic.taus

    diags = dist_critic_diagnostics(z, taus)
    out_dir = os.path.join("outputs", "vis")
    fig1 = os.path.join(out_dir, f"{split}_quantile_ribbon_step_{wb_step}.png")
    fig2 = os.path.join(out_dir, f"{split}_example_states_step_{wb_step}.png")

    plot_quantile_ribbon(
        taus, diags["_z_mean"], diags["_z_q25"], diags["_z_q75"], out_path=fig1,
        title=f"{split.upper()} critic: batch quantile ribbon"
    )
    plot_example_states(
        taus, z, out_path=fig2,
        title=f"{split.upper()} critic: example state quantiles"
    )

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