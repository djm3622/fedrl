from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import imageio.v2 as iio

from utils.general import _ensure_dir

import matplotlib
matplotlib.use("Agg")  # belt-and-suspenders; also must precede pyplot


@torch.no_grad()
def save_round_plots(
    *,
    algo: str,
    round_idx: int,
    client_models: Sequence[torch.nn.Module],
    env,
    z: np.ndarray,                         # shape [A]
    out_root: str = "plots",
    truth_nsamp: int = 50_000,
    global_pbars: Optional[np.ndarray] = None,  # shape [K, A] if provided
    loo_pbars: Optional[np.ndarray] = None,     # shape [N, K, A] if provided
    save_npz: bool = False,
) -> None:

    out_dir = Path(out_root) / algo / f"round_{round_idx:03d}"
    _ensure_dir(out_dir)

    device = next(client_models[0].parameters()).device
    K = int(client_models[0].n_arms)

    # --- sanitize shapes/types ---
    z = np.asarray(z, dtype=float).ravel()

    dz = float(z[1] - z[0]) if z.size > 1 else 1.0

    for i, model in enumerate(client_models):
        for a in range(K):
            # Predicted categorical distribution for (client i, arm a)
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


def compile_gifs(
    *,
    algo: str,
    out_root: str,
    n_clients: int,
    n_arms: int,
    fps: int = 6,
    pattern: str = "round_*",
) -> None:

    base = Path(out_root) / algo
    gif_dir = base / "gifs"
    _ensure_dir(gif_dir)

    rounds = sorted(base.glob(pattern))
    for i in range(n_clients):
        for a in range(n_arms):
            frames = []
            for r in rounds:
                f = r / f"client_{i:02d}_arm_{a:02d}.png"
                if f.exists():
                    frames.append(iio.imread(f))
            if frames:
                out = gif_dir / f"client_{i:02d}_arm_{a:02d}.gif"
                iio.mimsave(out, frames, duration=1.0 / fps)
