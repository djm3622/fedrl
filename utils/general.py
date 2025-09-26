import os
import random
from typing import List, get_origin, get_args, Dict, Type
import yaml
from pathlib import Path
from configs.config_templates.case_study_1_1 import Config
from configs.config_templates.case_study_2_1 import MultiAgentGridConfig
import numpy as np
import torch
from dataclasses import fields
import imageio.v2 as imageio  # pyright: ignore[reportMissingImports] # for GIF writing

CONFIGS: Dict[str, Type] = {
    "case_1": Config,
    "case_2": MultiAgentGridConfig,
}

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm_vol_weights(raw_w: np.ndarray) -> np.ndarray:
    if np.any(raw_w < 0):
        raise ValueError("client_weights must be nonnegative.")
    s = raw_w.sum()
    if s <= 0:
        raise ValueError("Sum of client_weights must be > 0.")
    return (raw_w / s).astype(np.float64)


def normalize_weights(w, n_clients):
    if w is None:
        w = np.ones(n_clients, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    assert w.shape[0] == n_clients
    w = np.clip(w, 1e-12, None)
    return (w / w.sum()).astype(np.float64)


def one_hot(indices: torch.Tensor, K: int) -> torch.Tensor:
    x = torch.zeros(indices.shape[0], K, device=indices.device, dtype=torch.float32)
    x.scatter_(1, indices.view(-1,1), 1.0)
    return x


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_hist(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clamp to [0,∞), replace NaNs/Infs, and renormalize to sum=1."""
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p[p < 0] = 0.0
    s = p.sum()
    if s <= eps:
        # fallback to uniform
        p = np.ones_like(p, dtype=np.float64) / len(p)
    else:
        p = p / s
    return p


def ema(xs: List[float], alpha: float = 0.1) -> List[float]:
    """Simple exponential moving average for plotting."""
    if not xs:
        return xs
    out = []
    m = xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def make_support(vmin, vmax, n_atoms, device) -> torch.Tensor:
    return torch.linspace(vmin, vmax, n_atoms, dtype=torch.float32, device=device)


def get_random_prior(client_id: int, arm_id: int, z: torch.Tensor) -> np.ndarray:
    z_np = z.detach().cpu().numpy()
    A = len(z_np)

    # Random mean within the support range
    mu = np.random.uniform(z_np[0], z_np[-1])
    # Random stddev covering 5–30% of the support range
    sigma = np.random.uniform(
        (z_np[-1] - z_np[0]) * 0.05,
        (z_np[-1] - z_np[0]) * 0.3
    )

    # Unnormalised bell curve
    p = np.exp(-0.5 * ((z_np - mu) / (sigma + 1e-12))**2)

    # Normalise
    p = np.maximum(p, 0.0)
    p = p / (p.sum() + 1e-12)

    return torch.from_numpy(p)


def _cast_scalar(x, t):
    if t is float: return float(x)
    if t is int:   return int(x)
    if t is str:   return str(x)
    return x


def _cast_list(xs, elem_t):
    return [ _cast_scalar(x, elem_t) for x in xs ]


def load_config(path: str, config_type: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # flatten nested sections
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)

    # cast by dataclass annotations
    kwargs = {}
    c = CONFIGS.get(config_type, None)
    for f in fields(c):
        if f.name in flat:
            tgt = f.type
            val = flat[f.name]
            origin = get_origin(tgt)
            if origin in (list, List):
                (elem_t,) = get_args(tgt)
                kwargs[f.name] = _cast_list(val, elem_t)
            else:
                kwargs[f.name] = _cast_scalar(val, tgt)

    return c(**kwargs)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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