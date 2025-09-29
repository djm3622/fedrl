from typing import Dict, List, Optional
import numpy as np
import torch


def normalize_participation_weights(full_w: np.ndarray, participate: Optional[List[int]]) -> np.ndarray:
    if participate is None:
        participate = list(range(len(full_w)))
    w = np.asarray(full_w, dtype=np.float64)[np.asarray(participate, dtype=int)]
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    return (w / s) if s > 0.0 else np.ones_like(w) / len(w)


@torch.no_grad()
def weighted_average_state_dict(state_dicts: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    avg = {}
    for k in state_dicts[0].keys():
        stacked = torch.stack([sd[k] for sd in state_dicts], dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, *([1] * (stacked.ndim - 1)))
        avg[k] = (w * stacked).sum(dim=0)
    return avg