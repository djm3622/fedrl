from typing import Optional
from utils.general import normalize_hist

import numpy as np

import ot 

def pot_wasserstein_barycenter(
    Ps: np.ndarray, z: np.ndarray, reg: float, weights: Optional[np.ndarray] = None, num_iter: int = 20_000, p: int = 1
) -> np.ndarray:
    """
    POT Sinkhorn barycenter with safer defaults (log-stabilization via larger reg).
    Still can be less stable than quantile method in 1D; provided for completeness.
    """
    N, A = Ps.shape
    Ps = np.vstack([normalize_hist(p) for p in Ps])
    if weights is None:
        weights = np.ones(N, dtype=np.float64) / N
    weights = weights / weights.sum()

    Z = z.reshape(-1, 1)
    M = ot.utils.dist(Z, Z, metric='euclidean') ** p  # [A, A]
    Ds = Ps.T.copy()
    pbar = ot.bregman.barycenter(Ds, M, reg, weights, numItermax=num_iter, stopThr=1e-12, verbose=False)
    return normalize_hist(np.asarray(pbar, dtype=np.float64))