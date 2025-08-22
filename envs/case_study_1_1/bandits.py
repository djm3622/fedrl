from dataclasses import dataclass
from typing import List, Tuple
from utils.general import normalize_hist

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ArmDistSpec:
    kind: str             # e.g. 'gauss','bimodal', etc.
    params: Tuple[float]  # distribution parameters
    mean: float           # the (approximate) mean that was enforced


class HeteroBandit:
    def __init__(
        self,
        n_clients: int,
        n_arms: int,
        rng: np.random.Generator,
        mean_slack: float = 0.2,
        ):
        self.n_clients = n_clients
        self.n_arms = n_arms
        self.rng = rng

        # Step 1: global mean per arm
        self.arm_means = rng.uniform(-1.0, 1.0, size=n_arms)

        # Step 2: client-perturbed shapes per arm
        self.specs: List[List[ArmDistSpec]] = []
        for c in range(n_clients):
            row = []
            for a in range(n_arms):
                μ = float(self.arm_means[a])
                # perturb mean slightly
                μc = μ + rng.normal(0, mean_slack)

                shape_type = rng.choice(['gauss','bimodal','student','skewnorm'])
                if shape_type=='gauss':
                    std = rng.uniform(0.1,0.5)                    # broader
                    row.append(ArmDistSpec('gauss',(μc,std),μc))

                elif shape_type=='bimodal':
                    δ   = rng.uniform(0.3,1.0)                    # wider +/- spread
                    s1  = rng.uniform(0.05,0.8)
                    s2  = rng.uniform(0.05,0.8)
                    w   = rng.uniform(0.1,0.9)                   # rarely extreme mixture weights
                    m1  = μc - δ; m2 = μc + δ
                    row.append(ArmDistSpec('bimodal',(m1,s1,m2,s2,w),μc))

                elif shape_type=='student':
                    scale = rng.uniform(0.05,0.5)
                    df    = int(rng.integers(3,10))
                    row.append(ArmDistSpec('student',(μc,scale,df),μc))

                else:
                    scale = rng.uniform(0.05,0.5)
                    alpha = rng.uniform(-8,8)
                    row.append(ArmDistSpec('skewnorm',(μc,scale,alpha),μc))
            self.specs.append(row)

    def sample(self,client_id:int,arm_id:int,n:int)->np.ndarray:
        spec = self.specs[client_id][arm_id]
        if spec.kind=='gauss':
            m,s = spec.params
            return self.rng.normal(m,s,size=n)
        if spec.kind=='bimodal':
            m1,s1,m2,s2,w = spec.params
            u  = self.rng.uniform(0,1,size=n)
            x1 = self.rng.normal(m1,s1,size=n)
            x2 = self.rng.normal(m2,s2,size=n)
            return np.where(u<w,x1,x2)
        if spec.kind=='student':
            m,scale,df = spec.params
            return self.rng.standard_t(df,size=n)*scale + m
        # skewnorm
        m,scale,alpha = spec.params
        u0 = self.rng.normal(0,1,size=n)
        v  = self.rng.normal(0,1,size=n)
        u1 = alpha*u0 + v
        x  = np.where(u0>=0, u1, -u1)
        return x*scale + m

    def estimate_truth_hist(
        self, z: np.ndarray, client_id: int, arm_id: int,
        nsamp: int = 30000
    ) -> np.ndarray:
        xs = self.sample(client_id, arm_id, nsamp)
        dz = z[1] - z[0]
        edges = np.concatenate(([z[0] - dz/2], (z[:-1] + z[1:]) / 2, [z[-1] + dz/2]))
        hist, _ = np.histogram(xs, bins=edges)
        return normalize_hist(hist.astype(np.float64))

    def true_mean(self, z: np.ndarray, client_id: int, arm_id: int) -> float:
        p = self.estimate_truth_hist(z, client_id, arm_id, nsamp=200000)
        return float(np.dot(p, z))

    def true_cvar(self, z: np.ndarray, client_id: int, arm_id: int, alpha: float = 0.1) -> float:
        p = self.estimate_truth_hist(z, client_id, arm_id, nsamp=200000)
        cdf = np.cumsum(p)
        w   = np.diff(np.concatenate(([0.0], np.minimum(cdf, alpha))))
        return float((w * z).sum() / (alpha + 1e-12))
    
class C51Bandit(nn.Module):
    def __init__(self, n_arms: int, feat_dim: int, hidden: int, n_atoms: int):
        super().__init__()
        self.n_arms = n_arms
        self.n_atoms = n_atoms
        self.emb = nn.Embedding(n_arms, feat_dim)   # shared across clients
        self.fc1 = nn.Linear(feat_dim, hidden)      # shared across clients
        self.fc2 = nn.Linear(hidden, n_atoms)       # personalized (kept local in FedTrunc)

    def forward(self, arm_idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(arm_idx)                                       # [B,d]
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)               # row-norm to prevent collapse
        h = F.relu(self.fc1(e))                                     # [B,H]
        logits = self.fc2(h)                                        # [B,A]
        return F.softmax(logits, dim=-1)

    def logits(self, arm_idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(arm_idx)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        h = F.relu(self.fc1(e))
        return self.fc2(h)