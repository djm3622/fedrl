import torch
import torch.nn as nn
import torch.nn.functional as F


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


def c51_project(rewards: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    vmin, vmax = z[0].item(), z[-1].item()
    A = z.shape[0]
    delta_z = (vmax - vmin) / (A - 1)
    b = torch.clamp((rewards - vmin) / delta_z, 0, A - 1)
    l = b.floor().long()
    u = b.ceil().long()
    m = torch.zeros(rewards.shape[0], A, device=rewards.device)
    off = (b - l.float())
    m.scatter_add_(1, l.view(-1,1), (1 - off).view(-1,1))
    m.scatter_add_(1, u.view(-1,1), off.view(-1,1))
    eq = (l == u)
    if eq.any():
        m[eq, l[eq]] = 1.0
    m = m / (m.sum(dim=1, keepdim=True) + 1e-12)
    return m