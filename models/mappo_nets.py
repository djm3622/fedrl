# models/mappo_nets.py
# Minimal MAPPO actor and centralized critic for your gridworld.
# Actor consumes ego crops (k x k x 7). Critic consumes global planes (H x W x 6).
# Comments use only ascii characters.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_hw(h: int, w: int, k: int = 3, s: int = 1, p: int = 1) -> Tuple[int, int]:
    # for conv2d with kernel k, stride s, padding p
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    return oh, ow


class EgoEncoder(nn.Module):
    def __init__(self, in_ch: int = 7, k: int = 5, feat_dim: int = 128, agent_id_dim: int = 8, n_agents: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        oh, ow = conv_out_hw(k, k, 3, 1, 1)
        flat_dim = 64 * oh * ow
        self.id_embed = nn.Embedding(num_embeddings=max(n_agents, 16), embedding_dim=agent_id_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim + agent_id_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, C=7, k, k], agent_ids: [B]
        h = self.conv(x)
        h = h.flatten(1)
        eid = self.id_embed(agent_ids)
        h = torch.cat([h, eid], dim=1)
        z = self.mlp(h)
        return z


class GlobalEncoder(nn.Module):
    def __init__(self, in_ch: int = 6, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self._feat_dim = feat_dim
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),  # assumes H=W=10 -> after strides approx 3x3
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=6, H, W]
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, output_size=(3, 3))
        h = h.flatten(1)
        z = self.head(h)
        return z


class ActorPolicy(nn.Module):
    def __init__(self, n_actions: int = 4, ego_k: int = 5, n_agents: int = 3,
                 hidden_dim: int = 128, eps_explore: float = 0.05):
        super().__init__()
        self.encoder = EgoEncoder(in_ch=7, k=ego_k, feat_dim=128, n_agents=n_agents)
        self.gru = nn.GRUCell(input_size=128, hidden_size=hidden_dim)
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.hidden_dim = hidden_dim
        self._eps_explore = eps_explore  # default exploration epsilon

    @property
    def eps_explore(self) -> float:
        return self._eps_explore

    def set_exploration_eps(self, eps: float):
        self._eps_explore = float(eps)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        # [B, hidden_dim]
        return torch.zeros(batch, self.hidden_dim, device=device)

    def forward(
        self,
        ego: torch.Tensor,            # [B, C, k, k]
        agent_ids: torch.Tensor,      # [B]
        h_in: Optional[torch.Tensor] = None,  # [B, hidden_dim]
        eps_override: Optional[float] = None,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        z = self.encoder(ego, agent_ids)      # [B, 128]
        if h_in is None:
            h_in = torch.zeros(z.size(0), self.hidden_dim, device=z.device)
        h_out = self.gru(z, h_in)             # [B, hidden_dim]
        logits = self.pi(h_out)               # [B, A]

        # --- ε-mixed exploration: p' = (1-ε) softmax + ε / A ---
        eps = self._eps_explore if eps_override is None else eps_override
        if eps > 0.0:
            probs = torch.softmax(logits, dim=-1)
            A = probs.size(-1)
            probs = (1.0 - eps) * probs + eps / float(A)
            dist = torch.distributions.Categorical(probs=probs)
        else:
            dist = torch.distributions.Categorical(logits=logits)

        return dist, h_out


class CentralCritic(nn.Module):
    # unchanged
    def __init__(self):
        super().__init__()
        self.encoder = GlobalEncoder(in_ch=6, feat_dim=256)
        self.v = nn.Linear(256, 1)
    def forward(self, global_planes: torch.Tensor) -> torch.Tensor:
        z = self.encoder(global_planes)
        v = self.v(z)
        return v.squeeze(-1)


@dataclass
class MAPPOModel:
    actor: ActorPolicy
    critic: CentralCritic

    @staticmethod
    def build(n_actions: int = 4, ego_k: int = 5, n_agents: int = 3,
              hidden_dim: int = 128, eps_explore: float = 0.05) -> "MAPPOModel":
        return MAPPOModel(
            actor=ActorPolicy(n_actions=n_actions, ego_k=ego_k, n_agents=n_agents,
                              hidden_dim=hidden_dim, eps_explore=eps_explore),
            critic=CentralCritic()
        )