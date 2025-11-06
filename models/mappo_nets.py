#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mappo_nets.py — Actor-Critic with optional forward-time prior regularization.

Highlights
- Critic trunk exposed as `.encoder` (compat with fedtrunc).
- Optional external spatial encoder (AE) still supported via attach_spatial_encoder().
- Forward-time prior path:
    * affine shrinkage toward a FROZEN prior critic
    * soft tanh tube clamp with radius delta^beta
  By default, the prior uses the FULL aggregated critic (encoder+proj+head).
- Local/fedavg training is unchanged unless prior is enabled.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_hw(h: int, w: int, k: int = 3, s: int = 1, p: int = 1) -> Tuple[int, int]:
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    return oh, ow


# -------------------- Prior regularization --------------------

@dataclass
class PriorRegCfg:
    enabled: bool = False        # turn on affine shrink + soft tube clamp
    alpha: float = 0.5           # shrink: y_pri + alpha * (y_loc - y_pri)
    beta: float = 1.0            # exponent in delta^beta
    radius_abs: float = 0.0      # absolute tube radius; if >0 overrides relative
    radius_rel: float = 0.10     # relative tube radius fraction
    use_full_trunk: bool = True  # if True, prior path uses full averaged trunk; else head-only


def _soft_tube(y: torch.Tensor, y0: torch.Tensor, radius: float, beta: float) -> torch.Tensor:
    """
    y0 + delta * tanh((y - y0) / delta^beta)
    """
    if radius <= 0.0:
        return y
    beta_eff = max(beta, 1e-8)
    denom = (radius ** beta_eff)
    return y0 + radius * torch.tanh((y - y0) / denom)


# -------------------- Encoders --------------------

class EgoEncoder(nn.Module):
    def __init__(self, in_ch: int = 7, k: int = 5, feat_dim: int = 128, agent_id_dim: int = 8, n_agents: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        oh, ow = conv_out_hw(k, k, 3, 1, 1)
        flat_dim = 64 * oh * ow
        self.id_embed = nn.Embedding(num_embeddings=max(n_agents, 16), embedding_dim=agent_id_dim)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim + agent_id_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        eid = self.id_embed(agent_ids)
        h = torch.cat([h, eid], dim=1)
        return self.mlp(h)


class GlobalEncoder(nn.Module):
    """
    Convolutional trunk for critic.
    """
    def __init__(self, in_ch: int = 6, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self._feat_dim = feat_dim
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, output_size=(3, 3)).flatten(1)
        return self.head(h)


# -------------------- Actor --------------------

class ActorPolicy(nn.Module):
    def __init__(self, n_actions: int = 5, ego_k: int = 5, n_agents: int = 3,
                 hidden_dim: int = 128, eps_explore: float = 0.05):
        super().__init__()
        self.encoder = EgoEncoder(in_ch=7, k=ego_k, feat_dim=128, n_agents=n_agents)
        self.gru = nn.GRUCell(input_size=128, hidden_size=hidden_dim)
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.hidden_dim = hidden_dim
        self._eps_explore = eps_explore

    @property
    def eps_explore(self) -> float:
        return self._eps_explore

    def set_exploration_eps(self, eps: float):
        self._eps_explore = float(eps)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def forward(
        self,
        ego: torch.Tensor,
        agent_ids: torch.Tensor,
        h_in: Optional[torch.Tensor] = None,
        eps_override: Optional[float] = None,
    ):
        z = self.encoder(ego, agent_ids)
        if h_in is None:
            h_in = torch.zeros(z.size(0), self.hidden_dim, device=z.device)
        h_out = self.gru(z, h_in)
        logits = self.pi(h_out)

        eps = self._eps_explore if eps_override is None else eps_override
        if eps > 0.0:
            probs = torch.softmax(logits, dim=-1)
            A = probs.size(-1)
            probs = (1.0 - eps) * probs + eps / float(A)
            dist = torch.distributions.Categorical(probs=probs)
        else:
            dist = torch.distributions.Categorical(logits=logits)
        return dist, h_out


# -------------------- Critic base (with AE hook + prior support) --------------------

class _CriticBase(nn.Module):
    """
    - Public .encoder: local trunk used by the critic (compat with older fedtrunc).
    - External spatial encoder (AE) can be attached. If attached, it's used in _encode() with no grads.
    - Prior support:
        * frozen copies: prior_encoder, prior_proj, and a prior head in subclasses
        * configurable via PriorRegCfg (incl. use_full_trunk)
    """
    def __init__(self, feat_dim: int = 256, in_ch: int = 6):
        super().__init__()
        self._feat_dim = feat_dim

        # Local trunk (public for fedtrunc compatibility)
        self.encoder = GlobalEncoder(in_ch=in_ch, feat_dim=feat_dim)

        # Optional external encoder (AE); if provided, gradients are stopped
        self._ext_spatial_encoder: Optional[nn.Module] = None

        # Local projection, built lazily if ext encoder latent dim != feat_dim
        self._proj: Optional[nn.Module] = None

        # Prior configuration + frozen prior copies
        self._prior_cfg = PriorRegCfg()
        self._prior_encoder: Optional[nn.Module] = None
        self._prior_proj: Optional[nn.Module] = None  # mirrors self._proj when needed

    def attach_spatial_encoder(self, enc: nn.Module):
        """
        enc(x): [B, C_latent, n, n] spatial latent (e.g., AE encoder).
        """
        self._ext_spatial_encoder = enc

    def set_prior_reg_cfg(self, cfg: PriorRegCfg):
        self._prior_cfg = cfg

    def _encode_local(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using local trunk (.encoder) OR external AE + proj if attached.
        """
        if self._ext_spatial_encoder is None:
            return self.encoder(x)  # local trunk to feat_dim

        # Use external spatial encoder without gradients, then pool + optional proj
        with torch.no_grad():
            z_spat = self._ext_spatial_encoder(x)  # [B, C_l, n, n]
        z_vec = z_spat.mean(dim=(2, 3))           # [B, C_l]

        if self._proj is None or (isinstance(self._proj, nn.Identity) and z_vec.size(-1) != self._feat_dim):
            in_dim = z_vec.size(-1)
            if in_dim == self._feat_dim:
                self._proj = nn.Identity()
            else:
                dev = next(self.parameters()).device
                self._proj = nn.Sequential(
                    nn.Linear(in_dim, self._feat_dim), nn.ReLU(inplace=True),
                ).to(dev)
        return self._proj(z_vec) if self._proj is not None else z_vec  # [B, feat_dim]

    def _encode_prior(self, x: torch.Tensor, head_only_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        When use_full_trunk==True and prior trunk exists, run x through the frozen prior trunk (+ prior proj).
        Otherwise, reuse head_only_features (z from local or ext path).
        """
        if self._prior_cfg.use_full_trunk and self._prior_encoder is not None:
            with torch.no_grad():
                z = self._prior_encoder(x)
            # mirror local proj if any
            if self._proj is not None and not isinstance(self._proj, nn.Identity):
                if self._prior_proj is None:
                    self._prior_proj = copy.deepcopy(self._proj).to(next(self.parameters()).device)
                    for p in self._prior_proj.parameters():
                        p.requires_grad_(False)
                with torch.no_grad():
                    z = self._prior_proj(z)
            return z
        # head-only prior: use the same features that local head received
        if head_only_features is None:
            # safest fallback
            return self._encode_local(x)
        return head_only_features


# -------------------- Central (expected) critic --------------------

class CentralCritic(_CriticBase):
    def __init__(self, in_ch: int = 6, feat_dim: int = 256):
        super().__init__(feat_dim=feat_dim, in_ch=in_ch)
        self.v = nn.Linear(feat_dim, 1)

        # Frozen prior head (same shape)
        self.v_prior = nn.Linear(feat_dim, 1)
        for p in self.v_prior.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_prior_from_state_dict(self, prior_sd: dict) -> None:
        """
        Update the frozen PRIOR from a full aggregated critic state_dict.
        Copies:
          - head weights (v.*)
          - trunk weights into self._prior_encoder (if present in state dict)
        """
        # 1) Head
        if "v.weight" in prior_sd and "v.bias" in prior_sd:
            self.v_prior.weight.copy_(prior_sd["v.weight"])
            self.v_prior.bias.copy_(prior_sd["v.bias"])

        # 2) Trunk (full prior by default)
        if self._prior_cfg.use_full_trunk:
            # Ensure a frozen prior trunk exists
            if self._prior_encoder is None:
                self._prior_encoder = GlobalEncoder(in_ch=6, feat_dim=self._feat_dim).to(next(self.parameters()).device)
                for p in self._prior_encoder.parameters():
                    p.requires_grad_(False)
            # Load encoder.* sub-state if provided
            enc_subsd = {k.split("encoder.", 1)[1]: v for k, v in prior_sd.items() if k.startswith("encoder.")}
            if enc_subsd:
                try:
                    self._prior_encoder.load_state_dict(enc_subsd, strict=False)
                except Exception:
                    pass  # be tolerant to partial state dicts

    def forward(self, global_planes: torch.Tensor) -> torch.Tensor:
        # Local path
        z_loc = self._encode_local(global_planes)          # [B, feat_dim]
        v_loc = self.v(z_loc).squeeze(-1)                  # [B]

        cfg = self._prior_cfg
        if not cfg.enabled:
            return v_loc

        # Prior path (full trunk or head-only)
        z_pri = self._encode_prior(global_planes, head_only_features=z_loc)
        with torch.no_grad():
            v_pri = self.v_prior(z_pri).squeeze(-1)        # [B]

        # Affine shrinkage toward prior
        v_shrunk = v_pri + cfg.alpha * (v_loc - v_pri)

        # Soft tube clamp
        base = torch.maximum(torch.ones_like(v_pri), torch.abs(v_pri))
        delta = cfg.radius_abs if cfg.radius_abs > 0.0 else float(cfg.radius_rel) * base
        delta_scalar = float(delta.mean().item())
        return _soft_tube(v_shrunk, v_pri, radius=delta_scalar, beta=cfg.beta)

    @torch.no_grad()
    def mean_value(self, global_planes: torch.Tensor) -> torch.Tensor:
        return self.forward(global_planes)


# -------------------- Distributional critic --------------------

class DistValueCritic(_CriticBase):
    def __init__(self, in_ch: int = 6, feat_dim: int = 256, n_quantiles: int = 51,
                 v_min: float = -1100.0, v_max: float = 1200.0, squash_temp: float = 10.0):
        super().__init__(feat_dim=feat_dim, in_ch=in_ch)
        self.n_quantiles = n_quantiles
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.squash_temp = float(squash_temp)

        self.head = nn.Linear(feat_dim, n_quantiles)
        nn.init.zeros_(self.head.weight)

        # Bias so mean ~ 0 after squashing
        target_mean = 0.0
        c = 0.5 * (self.v_max + self.v_min)
        h = 0.5 * (self.v_max - self.v_min)
        y = (target_mean - c) / max(h, 1e-6)
        y = float(max(-0.999, min(0.999, y)))
        bias0 = self.squash_temp * math.atanh(y)
        nn.init.constant_(self.head.bias, bias0)

        # Frozen prior head
        self.head_prior = nn.Linear(feat_dim, n_quantiles)
        for p in self.head_prior.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_prior_from_state_dict(self, prior_sd: dict) -> None:
        """
        Update frozen prior from full aggregated critic state_dict.
        Copies:
          - head weights (head.*)
          - trunk weights into self._prior_encoder if present
        """
        # 1) Head
        if "head.weight" in prior_sd and "head.bias" in prior_sd:
            self.head_prior.weight.copy_(prior_sd["head.weight"])
            self.head_prior.bias.copy_(prior_sd["head.bias"])

        # 2) Trunk
        if self._prior_cfg.use_full_trunk:
            if self._prior_encoder is None:
                self._prior_encoder = GlobalEncoder(in_ch=6, feat_dim=self._feat_dim).to(next(self.parameters()).device)
                for p in self._prior_encoder.parameters():
                    p.requires_grad_(False)
            enc_subsd = {k.split("encoder.", 1)[1]: v for k, v in prior_sd.items() if k.startswith("encoder.")}
            if enc_subsd:
                try:
                    self._prior_encoder.load_state_dict(enc_subsd, strict=False)
                except Exception:
                    pass

    def _squash(self, q_raw: torch.Tensor) -> torch.Tensor:
        c = 0.5 * (self.v_max + self.v_min)
        h = 0.5 * (self.v_max - self.v_min)
        return c + h * torch.tanh(q_raw / self.squash_temp)

    def forward(self, global_planes: torch.Tensor) -> torch.Tensor:
        # Local path
        z_loc = self._encode_local(global_planes)
        q_loc = self._squash(self.head(z_loc))             # [B, N]

        cfg = self._prior_cfg
        if not cfg.enabled:
            return q_loc

        # Prior path (full trunk or head-only)
        z_pri = self._encode_prior(global_planes, head_only_features=z_loc)
        with torch.no_grad():
            q_pri = self._squash(self.head_prior(z_pri))   # [B, N]

        # Affine shrinkage per quantile
        q_shrunk = q_pri + cfg.alpha * (q_loc - q_pri)

        # Tube radius: absolute or relative to range
        rng = (self.v_max - self.v_min)
        delta = cfg.radius_abs if cfg.radius_abs > 0.0 else float(cfg.radius_rel) * rng
        delta_scalar = float(delta)

        return _soft_tube(q_shrunk, q_pri, radius=delta_scalar, beta=cfg.beta)

    @torch.no_grad()
    def mean_value(self, global_planes: torch.Tensor) -> torch.Tensor:
        q = self.forward(global_planes)
        return q.mean(dim=-1)


# -------------------- Model wrapper --------------------

@dataclass
class MAPPOModel:
    actor: ActorPolicy
    critic: nn.Module

    @staticmethod
    def build(
        n_actions: int = 5, ego_k: int = 5, n_agents: int = 3,
        hidden_dim: int = 128, eps_explore: float = 0.05,
        critic_type: str = "expected", n_quantiles: int = 21
    ) -> "MAPPOModel":
        actor = ActorPolicy(
            n_actions=n_actions, ego_k=ego_k, n_agents=n_agents,
            hidden_dim=hidden_dim, eps_explore=eps_explore
        )
        if critic_type == "expected":
            critic = CentralCritic(in_ch=6, feat_dim=256)
        elif critic_type == "distributional":
            critic = DistValueCritic(in_ch=6, feat_dim=256, n_quantiles=n_quantiles)
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}")
        return MAPPOModel(actor=actor, critic=critic)

    # AE entry point (optional)
    def attach_spatial_encoder(self, enc: nn.Module):
        if hasattr(self.critic, "attach_spatial_encoder"):
            self.critic.attach_spatial_encoder(enc)

    # Update frozen PRIOR from server-averaged critic state_dict
    @torch.no_grad()
    def update_critic_prior(self, prior_state_dict: dict):
        if hasattr(self.critic, "update_prior_from_state_dict"):
            self.critic.update_prior_from_state_dict(prior_state_dict)

    # Toggle/configure forward-time prior regularization.
    # use_full_trunk=True by default to make the prior = full averaged critic.
    def set_prior_regularization(self, enabled: bool, alpha: float, beta: float,
                                 radius_abs: float = 0.0, radius_rel: float = 0.10,
                                 use_full_trunk: bool = True):
        if hasattr(self.critic, "set_prior_reg_cfg"):
            self.critic.set_prior_reg_cfg(PriorRegCfg(
                enabled=bool(enabled),
                alpha=float(alpha),
                beta=float(beta),
                radius_abs=float(radius_abs),
                radius_rel=float(radius_rel),
                use_full_trunk=bool(use_full_trunk),
            ))
