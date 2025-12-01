#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer.py — Unified PPOTrainer for local, FedAvg/FedTrunc, and FedRL-prior modes.

- Local / FedAvg / FedTrunc:
    * Flat-vector helpers preserved: get_actor_flat/load_actor_flat, get_critic_flat/load_critic_flat
    * Loaders accept either: (a) flat 1-D torch.Tensor, or (b) state_dict (with or without "actor."/ "critic." prefixes)
    * reset_opt kwarg supported to clear optimizer state after cross-client swaps

- FedRL-prior:
    * Optional AE side-car remains (gated by cfg.enable_ae_aux)
    * Stability probe metrics: param_drift_critic, out_drift_probe, prior_gap_probe, clamp_effect_probe
      (all W&B-guarded and cheap; do nothing if W&B disabled)

- NEW (robust risk signal):
    * PPO actor is trained on reward-only GAE (stable baseline).
    * PLUS a second, small, UNCLIPPED auxiliary actor term that maximizes CVaR-advantage.
      This keeps tail-risk pressure alive despite normalization and PPO clipping, and
      empirically separates FedAvg from FedRL when hazards are stochastic (slip).
"""

from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
import wandb

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .helpers import (
    compute_gae, to_tchw, ego_list_to_tchw, quantile_huber_loss,
    setup_optimizers, centralized_value_mean, ppo_epoch_update
)
from federation.case_study_2_1.barycenter import RoundDistBuffer, RoundDistBufferCfg
from .buffer import RolloutBuffer
from eval.distributions import log_distributional_visuals_wandb
from eval.metric_losses import run_eval_rollout

# ---------- optional AE import (backwards compatible) ----------
_HAS_AE = False
_build_autoencoder = None
try:
    from models.fedrl_nets import build_autoencoder as _build_autoencoder  # type: ignore
    _HAS_AE = True
except Exception:
    _HAS_AE = False
    _build_autoencoder = None


class PPOTrainer:
    """
    Manages one client (env, model, buffer, optimizers).
    Exposes small steps so a multi-client runner can interleave collection and updates.
    """

    def __init__(self, cfg: Any, env: Any, model: Any, device: torch.device, client_id: int = 0, wb_step_base: int = 0):
        self.cfg = cfg
        self.env = env
        self.model = model
        self.device = torch.device(device)
        self.client_id = client_id
        self.wb_step_base = int(wb_step_base)  # absolute step offset for monotonic W&B logging

        # model to device
        self.model.actor.to(self.device)
        self.model.critic.to(self.device)

        # exploration (guard older actors)
        if hasattr(self.model.actor, "set_exploration_eps"):
            self.model.actor.set_exploration_eps(getattr(cfg, "exploration_eps", 0.0))

        # hidden init
        self.hidden_dim = getattr(self.model.actor, "hidden_dim", 0)
        self.h_actor = self.model.actor.init_hidden(self.env.cfg.n_agents, self.device)

        # optimizers
        self.opt_actor, self.opt_critic = setup_optimizers(self.model, cfg)

        # reset and infer shapes
        (actor_obs, critic_obs), _ = self.env.reset(seed=cfg.seed)
        self.n_agents = self.env.cfg.n_agents
        ego0 = ego_list_to_tchw(actor_obs)
        glob0 = to_tchw(critic_obs)

        self.roll = RolloutBuffer(
            n_agents=self.n_agents,
            ego_shape=(ego0.shape[1], ego0.shape[2], ego0.shape[3]),
            glob_shape=(glob0.shape[0], glob0.shape[1], glob0.shape[2]),
            rollout_len=cfg.rollout_len,
            device=str(self.device),
            hidden_dim=self.hidden_dim,
        )

        # stability probe (safe/no-op if unused)
        self._probe_glob: List[torch.Tensor] = []
        self._probe_max = 64
        self._prev_critic_vec: Union[torch.Tensor, None] = None
        self._prev_probe_out: Union[torch.Tensor, None] = None

        # seed a few probe states
        with torch.no_grad():
            for _ in range(min(8, self._probe_max)):
                (a_obs_seed, c_obs_seed), _ = self.env.reset()
                self._probe_glob.append(to_tchw(c_obs_seed).to(self.device))

        self.actor_obs = actor_obs
        self.critic_obs = critic_obs

        self.total_env_steps = 0
        self.ep_returns: list[float] = []
        self.ep_len = 0
        self.ep_return = 0.0
        self.next_eval = getattr(cfg, "eval_every_steps", 0)

        self.is_dist = hasattr(self.model.critic, "taus")

        # per-round distributional prior buffer (fedrl only; gated by cfg)
        self.enable_dist_prior_buffer = bool(getattr(self.cfg, "enable_dist_prior_buffer", False)) and self.is_dist
        self._dist_prior_buffer: Optional[RoundDistBuffer] = None
        if self.enable_dist_prior_buffer:
            alpha_tail = float(getattr(self.cfg, "prior_alpha_tail", getattr(self.cfg, "cvar_alpha", 0.10)))
            self._dist_prior_buffer = RoundDistBuffer(RoundDistBufferCfg(alpha_tail=alpha_tail))


        # ---------- AE auxiliary (optional) ----------
        self.ae_enabled = bool(getattr(cfg, "enable_ae_aux", False)) and _HAS_AE
        self.encoder = None
        self.decoder = None
        self.opt_enc = None
        self.opt_dec = None
        self.ae_rec_lambda = float(getattr(self.cfg, "fedrl_rec_lambda", 1e-2))

        if self.ae_enabled:
            try:
                d_latent = int(getattr(cfg, "fedrl_d_latent", 128))
                ae = _build_autoencoder(in_ch=glob0.shape[0], d_latent=d_latent)  # type: ignore
                self.encoder = ae.encoder.to(self.device)
                self.decoder = ae.decoder.to(self.device)
                self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=float(getattr(cfg, "fedrl_enc_lr", 1e-3)))
                self.opt_dec = torch.optim.Adam(self.decoder.parameters(), lr=float(getattr(cfg, "fedrl_dec_lr", 5e-4)))

                # allow critic to consume spatial encoder if it supports it
                if hasattr(self.model, "attach_spatial_encoder"):
                    self.model.attach_spatial_encoder(self.encoder)
                    self.model.critic.to(self.device)
            except Exception:
                # fallback clean disable
                self.ae_enabled = False
                self.encoder = None
                self.decoder = None
                self.opt_enc = None
                self.opt_dec = None

        self._round_episodes = 0
        self._round_catastrophes = 0

    # ---------- FedAvg / FedTrunc helpers: flat <-> params; also accept state_dicts ----------

    def _normalize_sd_view(
        self, module_prefix: str, sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Accept either full module state_dict or a flat dict with 'module_prefix.' keys;
        return a state_dict keyed relative to the module (no prefix).
        """
        # Case A: keys already relative to the module (e.g., 'weight', 'bias' ...).
        if all(('.' not in k) or (k.split('.')[0] not in ("actor", "critic")) for k in sd.keys()):
            return sd

        # Case B: flatten with 'actor.' / 'critic.' prefix
        out: Dict[str, torch.Tensor] = {}
        prefix = f"{module_prefix}."
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        if not out:
            # If prefix not found, try to load as-is to surface an informative error upstream.
            return sd
        return out

    def get_actor_flat(self, detach: bool = True, cpu: bool = True, **_kwargs) -> torch.Tensor:
        params = [p for p in self.model.actor.parameters() if p.requires_grad]
        if not params:
            return torch.empty(0)
        vec = parameters_to_vector([p.detach() if detach else p for p in params])
        return vec.cpu() if cpu else vec

    def load_actor_flat(
        self,
        flat_or_sd: Union[torch.Tensor, Dict[str, torch.Tensor]],
        strict: bool = True,
        reset_opt: bool = False,
        **_kwargs,
    ) -> None:
        if isinstance(flat_or_sd, dict):
            sd = self._normalize_sd_view("actor", flat_or_sd)
            self.model.actor.load_state_dict(sd, strict=strict)
            self.model.actor.to(self.device)
        else:
            params = [p for p in self.model.actor.parameters() if p.requires_grad]
            if not params:
                return
            expected = parameters_to_vector([p.detach() for p in params]).numel()
            if strict and int(flat_or_sd.numel()) != int(expected):
                raise RuntimeError(f"load_actor_flat: size mismatch {flat_or_sd.numel()} != {expected}")
            vector_to_parameters(flat_or_sd.to(params[0].device, dtype=params[0].dtype), params)

        if reset_opt and getattr(self, "opt_actor", None) is not None:
            try:
                self.opt_actor.state.clear()
            except Exception:
                self.opt_actor = type(self.opt_actor)(self.model.actor.parameters(), **self.opt_actor.defaults)

    def get_critic_flat(self, detach: bool = True, cpu: bool = True, **_kwargs) -> torch.Tensor:
        params = [p for p in self.model.critic.parameters() if p.requires_grad]
        if not params:
            return torch.empty(0)
        vec = parameters_to_vector([p.detach() if detach else p for p in params])
        return vec.cpu() if cpu else vec

    def load_critic_flat(
        self,
        flat_or_sd: Union[torch.Tensor, Dict[str, torch.Tensor]],
        strict: bool = True,
        reset_opt: bool = False,
        **_kwargs,
    ) -> None:
        if isinstance(flat_or_sd, dict):
            sd = self._normalize_sd_view("critic", flat_or_sd)
            self.model.critic.load_state_dict(sd, strict=strict)
            self.model.critic.to(self.device)
        else:
            params = [p for p in self.model.critic.parameters() if p.requires_grad]
            if not params:
                return
            expected = parameters_to_vector([p.detach() for p in params]).numel()
            if strict and int(flat_or_sd.numel()) != int(expected):
                raise RuntimeError(f"load_critic_flat: size mismatch {flat_or_sd.numel()} != {expected}")
            vector_to_parameters(flat_or_sd.to(params[0].device, dtype=params[0].dtype), params)

        if reset_opt and getattr(self, "opt_critic", None) is not None:
            try:
                self.opt_critic.state.clear()
            except Exception:
                self.opt_critic = type(self.opt_critic)(self.model.critic.parameters(), **self.opt_critic.defaults)

    def _compat_encode(self, critic, xs):
        # New critics expose _encode; legacy ones expose encoder(...)
        if hasattr(critic, "_encode"):
            return critic._encode(xs)
        if hasattr(critic, "encoder"):
            with torch.no_grad():
                return critic.encoder(xs)
        return None

    def _compat_squash(self, critic, y):
        # Distributional critics have _squash; expected critics don’t
        if hasattr(critic, "_squash"):
            return critic._squash(y)
        return y

    def _compat_prior_forward(self, critic, z, xs):
        # Prior-only forward on frozen heads if present; returns None if not wired
        if hasattr(critic, "v_prior"):
            if z is None: z = self._compat_encode(critic, xs)
            return critic.v_prior(z).squeeze(-1)
        if hasattr(critic, "head_prior"):
            if z is None: z = self._compat_encode(critic, xs)
            return self._compat_squash(critic, critic.head_prior(z))
        return None

    def _compat_raw_local_forward(self, critic, z, xs):
        # Local head bypass (no affine+clamp); returns None if we can’t bypass
        if hasattr(critic, "v"):
            if z is None: z = self._compat_encode(critic, xs)
            return critic.v(z).squeeze(-1)
        if hasattr(critic, "head"):
            if z is None: z = self._compat_encode(critic, xs)
            return self._compat_squash(critic, critic.head(z))
        return None

    def get_round_metrics(self, reset: bool = True) -> Dict[str, float]:
        eps = float(self._round_episodes)
        cats = float(self._round_catastrophes)
        hazard_rate = (cats / eps) if eps > 0.0 else 0.0
        out = {
            "episodes": eps,
            "catastrophes": cats,
            "hazard_rate": hazard_rate,
        }
        if reset:
            self._round_episodes = 0
            self._round_catastrophes = 0
        return out
    
    # ---------- distributional prior buffer api (one client, one round) ----------
    def reset_dist_prior_buffer(self) -> None:
        """
        Reset the per-round distributional buffer.

        Call this at the start of a communication round when using FedRL
        with distributional priors. No-op if the buffer is disabled.
        """
        if self._dist_prior_buffer is not None:
            self._dist_prior_buffer.reset()

    def accumulate_dist_prior_batch(self, z_batch: torch.Tensor, alpha_tail: Optional[float] = None) -> None:
        """
        Add a batch of critic quantile outputs to the per-round buffer.

        Args:
          z_batch: tensor [B, Nq] with critic quantiles for visited states.
          alpha_tail: optional override for tail fraction used in cvar.
        """
        if self._dist_prior_buffer is None:
            return
        self._dist_prior_buffer.add_batch(z_batch, alpha_tail=alpha_tail)

    def build_round_distributional_prior(
        self,
        use_cvar_weight: bool = True,
        cvar_scale: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """
        Build the per-client, per-round barycentric prior over critic quantiles.

        Returns:
          q_prior: tensor [Nq] on cpu, or None if no data was accumulated
                   or the buffer is disabled.
        """
        if self._dist_prior_buffer is None:
            return None
        return self._dist_prior_buffer.build_prior(
            use_cvar_weight=use_cvar_weight,
            cvar_scale=float(cvar_scale),
        )


    # --- absolute W&B step helper (monotonic across resumes) ---
    def _wb_step(self) -> int:
        return int(self.wb_step_base + self.total_env_steps)

    # --- train for a fixed number of epochs, honoring cfg.total_steps as early stop
    def train_for_epochs(self, n_epochs: int) -> int:
        steps_start = self.total_env_steps
        for _ in range(int(n_epochs)):
            while not self.roll.full():
                self.collect_rollout_step()
                if self.total_env_steps >= self.cfg.total_steps:
                    break
            if self.roll.ptr > 0:
                self.update_epoch()
            self.maybe_eval()
            if self.total_env_steps >= self.cfg.total_steps:
                break
        return int(self.total_env_steps - steps_start)

    # ---------- collection ----------

    def _forward_policy(self, ego_bna: torch.Tensor, agent_ids: torch.Tensor, h_in: torch.Tensor):
        with torch.no_grad():
            dists, h_out = self.model.actor(ego_bna, agent_ids, h_in=h_in)
            actions = dists.sample()
            logprobs = dists.log_prob(actions)
        return actions, logprobs, h_out

    def _forward_value(self, glob_b: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return centralized_value_mean(self.model, glob_b)

    def collect_rollout_step(self) -> bool:
        """
        Single env step; push into buffer.
        Returns True if buffer is now full.
        """
        ego_bna = ego_list_to_tchw(self.actor_obs).to(self.device)
        glob_b = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
        agent_ids = torch.arange(self.n_agents, dtype=torch.long, device=self.device)

        h_in = self.h_actor.detach()

        actions, logprobs, h_out = self._forward_policy(ego_bna, agent_ids, h_in=h_in)
        v = self._forward_value(glob_b)

        act_dict = {i: int(actions[i].item()) for i in range(self.n_agents)}
        (actor_obs_next, critic_obs_next), team_rew, terminated, truncated, info = self.env.step(act_dict)
        self.h_actor = h_out.detach()

        self.roll.add(
            ego_bna=ego_bna,
            agent_ids=agent_ids,
            glob_b=glob_b.squeeze(0),
            act_bna=actions,
            logp_bna=logprobs,
            v_b=v.detach(),
            rew_b=torch.tensor(team_rew, dtype=torch.float32, device=self.device),
            done_b=torch.tensor(float(terminated or truncated), device=self.device),
            h_actor_in=h_in,
        )

        self.total_env_steps += 1
        self.ep_len += 1
        self.ep_return += float(team_rew)

        self.actor_obs, self.critic_obs = actor_obs_next, critic_obs_next

        if terminated or truncated:
            term_reason = info.get("terminated_by", "unknown")

            # --- add these two lines right here ---
            self._round_episodes += 1
            if term_reason == "catastrophe":
                self._round_catastrophes += 1

            if wandb.run is not None:
                wandb.log({
                    f"client{self.client_id}/ep/len": self.ep_len,
                    f"client{self.client_id}/ep/return_team": self.ep_return,
                    f"client{self.client_id}/ep/objects_delivered_total": int(info.get("objects_delivered_total", 0)),
                    f"client{self.client_id}/ep/term/time_limit": 1.0 if term_reason == "time_limit" else 0.0,
                    f"client{self.client_id}/ep/term/catastrophe": 1.0 if term_reason == "catastrophe" else 0.0,
                    f"client{self.client_id}/ep/term/full_success": 1.0 if term_reason == "all_goals_and_objects" else 0.0,
                }, step=self._wb_step())

            self.ep_returns.append(self.ep_return)
            self.h_actor = self.model.actor.init_hidden(self.n_agents, self.device)
            (self.actor_obs, self.critic_obs), _ = self.env.reset()
            self.ep_return = 0.0
            self.ep_len = 0

        # opportunistically top up probe buffer with latest critic observation
        if len(self._probe_glob) < self._probe_max:
            self._probe_glob.append(self.roll.glob[self.roll.ptr - 1].detach().to(self.device))

        return self.roll.full()

    # ---------- value bootstrap, adv, ppo update + optional AE auxiliary ----------

    def _bootstrap_last_value(self) -> float:
        glob_last = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
        last_v = self._forward_value(glob_last)
        return float(last_v.item())

    def _compute_cvar_advantage(self, alpha: float) -> Dict[str, torch.Tensor]:
        """
        Build a lower-tail (CVaR-like) value sequence from the distributional critic,
        then run GAE on that sequence to obtain a tail-sensitive advantage.

        Returns a dict possibly containing:
          adv_cvar: [T] (standardized)
          v_cvar_mb: scalar (mean CVaR value)
          mean_minus_cvar_mb: scalar (gap)
        """
        out: Dict[str, torch.Tensor] = {}
        T = self.roll.ptr
        if T == 0:
            return out

        glob_batch = self.roll.glob[:T].to(self.device)

        with torch.no_grad():
            z = self.model.critic(glob_batch)            # [T, Nq]
            Nq = z.size(-1)
            k = max(1, int(alpha * Nq))
            v_mean_seq = z.mean(dim=-1)                  # [T]
            v_cvar_seq = z[:, :k].mean(dim=-1)           # [T]

            # CVaR bootstrap at the last state
            glob_last = to_tchw(self.critic_obs).unsqueeze(0).to(self.device)
            z_last = self.model.critic(glob_last)        # [1, Nq]
            last_v_cvar = z_last[:, :k].mean(dim=-1).item()

        rew_b  = self.roll.rewards[:T].to(self.device)   # [T]
        done_b = self.roll.dones[:T].to(self.device)     # [T]

        # GAE on the tail value sequence (no mixing here)
        adv_tail = torch.zeros_like(v_cvar_seq)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = last_v_cvar if t == T - 1 else v_cvar_seq[t + 1]
            delta = rew_b[t] + self.cfg.gamma * next_v * (1.0 - done_b[t]) - v_cvar_seq[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - done_b[t]) * gae
            adv_tail[t] = gae
        # standardize tail advantage on its own statistics
        adv_tail = (adv_tail - adv_tail.mean()) / (adv_tail.std(unbiased=False) + 1e-8)

        out["adv_cvar"] = adv_tail.detach()
        out["v_cvar_mb"] = v_cvar_seq.mean().detach()
        out["mean_minus_cvar_mb"] = (v_mean_seq - v_cvar_seq).mean().detach()
        return out

    def _aux_actor_cvar_pass(
        self,
        adv_cvar: torch.Tensor,
        ego_flat: torch.Tensor,
        agent_ids_flat: torch.Tensor,
        h_flat: torch.Tensor,
        act_flat: torch.Tensor,
        old_logp_flat: torch.Tensor,
    ) -> Dict[str, float]:
        """
        A second, lightweight actor pass: maximizes CVaR advantage with an UNCLIPPED surrogate.
        Uses ratio.detach() to decouple from PPO clip and preserve stability.

        Returns logging metrics.
        """
        cfg = self.cfg
        T = self.roll.ptr
        n_agents = self.roll.ego.shape[1]

        lambda_cvar = float(getattr(cfg, "lambda_cvar", 0.0))
        tau_cvar = float(getattr(cfg, "tau_cvar", 1.0))
        if lambda_cvar <= 0.0:
            return {}

        # optional temperature scaling to keep gradients well-conditioned
        adv_cvar_scaled = adv_cvar / max(tau_cvar, 1e-6)

        # minibatch schedule (local; mirrors helpers.make_minibatches)
        n_mb = max(int(getattr(cfg, "minibatches", 1)), 1)
        mb_size = T // n_mb if n_mb > 0 else T
        if mb_size == 0:
            mb_slices = [np.arange(T)]
        else:
            idxs = np.arange(T)
            np.random.shuffle(idxs)
            mb_slices = [idxs[s: min(s + mb_size, T)] for s in range(0, T, mb_size)]

        total_aux = 0.0
        count = 0

        for mb in mb_slices:
            # expand to per-agent mask over flattened [T*n_agents]
            mask_idx = []
            for t in mb:
                base = int(t) * n_agents
                mask_idx.extend(range(base, base + n_agents))
            mask = torch.as_tensor(mask_idx, dtype=torch.long, device=self.device)

            dists_new, _ = self.model.actor(
                ego_flat[mask].to(self.device),
                agent_ids_flat[mask].to(self.device),
                h_in=h_flat[mask].to(self.device),
            )
            new_logp = dists_new.log_prob(act_flat[mask].to(self.device))
            ratio = torch.exp(new_logp - old_logp_flat[mask])

            # broadcast per-time-step tail advantage to per-agent
            adv_mb = adv_cvar_scaled[mb].to(self.device)
            adv_mb = adv_mb.repeat_interleave(n_agents)

            # UNCLIPPED auxiliary (ratio is detached to avoid tug-of-war with PPO clip)
            aux_loss = - lambda_cvar * (ratio * adv_mb).mean()

            self.opt_actor.zero_grad(set_to_none=True)
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), cfg.max_grad_norm)
            self.opt_actor.step()

            total_aux += float(aux_loss.detach().item())
            count += 1

        if count == 0:
            return {}
        return {"loss/aux_cvar": total_aux / max(count, 1)}

    def _ae_aux_step(self, glob_batch: torch.Tensor) -> float:
        """
        Trains the auxiliary AE only. Does not affect actor/critic gradients.
        No-op (returns 0.0) if AE is disabled.
        glob_batch: [T, C, H, W]
        """
        if not self.ae_enabled or self.encoder is None or self.decoder is None:
            return 0.0
        if glob_batch.numel() == 0:
            return 0.0

        mb = min(256, glob_batch.size(0))
        idx = torch.randint(0, glob_batch.size(0), (mb,), device=glob_batch.device)
        x = glob_batch[idx].to(self.device)          # [mb, C, H, W]
        z = self.encoder(x)                          # [mb, spatial latent]
        x_hat = self.decoder(z)                      # [mb, C, H, W]
        rec_loss = torch.nn.functional.mse_loss(x_hat, x)

        assert self.opt_enc is not None and self.opt_dec is not None
        self.opt_enc.zero_grad(set_to_none=True)
        self.opt_dec.zero_grad(set_to_none=True)
        rec_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        self.opt_enc.step()
        self.opt_dec.step()

        return float(rec_loss.item())

    def update_epoch(self) -> Dict[str, float]:
        # ---------- 1) Reward-only GAE for PPO (stable baseline) ----------
        last_v_mean = self._bootstrap_last_value()
        adv_mean, ret_mean = compute_gae(
            self.roll, last_value=last_v_mean,
            gamma=self.cfg.gamma, lam=self.cfg.gae_lambda, device=str(self.device)
        )
        adv_mean = (adv_mean - adv_mean.mean()) / (adv_mean.std(unbiased=False) + 1e-8)

        # ---------- 2) Optional CVaR advantage (separate path; no mixing with PPO adv) ----------
        use_risk = bool(getattr(self.cfg, "risk_enable", False)) and self.is_dist
        beta = float(getattr(self.cfg, "risk_beta", 0.0))     # used only as a gate here
        alpha = float(getattr(self.cfg, "cvar_alpha", 0.10))
        adv_cvar = None
        v_cvar_mb = None
        mean_minus_cvar_mb = None

        if use_risk and beta > 0.0:
            cvar_pack = self._compute_cvar_advantage(alpha)
            adv_cvar = cvar_pack.get("adv_cvar", None)
            v_cvar_mb = cvar_pack.get("v_cvar_mb", None)
            mean_minus_cvar_mb = cvar_pack.get("mean_minus_cvar_mb", None)

        # ---------- 3) Prepare tensors for both stages ----------
        T = self.roll.ptr
        h_flat = self.roll.h_actor[:T].reshape(T * self.n_agents, self.hidden_dim)
        logp_flat = self.roll.logprobs[:T].reshape(T * self.n_agents)
        act_flat = self.roll.actions[:T].reshape(T * self.n_agents)
        agent_ids_flat = self.roll.agent_ids[:T].reshape(T * self.n_agents)
        ego_flat = self.roll.ego[:T].reshape(T * self.n_agents, *self.roll.ego.shape[2:])
        old_logp_flat = logp_flat.detach()
        glob_batch = self.roll.glob[:T]
        values_old = self.roll.values[:T].detach()

        # per-round distributional prior accumulation (fedrl only)
        z_all: Optional[torch.Tensor] = None
        if self.enable_dist_prior_buffer and self.is_dist and T > 0:
            with torch.no_grad():
                z_all = self.model.critic(glob_batch.to(self.device))  # [T, Nq]
            # use the same tail level as cvar_alpha by default
            self.accumulate_dist_prior_batch(z_all, alpha_tail=alpha)


        # ---------- 4) Baseline PPO update (reward-only) ----------
        metrics = ppo_epoch_update(
            cfg=self.cfg,
            model=self.model,
            opt_actor=self.opt_actor,
            opt_critic=self.opt_critic,
            roll=self.roll,
            adv=adv_mean.detach(),              # actor sees reward-only GAE
            ret=ret_mean,                      # critic target remains mean-return
            ego_flat=ego_flat,
            agent_ids_flat=agent_ids_flat,
            h_flat=h_flat,
            act_flat=act_flat,
            old_logp_flat=old_logp_flat,
            glob_batch=glob_batch,
            values_old=values_old,
            device=str(self.device),
            quantile_huber_loss=quantile_huber_loss,
        )

        # ---------- 5) Unclipped auxiliary actor pass on CVaR advantage ----------
        if adv_cvar is not None and float(getattr(self.cfg, "lambda_cvar", 0.0)) > 0.0:
            aux_metrics = self._aux_actor_cvar_pass(
                adv_cvar=adv_cvar,
                ego_flat=ego_flat,
                agent_ids_flat=agent_ids_flat,
                h_flat=h_flat,
                act_flat=act_flat,
                old_logp_flat=old_logp_flat,
            )
            metrics.update(aux_metrics)

        # ---------- 6) Optional logging ----------
        if wandb.run is not None:
            metrics[f"client{self.client_id}/risk/beta"] = beta
            metrics[f"client{self.client_id}/risk/alpha"] = alpha
            if v_cvar_mb is not None:
                metrics[f"client{self.client_id}/critic/v_cvar_mb"] = float(v_cvar_mb.item())
            if mean_minus_cvar_mb is not None:
                metrics[f"client{self.client_id}/critic/mean_minus_cvar_mb"] = float(mean_minus_cvar_mb.item())

        # ---------- 7) AE auxiliary (unchanged) ----------
        rec_loss = self._ae_aux_step(glob_batch)
        if self.ae_enabled and wandb.run is not None:
            metrics[f"client{self.client_id}/ae/rec_mse"] = float(rec_loss)

        # ---------- 8) Quick distributional debug ----------
        if self.is_dist:
            with torch.no_grad():
                if z_all is not None:
                    z_dbg = z_all[: min(64, z_all.size(0))].to(self.device)
                else:
                    z_dbg = self.model.critic(glob_batch[: min(64, T)].to(self.device))
                v_mean_dbg = z_dbg.mean(dim=-1).mean().item()
            metrics[f"client{self.client_id}/critic/v_mean_mb"] = v_mean_dbg


        metrics.update({
            f"client{self.client_id}/optim/lr_actor": self.opt_actor.param_groups[0]["lr"],
            f"client{self.client_id}/optim/lr_critic": self.opt_critic.param_groups[0]["lr"],
            f"client{self.client_id}/env/steps": self.total_env_steps,
        })

        if wandb.run is not None:
            wandb.log(metrics, step=self._wb_step())

        if self.total_env_steps % getattr(self.cfg, "log_interval", 1000) == 0:
            mean_ret = float(np.mean(self.ep_returns[-10:])) if self.ep_returns else 0.0
            if wandb.run is not None:
                wandb.log({
                    f"client{self.client_id}/train/mean_ep_ret_10": mean_ret,
                    f"client{self.client_id}/train/last_v": float(last_v_mean),
                    f"client{self.client_id}/env/episodes": len(self.ep_returns),
                }, step=self._wb_step())
                if self.is_dist:
                    glob_dbg = glob_batch[: min(128, glob_batch.size(0))]
                    log_distributional_visuals_wandb(
                        glob_batch=glob_dbg,
                        model=self.model,
                        device=str(self.device),
                        wb_step=self._wb_step(),
                        split=f"train_client{self.client_id}",
                        max_batch=128
                    )

        self.roll.clear()

        # --- stability metrics (cheap, W&B-guarded) ---
        try:
            # 1) parameter drift (critic)
            vecs = [p.detach().flatten().cpu() for p in self.model.critic.parameters() if p.requires_grad]
            cur_vec = torch.cat(vecs) if vecs else torch.empty(0)
            if self._prev_critic_vec is not None and cur_vec.numel() == self._prev_critic_vec.numel():
                param_drift = torch.norm(cur_vec - self._prev_critic_vec).item()
            else:
                param_drift = 0.0
            self._prev_critic_vec = cur_vec

            # 2–4) probe-based metrics
            out_drift = prior_gap = clamp_effect = 0.0
            if len(self._probe_glob) > 0:
                xs_probe = torch.stack(self._probe_glob, dim=0).to(self.device)
                c = self.model.critic

                with torch.no_grad():
                    y_con = c(xs_probe).detach()
                    z = self._compat_encode(c, xs_probe)

                    y_pri = self._compat_prior_forward(c, z, xs_probe)
                    if y_pri is None:
                        y_pri = torch.zeros_like(y_con, device=self.device)

                    y_raw = self._compat_raw_local_forward(c, z, xs_probe)
                    if y_raw is None:
                        y_raw = y_con

                    y_con_cpu = y_con.cpu()
                    y_pri_cpu = y_pri.cpu()
                    y_raw_cpu = y_raw.cpu()

                def mad(a, b):
                    return (a - b).abs().mean().item()

                if self._prev_probe_out is not None and self._prev_probe_out.shape == y_con_cpu.shape:
                    out_drift = mad(y_con_cpu, self._prev_probe_out)
                self._prev_probe_out = y_con_cpu

                prior_gap = mad(y_con_cpu, y_pri_cpu)
                clamp_effect = mad(y_con_cpu, y_raw_cpu)

            if wandb.run is not None:
                wandb.log({
                    f"client{self.client_id}/stab/param_drift_critic": float(param_drift),
                    f"client{self.client_id}/stab/out_drift_probe": float(out_drift),
                    f"client{self.client_id}/stab/prior_gap_probe": float(prior_gap),
                    f"client{self.client_id}/stab/clamp_effect_probe": float(clamp_effect),
                }, step=self._wb_step())

        except Exception as _e:
            print(f"[Client {self.client_id}] Stability probe failed: {_e}")

        return metrics


    def maybe_eval(self) -> None:
        if self.next_eval and self.total_env_steps >= self.next_eval:
            eval_gif = f"outputs/eval/client_{self.client_id}_seed_{self.cfg.seed}_step_{self.total_env_steps}.gif"
            _ = run_eval_rollout(
                self.env, self.model, self.device,
                deterministic=True,
                record=True,
                log_wandb=True,
                gif_path=eval_gif,
                wb_step=self._wb_step(),
            )
            self.next_eval += getattr(self.cfg, "eval_every_steps", 0)
