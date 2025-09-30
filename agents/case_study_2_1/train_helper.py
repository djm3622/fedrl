import torch
from typing import Tuple, List
import numpy as np
from torch import optim, nn

class RolloutBuffer:
    def __init__(
        self, n_agents: int, ego_shape: Tuple[int, int, int], 
        glob_shape: Tuple[int, int, int],
        rollout_len: int, device: str, hidden_dim: int = 128
    ):
        self.n_agents = n_agents
        self.rollout_len = rollout_len
        self.device = device

        B = rollout_len
        C_ego, H_ego, W_ego = ego_shape
        C_glb, H_glb, W_glb = glob_shape

        self.ego = torch.zeros(B, n_agents, C_ego, H_ego, W_ego, device=device)
        self.agent_ids = torch.zeros(B, n_agents, dtype=torch.long, device=device)
        self.glob = torch.zeros(B, C_glb, H_glb, W_glb, device=device)
        self.actions = torch.zeros(B, n_agents, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(B, n_agents, device=device)
        self.values = torch.zeros(B, device=device)
        self.rewards = torch.zeros(B, device=device)   
        self.dones = torch.zeros(B, device=device)
        self.h_actor = torch.zeros(B, n_agents, hidden_dim, device=device)

        self.ptr = 0

    def add(self, ego_bna, agent_ids, glob_b, act_bna, logp_bna, v_b, rew_b, done_b, h_actor_in):
        t = self.ptr
        self.ego[t] = ego_bna
        self.agent_ids[t] = agent_ids
        self.glob[t] = glob_b
        self.actions[t] = act_bna
        self.logprobs[t] = logp_bna
        self.values[t] = v_b
        self.rewards[t] = rew_b
        self.dones[t] = done_b
        self.h_actor[t] = h_actor_in
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.rollout_len

    def clear(self):
        self.ptr = 0


def compute_gae(roll: RolloutBuffer, last_value: float, gamma: float, lam: float, device: str):
    T = roll.ptr
    adv = torch.zeros(T, device=device)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - roll.dones[t]
        next_value = last_value if t == T - 1 else roll.values[t + 1]
        delta = roll.rewards[t] + gamma * next_value * nonterminal - roll.values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + roll.values[:T]
    return adv, ret


def to_tchw(x_hw_c: np.ndarray) -> torch.Tensor:
    # input HxWxC float32 -> torch [C,H,W]
    assert x_hw_c.ndim == 3
    x = torch.from_numpy(x_hw_c).permute(2, 0, 1).contiguous().float()
    return x


def ego_list_to_tchw(ego_list: List[np.ndarray]) -> torch.Tensor:
    # list of kxkxC -> [B, C, k, k]
    arr = np.stack(ego_list, axis=0)  # [B, k, k, C]
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float()
    return x


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    u = target.unsqueeze(1) - pred.unsqueeze(2)      # [B, N, M]
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * (u ** 2), kappa * (abs_u - 0.5 * kappa))
    tau = taus.view(1, -1, 1)                        # [1, N, 1]
    weight = torch.abs(tau - (u < 0).float())        # |tau - 1_{u<0}|
    return (weight * huber).mean()


def setup_optimizers(model, cfg):
    lr_actor, lr_critic = cfg.lr_actor, cfg.lr_critic
    opt_actor = optim.Adam(model.actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(
        model.critic.parameters(),
        lr=lr_critic,
        weight_decay=getattr(cfg, "critic_weight_decay", 0.0),
    )
    return opt_actor, opt_critic


@torch.no_grad()
def centralized_value_mean(model, glob_b):
    return model.critic.mean_value(glob_b).squeeze(0)


def make_minibatches(T, n_mb):
    idxs = np.arange(T)
    mb_size = T // max(n_mb, 1)
    if mb_size == 0:
        return [idxs]
    out = []
    np.random.shuffle(idxs)
    for s in range(0, T, mb_size):
        e = min(s + mb_size, T)
        if e > s:
            out.append(idxs[s:e])
    return out


def build_actor_masks(batch_indices, n_agents):
    # expand time indices into per-agent indices on flattened [T*n_agents]
    masks = []
    for mb in batch_indices:
        mask = []
        for t in mb:
            base = int(t) * n_agents
            mask.extend(range(base, base + n_agents))
        masks.append(torch.as_tensor(mask, dtype=torch.long))
    return masks


def critic_update_expected(cfg, model, opt_critic, glob_batch, ret_batch, values_old, mb, device):
    v_pred = model.critic(glob_batch[mb].to(device))  # [mb]
    target = ret_batch[mb].to(device)
    v_loss_unclipped = (v_pred - target) ** 2
    v_clipped = values_old[mb].to(device) + torch.clamp(
        v_pred - values_old[mb].to(device), -cfg.clip_eps, cfg.clip_eps
    )
    v_loss_clipped = (v_clipped - target) ** 2
    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

    opt_critic.zero_grad(set_to_none=True)
    v_loss.backward()
    nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
    opt_critic.step()
    return v_loss.detach().item()


def critic_update_distributional(cfg, model, opt_critic, roll, glob_batch, mb, device, quantile_huber_loss):
    # current quantiles
    z_pred = model.critic(glob_batch[mb].to(device))  # [mb, N]

    # bootstrap one step ahead on time indices clipped to T-1
    T = glob_batch.size(0)
    mb_np = np.asarray(mb, dtype=np.int64)
    mb_next = np.clip(mb_np + 1, 0, T - 1)
    glob_next_mb = glob_batch[mb_next].to(device)

    with torch.no_grad():
        z_next = model.critic(glob_next_mb)  # [mb, N]
        rew_mb = roll.rewards[:T][mb].to(device)       # [mb]
        done_mb = roll.dones[:T][mb].to(device)        # [mb]
        z_tgt = rew_mb.unsqueeze(-1) + cfg.gamma * (1.0 - done_mb.unsqueeze(-1)) * z_next  # [mb, N]

    v_loss = quantile_huber_loss(z_pred, z_tgt, model.critic.taus, kappa=getattr(cfg, "qr_kappa", 1.0))

    opt_critic.zero_grad(set_to_none=True)
    v_loss.backward()
    nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
    opt_critic.step()
    return v_loss.detach().item()



def actor_update_minibatch(
    cfg, model, opt_actor, ego_flat, agent_ids_flat, h_flat, act_flat,
    old_logp_flat, adv_batch, n_agents, mask, device
):
    dists_new, _ = model.actor(
        ego_flat[mask].to(device),
        agent_ids_flat[mask].to(device),
        h_in=h_flat[mask].to(device),
    )
    new_logp = dists_new.log_prob(act_flat[mask].to(device))
    ratio = torch.exp(new_logp - old_logp_flat[mask])
    adv_per_agent = adv_batch.to(device).repeat_interleave(n_agents)

    pg1 = ratio * adv_per_agent
    pg2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_per_agent
    policy_loss = -torch.min(pg1, pg2).mean()
    entropy = dists_new.entropy().mean()
    loss = policy_loss - cfg.ent_coef * entropy

    opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.actor.parameters(), cfg.max_grad_norm)
    opt_actor.step()

    return policy_loss.detach().item(), entropy.detach().item()


def ppo_epoch_update(
    cfg, model, opt_actor, opt_critic, roll, adv, ret,
    ego_flat, agent_ids_flat, h_flat,
    act_flat, old_logp_flat, glob_batch,
    values_old, device, quantile_huber_loss
):
    T = roll.ptr
    n_agents = roll.ego.shape[1]
    total_v_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0
    n_mb = 0

    # expected vs distributional
    is_dist = hasattr(model.critic, "taus")

    # minibatches over time steps
    for _ in range(cfg.update_epochs):
        batches = make_minibatches(T, cfg.minibatches)
        masks = build_actor_masks(batches, n_agents)
        for mb, mask in zip(batches, masks):
            if is_dist:
                v_item = critic_update_distributional(
                    cfg, model, opt_critic, roll, glob_batch, mb, device, quantile_huber_loss
                )
            else:
                v_item = critic_update_expected(
                    cfg, model, opt_critic, glob_batch, ret.detach(), values_old, mb, device
                )

            adv_mb = adv.detach()[mb]
            pol_item, ent_item = actor_update_minibatch(
                cfg, model, opt_actor, ego_flat, agent_ids_flat, h_flat,
                act_flat, old_logp_flat, adv_mb, n_agents, mask.to(device), device
            )
            total_v_loss += v_item
            total_policy_loss += pol_item
            total_entropy += ent_item
            n_mb += 1

    div = max(n_mb, 1)
    return {
        "loss/value": total_v_loss / div,
        "loss/policy": total_policy_loss / div,
        "policy/entropy": total_entropy / div,
        "rollout/len": T,
    }