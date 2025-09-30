import torch
from typing import Tuple


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