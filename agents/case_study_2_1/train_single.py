from agents.case_study_2_1.ppo_utils.trainer import PPOTrainer
from utils.general import normalize_weights
import math

def _compute_client_epochs(local_epochs_per_round: int, n_clients: int, weights):
    return max(1, math.ceil(local_epochs_per_round * (weights[0] * n_clients)))

def train(cfg, env, model, device):
    tr = PPOTrainer(cfg, env, model, device)

    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    epochs_per_round = _compute_client_epochs(cfg.local_epochs_per_round, cfg.n_clients, w)

    for _ in range(int(cfg.num_communication_rounds)):
        for _ in range(epochs_per_round):
            # one PPO "epoch" == collect a rollout buffer then update once
            while not tr.roll.full():
                tr.collect_rollout_step()
                if tr.total_env_steps >= cfg.total_steps:
                    break
            if tr.roll.ptr > 0:
                tr.update_epoch()
            tr.maybe_eval()
            if tr.total_env_steps >= cfg.total_steps:
                return tr.model
        return tr.model

    return tr.model
