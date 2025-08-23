from typing import Dict
from configs.config_templates.case_study_1_1 import Config

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from models.c51_bandit import C51Bandit, c51_project
from envs.case_study_1_1.bandits import HeteroBandit
from utils.general import normalize_weights, normalize_hist

from eval.tabular import compute_metrics

from federation.case_study_1_1.fedrl import FeDRLServer


def train_fedrl(env: HeteroBandit, cfg: Config, summary: Dict):
    w = normalize_weights(cfg.client_weights, cfg.n_clients) 
    client_models = [C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device) for _ in range(env.n_clients)]
    z_c51 = torch.linspace(-4, 4, client_models[0].n_atoms).to(cfg.device)
    z_np = z_c51.cpu().numpy()
    global_model = C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device)
    fedrl_server = FeDRLServer(global_model=global_model, cfg=cfg, z=z_c51)

    client_optimizers = [AdamW(m.parameters(), lr=cfg.lr) for m in client_models]

    init_priors = None
    # init_priors = [[get_random_prior(client_id, arm_id, z_c51) for arm_id in range(env.n_arms)] for client_id in range(env.n_clients)]

    for round in range(cfg.num_communication_rounds):
        round_ce = [] 
        round_kl = []
        for client_id in range(env.n_clients):
            model = client_models[client_id]
            optimizer = client_optimizers[client_id]
            model.train() 

            batch_size = max(1, int(np.ceil(cfg.batch_size * (w[client_id] * cfg.n_clients))))

            client_losses = [] 
            for local_epoch in range(cfg.local_epochs_per_round):

                for arm_id in range(env.n_arms):
                    optimizer.zero_grad()
                    rewards_tensor = torch.tensor(env.sample(client_id, arm_id, batch_size), dtype=torch.float32).to(cfg.device)

                    # Forward pass
                    predicted = model(torch.tensor([[arm_id]]).repeat(batch_size, 1).to(cfg.device))
                    target_distribution = c51_project(rewards_tensor, z_c51)

                    # Calculate loss
                    ce = F.kl_div(predicted.log(), target_distribution, reduction='batchmean')

                    base = cfg.kappa                        # e.g., start with 2.0–5.0
                    wi   = max(w[client_id], 1e-3)         # w is your normalized client volume
                    kappa_scaled = base / (wi ** 1.0)    # alpha in [1, 1.5]; try alpha=1.0 first
                    kappa_scaled = min(kappa_scaled, 20.0)  # e.g., kappa_max = 20.0

                    if init_priors != None:
                        kl = F.kl_div(predicted.log(), init_priors[client_id][arm_id].to(cfg.device), reduction='batchmean')
                        loss = ce + kappa_scaled * kl
                    else:
                        with torch.no_grad():
                            prior = fedrl_server.global_model(torch.tensor([[arm_id]]).repeat(batch_size, 1).to(cfg.device))

                        kl = F.kl_div(predicted.log(), prior, reduction='batchmean')
                        loss = ce + kappa_scaled * kl

                    client_losses.append(loss.item())        

                    # Backward pass and optimize
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                    round_ce.append(ce.item())
                    round_kl.append(kl.item())

        fedrl_server.aggregate(client_models)
        _, _ = fedrl_server.distill(client_models)

        if round >= 5: 
            init_priors = None

        # ----- Compute metrics averaged across clients -----
        all_metrics = []
        for client_id in range(env.n_clients):
            pred_hist = []
            truth_hist = []
            for arm_id in range(env.n_arms):
                with torch.no_grad():
                    p = model = client_models[client_id](
                        torch.tensor([arm_id]).to(cfg.device).unsqueeze(0)
                    )[0].cpu().numpy()
                pred_hist.append(normalize_hist(p))

                truth = env.estimate_truth_hist(z_np, client_id, arm_id, nsamp=20000)
                truth_hist.append(truth)

            pred_hist = np.stack(pred_hist, axis=0)
            truth_hist = np.stack(truth_hist, axis=0)

            m = compute_metrics(pred_hist, truth_hist, z_np, cvar_alpha=cfg.cvar_alpha)
            all_metrics.append(m)

        # Average across clients
        avg_metrics = {
            k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()
        }

        summary['fedrl']['losses'].append(float(np.mean(round_ce)))
        summary['fedrl']['kl_losses'].append(float(np.mean(round_kl)))
        for k, v in avg_metrics.items():
            summary['fedrl']['metrics'][k].append(v)
        
    return summary