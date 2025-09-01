from typing import Dict
from configs.config_templates.case_study_1_1 import Config

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from models.c51_bandit import C51Bandit, c51_project
from envs.case_study_1_1.bandits import HeteroBandit
from utils.general import normalize_weights, normalize_hist

from eval.tabular import compute_metrics

from federation.case_study_1_1.fedtrunc import FedTruncServer
from eval.distributions import save_round_plots, compile_gifs


def train_fedtrunc(env: HeteroBandit, cfg: Config, summary: Dict):
    w = normalize_weights(cfg.client_weights, cfg.n_clients) 
    client_models = [C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device) for _ in range(env.n_clients)]
    z_c51 = torch.linspace(-4, 4, client_models[0].n_atoms).to(cfg.device)
    z_np = z_c51.cpu().numpy()
    global_model = C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device)
    fedtrunc_server = FedTruncServer(global_model=global_model, cfg=cfg)

    client_optimizers = [AdamW(m.parameters(), lr=cfg.lr) for m in client_models]

    for round in range(cfg.num_communication_rounds):

        fedtrunc_server.broadcast(client_models)

        round_losses = [] 
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
                    loss = F.kl_div(predicted.log(), target_distribution, reduction='batchmean')
                    client_losses.append(loss.item())

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    round_losses.append(loss.item())

        fedtrunc_server.aggregate(client_models)

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

        summary['fedtrunc']['losses'].append(float(np.mean(round_losses)))
        for k, v in avg_metrics.items():
            summary['fedtrunc']['metrics'][k].append(v)
        
        save_round_plots(
            algo="fedtrunc",
            round_idx=round,
            client_models=client_models,
            env=env,
            z=z_np,
            out_root=getattr(cfg, "plot_dir", "plots"),
        )
        
    compile_gifs(
        algo="fedtrunc",             
        out_root=getattr(cfg, "plot_dir", "plots"),
        n_clients=env.n_clients,
        n_arms=cfg.n_arms,
        fps=6,
    )

    return summary