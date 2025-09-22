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
from eval.distributions import save_round_plots, compile_videos


def train_fedrl(env: HeteroBandit, cfg: Config, summary: Dict):
    w = normalize_weights(cfg.client_weights, cfg.n_clients)
    client_models = [C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device) for _ in range(env.n_clients)]
    z_c51 = torch.linspace(-4, 4, client_models[0].n_atoms).to(cfg.device)
    z_np = z_c51.cpu().numpy()
    global_model = C51Bandit(cfg.n_arms, 16, 64, 101).to(cfg.device)
    fedrl_server = FeDRLServer(global_model=global_model, cfg=cfg, z=z_c51)

    client_optimizers = [AdamW(m.parameters(), lr=cfg.lr) for m in client_models]

    # ---------- Trust-region KL (dual) ----------
    use_tr  = bool(getattr(cfg, "tr_kl_enabled", True))
    per_arm = bool(getattr(cfg, "tr_per_arm", False))
    dual_lr = float(getattr(cfg, "tr_dual_lr", 1e-3))
    tau     = float(getattr(cfg, "tr_ema_tau", 0.95))
    lam_init= float(getattr(cfg, "tr_lambda_init", 0.0))

    if use_tr:
        if per_arm:
            lam    = torch.full((env.n_clients, cfg.n_arms), lam_init, device=cfg.device)
            kl_ema = torch.zeros_like(lam)
        else:
            lam    = torch.full((env.n_clients,), lam_init, device=cfg.device)
            kl_ema = torch.zeros_like(lam)

        # ---- per-client budgets (eps_i) and caps (lam_max_i) ----
        w_t   = torch.as_tensor(w, device=cfg.device, dtype=torch.float32)
        w_max = float(w_t.max())
        eps_base = float(getattr(cfg, "tr_eps", 0.03))
        alpha    = float(getattr(cfg, "tr_eps_alpha", 0.8))     # loosen big clients' budgets
        lam_base = float(getattr(cfg, "tr_lambda_max", 5.0))
        beta     = float(getattr(cfg, "tr_lammax_beta", 0.6))   # shrink big clients' caps

        eps_i = eps_base * (1.0 + alpha * (w_t / (w_max + 1e-12)))                      # [N]
        lam_max_i = lam_base * (1.0 - beta * (w_t / (w_max + 1e-12)))                    # [N]
        lam_max_i = torch.clamp(lam_max_i, min=0.3 * lam_base)

    # ------------ training ------------
    for round in range(cfg.num_communication_rounds):
        round_ce, round_kl = [], []

        for client_id in range(env.n_clients):
            model = client_models[client_id]
            optimizer = client_optimizers[client_id]
            model.train()

            batch_size = max(1, int(np.ceil(cfg.batch_size * (w[client_id] * cfg.n_clients))))

            for local_epoch in range(cfg.local_epochs_per_round):
                for arm_id in range(cfg.n_arms):
                    optimizer.zero_grad(set_to_none=True)

                    rewards_tensor = torch.tensor(
                        env.sample(client_id, arm_id, batch_size),
                        dtype=torch.float32, device=cfg.device
                    )
                    arm_tensor = torch.tensor([[arm_id]], device=cfg.device).repeat(batch_size, 1)

                    # Forward
                    predicted = model(arm_tensor)  # [B, A], probabilities (softmax in C51Bandit)
                    target_distribution = c51_project(rewards_tensor, z_c51)  # [B, A]

                    # Data term: KL(target || pred)  (F.kl_div expects log-input)
                    ce = F.kl_div(predicted.log(), target_distribution, reduction='batchmean')

                    # Teacher (prior): prefer LOO if available; otherwise server global model
                    last_loo = getattr(fedrl_server, "last_loo_bary", None)
                    if last_loo is not None:
                        prior = torch.as_tensor(last_loo[client_id, arm_id],
                                                device=cfg.device, dtype=predicted.dtype).unsqueeze(0).repeat(batch_size, 1)
                    else:
                        with torch.no_grad():
                            prior = fedrl_server.global_model(arm_tensor)  # [B, A]

                    # ---- Teacher KL (mode-seeking): KL(pred || prior) ----
                    eps = 1e-8
                    p = torch.clamp(predicted, eps, 1.0)
                    p = p / p.sum(dim=1, keepdim=True)
                    q = torch.clamp(prior, eps, 1.0)
                    q = q / q.sum(dim=1, keepdim=True)
                    kl_teacher = (p * (p.log() - q.log())).sum(dim=1).mean()

                    # Loss with dual-controlled λ (or fixed κ for ablations)
                    if use_tr:
                        lam_i = lam[client_id] if not per_arm else lam[client_id, arm_id]
                        loss = ce + lam_i * kl_teacher
                        # EMA of KL for dual update
                        if not per_arm:
                            kl_ema[client_id] = tau * kl_ema[client_id] + (1 - tau) * kl_teacher.detach()
                        else:
                            kl_ema[client_id, arm_id] = tau * kl_ema[client_id, arm_id] + (1 - tau) * kl_teacher.detach()
                    else:
                        base = float(getattr(cfg, "kappa", 0.0))
                        wi = max(w[client_id], 1e-3)
                        kappa_scaled = min(base / (wi ** 1.0), 20.0)
                        loss = ce + kappa_scaled * kl_teacher

                    # Backprop
                    loss.backward()
                    if getattr(cfg, "grad_clip", None) is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                    round_ce.append(ce.item())
                    round_kl.append(kl_teacher.item())

                # ----- Dual ascent update (once per local epoch) -----
                if use_tr:
                    if not per_arm:
                        err = kl_ema - eps_i                       # [N]
                        lam = torch.clamp(lam + dual_lr * err, min=0.0)
                        lam = torch.minimum(lam, lam_max_i)
                    else:
                        err = kl_ema - eps_i[:, None]              # [N, K]
                        lam = torch.clamp(lam + dual_lr * err, min=0.0)
                        lam = torch.minimum(lam, lam_max_i[:, None])

        # ----- Server ops -----
        fedrl_server.aggregate(client_models)
        # Produces global teacher; if your server computes LOO, it will also populate last_loo_bary
        p_bars, _ = fedrl_server.distill(client_models)

        # ----- Metrics -----
        all_metrics = []
        for client_id in range(env.n_clients):
            pred_hist, truth_hist = [], []
            for arm_id in range(cfg.n_arms):
                with torch.no_grad():
                    pnow = client_models[client_id](
                        torch.tensor([arm_id], device=cfg.device).unsqueeze(0)
                    )[0].cpu().numpy()
                pred_hist.append(normalize_hist(pnow))
                truth = env.estimate_truth_hist(z_np, client_id, arm_id, nsamp=20000)
                truth_hist.append(truth)
            pred_hist = np.stack(pred_hist, axis=0); truth_hist = np.stack(truth_hist, axis=0)
            m = compute_metrics(pred_hist, truth_hist, z_np, cvar_alpha=cfg.cvar_alpha)
            all_metrics.append(m)

        avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
        summary['fedrl']['losses'].append(float(np.mean(round_ce)))
        summary['fedrl']['kl_losses'].append(float(np.mean(round_kl)))
        for k, v in avg_metrics.items():
            summary['fedrl']['metrics'][k].append(v)

        # (Optional) log λ and KL̂ for debugging
        if use_tr:
            summary['fedrl'].setdefault('lambda', []).append(lam.detach().cpu().numpy().copy())
            summary['fedrl'].setdefault('kl_ema', []).append(kl_ema.detach().cpu().numpy().copy())

        if False:
            save_round_plots(
                algo="fedrl",
                round_idx=round,
                client_models=client_models,
                env=env,
                z=z_np,
                out_root=getattr(cfg, "plot_dir", "plots"),
                global_pbars=p_bars,          # overlays dashed "Global (bary)"
                # loo_pbars=fedrl_server.last_loo_bary if you maintain it as [N,K,A]
            )

    if False:
        compile_videos(
            algo="fedrl",            
            out_root=getattr(cfg, "plot_dir", "plots"),
            n_clients=env.n_clients,
            n_arms=cfg.n_arms,
        )

    return summary
