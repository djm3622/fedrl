from typing import List, Optional
from models.c51_bandit import C51Bandit
from utils.general import norm_vol_weights
from utils.general import to_numpy, normalize_hist
from federation.case_study_1_1.barycenter import pot_wasserstein_barycenter
from configs.config_templates.case_study_1_1 import Config

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

class FeDRLServer:
    def __init__(self, global_model: C51Bandit, cfg: Config, z: torch.Tensor):
        self.global_model = global_model
        self.cfg = cfg
        self.z_np = to_numpy(z)

    def _bary(self, Ps: np.ndarray) -> np.ndarray:
        return pot_wasserstein_barycenter(Ps, self.z_np, reg=self.cfg.ot_reg, p=self.cfg.ot_p)

    @torch.no_grad()
    def aggregate(
        self,
        client_models: List[C51Bandit],
        participate: Optional[List[int]] = None,  # optional indices of reporting clients
    ):
        full_w = np.asarray(self.cfg.client_weights, dtype=np.float64)

        if participate is None:
            participate = list(range(len(client_models)))

        # normalize over participating clients
        w_part = norm_vol_weights(full_w[np.array(participate, dtype=int)])

        # collect participant states
        sds = [client_models[i].state_dict() for i in participate]

        # volume-weighted average of ALL params/buffers
        avg = {}
        for k in sds[0].keys():
            stacked = torch.stack([sd[k] for sd in sds], dim=0)  # [P, ...]
            w = torch.tensor(w_part, dtype=stacked.dtype, device=stacked.device).view(
                -1, *([1] * (stacked.ndim - 1))
            )
            avg[k] = (w * stacked).sum(dim=0)

        self.global_model.load_state_dict(avg, strict=True)

    def distill(self, client_models: List[C51Bandit]):
        client_preds_per_arm = [[] for _ in range(self.global_model.n_arms)]
        device = next(self.global_model.parameters()).device
        for m in client_models:
            for arm_id in range(self.global_model.n_arms):
                arm_tensor = torch.tensor([arm_id]).to(device)
                with torch.no_grad():
                    predicted_distribution = m(arm_tensor).squeeze().cpu().numpy()
                client_preds_per_arm[arm_id].append(predicted_distribution)
        return self.distill_to_barycenters(client_preds_per_arm)

    def distill_to_barycenters(self, client_preds_per_arm: List[np.ndarray], freeze_fc1: bool = False):
        device = next(self.global_model.parameters()).device

        # Choose params to optimize
        if freeze_fc1:
            params = [p for n, p in self.global_model.named_parameters()
                      if not (n.startswith('emb.') or n.startswith('fc1.'))]
        else:
            params = list(self.global_model.parameters())
        opt = AdamW(params, lr=self.cfg.server_kd_lr)

        # Compute barycenters per arm
        p_bars = []
        for a, Ps in enumerate(client_preds_per_arm):
            Ps = np.vstack([normalize_hist(p) for p in Ps])
            pbar = self._bary(Ps)
            p_bars.append(pbar)
        p_bars = np.stack(p_bars, axis=0)  # [K, A]
        p_bars_t = torch.from_numpy(p_bars).to(device=device, dtype=torch.float32)

        arms = torch.arange(p_bars_t.shape[0], device=device)
        kd_losses = []
        for _ in range(self.cfg.server_kd_steps):
            p_pred = self.global_model(arms)  # [K, A]
            loss = F.kl_div(p_pred.log(), p_bars_t, reduction='batchmean')
            kd_losses.append(float(loss.detach().cpu().item()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None and len(params) > 0:
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            opt.step()
        return p_bars, kd_losses