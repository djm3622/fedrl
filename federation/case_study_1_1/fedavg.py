from typing import List, Optional
from models.c51_bandit import C51Bandit
from utils.general import norm_vol_weights
from configs.config_templates.case_study_1_1 import Config

import numpy as np

import torch

class FedAvgServer:
    def __init__(self, global_model: C51Bandit, cfg: Config):
        self.global_model = global_model
        self.cfg = cfg

    @torch.no_grad()
    def aggregate(
        self,
        client_models: List[C51Bandit],
        participate: Optional[List[int]] = None,  # optional: indices of reporting clients
    ):
        if participate is None:
            participate = list(range(len(client_models)))

        full_w = np.asarray(self.cfg.client_weights, dtype=np.float64)

        # take only participating clients' weights and normalize
        w_part = norm_vol_weights(full_w[np.array(participate, dtype=int)])

        # collect participant state_dicts
        sds = [client_models[i].state_dict() for i in participate]

        # weighted average per key
        avg = {}
        for k in sds[0].keys():
            stacked = torch.stack([sd[k] for sd in sds], dim=0)
            w = torch.tensor(w_part, dtype=stacked.dtype, device=stacked.device).view(
                -1, *([1] * (stacked.ndim - 1))
            )
            avg[k] = (w * stacked).sum(dim=0)

        self.global_model.load_state_dict(avg, strict=True)

    def broadcast(self, client_models: List[C51Bandit]):
        sd = self.global_model.state_dict()
        for m in client_models:
            m.load_state_dict(sd, strict=True)