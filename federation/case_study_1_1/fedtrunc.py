from typing import List, Optional
from models.c51_bandit import C51Bandit
from utils.general import norm_vol_weights
from configs.config_templates.case_study_1_1 import Config

import numpy as np

import torch


class FedTruncServer:
    def __init__(self, global_model: C51Bandit, cfg: Config):
        self.global_model = global_model
        self.cfg = cfg
        # Only these layers are synchronized across clients
        self.shared_keys = ['emb.weight', 'fc1.weight', 'fc1.bias']

    @torch.no_grad()
    def aggregate(
        self,
        client_models: List[C51Bandit],
        participate: Optional[List[int]] = None,  # optional: indices of reporting clients
    ):
        full_w = np.asarray(self.cfg.client_weights, dtype=np.float64)

        if participate is None:
            participate = list(range(len(client_models)))

        # weights restricted to participating clients
        w_part = norm_vol_weights(full_w[np.array(participate, dtype=int)])

        # pull state dicts of participating clients
        sds = [client_models[i].state_dict() for i in participate]
        gsd = self.global_model.state_dict()

        # sanity check: all shared keys must exist in participants
        for k in self.shared_keys:
            if k not in gsd:
                raise KeyError(f"Shared key '{k}' not found in global model state_dict.")
            for sd in sds:
                if k not in sd:
                    raise KeyError(f"Shared key '{k}' not found in a client model state_dict.")

        # weighted average only for shared layers; everything else remains as in global
        for k in self.shared_keys:
            stacked = torch.stack([sd[k] for sd in sds], dim=0)              # [P, ...]
            w = torch.tensor(w_part, dtype=stacked.dtype, device=stacked.device) \
                    .view(-1, *([1] * (stacked.ndim - 1)))
            gsd[k] = (w * stacked).sum(dim=0)

        self.global_model.load_state_dict(gsd, strict=True)

    def broadcast(self, client_models: List[C51Bandit]):
        # copy shared layers from global into each client; keep client-specific heads intact
        gsd = self.global_model.state_dict()
        for m in client_models:
            msd = m.state_dict()
            for k in self.shared_keys:
                if k not in gsd:
                    raise KeyError(f"Shared key '{k}' not found in global model state_dict.")
                msd[k] = gsd[k]
            m.load_state_dict(msd, strict=True)