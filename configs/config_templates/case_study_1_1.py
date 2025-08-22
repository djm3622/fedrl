from dataclasses import field, dataclass

@dataclass
class Config:
    # model
    device: str = "mps"
    lr: float = 1e-4
    batch_size: int = 32
    grad_clip: float = 10.0
    seeds: list = field(default_factory=lambda: [0])

    # federation
    num_communication_rounds: int = 500
    local_epochs_per_round: int = 4
    n_clients: int = 5
    n_arms: int = 3
    cvar_alpha: float = 0.1
    client_weights: list = field(default_factory=lambda: [1.0])

    # fedrl specific
    kappa: float = 0.05
    server_kd_lr: float = 1e-2
    ot_reg: float = 1e-1
    ot_p: int = 1
    server_kd_steps: int = 100