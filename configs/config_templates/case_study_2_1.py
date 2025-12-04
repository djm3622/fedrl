from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterable, Sequence, List, Dict, Literal, Union
from collections import deque

Coord = Tuple[int, int]  # (row, col)

def _canon(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return "".join(ch for ch in s.lower() if ch.isalnum())

# Six obstacle layouts (WarehouseA is your current one).
_OBSTACLE_TEMPLATES: Dict[str, Iterable[Coord]] = {
    # Original layout: two vertical shelves
    "warehousea": [
        *[(r, 3) for r in range(2, 6)],  # vertical shelf 1
        *[(r, 6) for r in range(3, 7)],  # vertical shelf 2
    ],

    # Example: two tall shelves with a doorway on the right
    "warehouseb": [
        *[(r, 2) for r in range(2, 9)],                     # left shelf
        *[(r, 7) for r in range(1, 8) if r not in (4, 5)],  # right shelf with gap
    ],

    # Example: U shaped layout around the middle
    "warehousec": [
        *[(6, c) for c in range(2, 8)],  # bottom of U
        *[(r, 2) for r in range(2, 4)],  # left side
        *[(r, 7) for r in range(2, 4)],  # right side
    ],

    # Example: simple maze like zigzag
    "warehoused": [
        *[(2, c) for c in range(1, 9)],
        *[(5, c) for c in range(0, 8) if c != 3],
        *[(r, 4) for r in range(3, 8) if r != 5],
    ],

    # Example: central block with side pillars
    "warehousee": [
        *[(r, c) for r in range(3, 6) for c in range(3, 6)],  # 3x3 center block
        *[(r, 1) for r in range(2, 8, 2)],                    # left pillars
        *[(r, 8) for r in range(1, 9, 2)],                    # right pillars
    ],

    # Example: sparse scattered obstacles
    "warehousef": [
        (1, 3), (1, 6), (2, 8), (3, 1), (4, 4),
        (4, 7), (5, 2), (6, 5), (7, 3), (7, 7),
    ],

    "empty": [],
}

# Hazard layout templates. Each template can define either "bounds" or "cells".
_HAZARD_TEMPLATES: Dict[str, Dict[str, Union[Iterable[Coord], Tuple[int, int, int, int]]]] = {
    # These are kept as generic fallbacks. They will be overridden by
    # _ENV_LAYOUT_TEMPLATES for the warehouse layouts below.
    "warehousea": {"bounds": (2, 6, 4, 8)},
    "warehouseb": {"bounds": (1, 5, 3, 8)},
    "warehousec": {"bounds": (5, 5, 8, 8)},
    "warehoused": {"bounds": (1, 2, 3, 4)},
    "warehousee": {"bounds": (4, 6, 7, 9)},
    "warehousef": {"bounds": (2, 1, 5, 3)},
}

# Full environment layouts keyed by warehouse id.
# Each entry corresponds to your Env1..Env6 definitions.
EnvLayoutSpec = Dict[str, Union[
    Sequence[Coord],              # starts, goals, object_starts
    Iterable[Coord],              # hazard_zone_cells
    Coord,                        # object_goal
    Tuple[int, int, int, int],    # hazard_zone_bounds
    None
]]

_ENV_LAYOUT_TEMPLATES: Dict[str, EnvLayoutSpec] = {
    # Env1
    "warehousea": {
        "starts": ((9, 0), (0, 0), (9, 9)),
        "goals": ((0, 5),),
        "object_starts": ((3, 1), (6, 1), (6, 8)),
        "object_goal": (9, 5),
        "hazard_zone_cells": None,
        "hazard_zone_bounds": (3, 4, 5, 5),
    },

    # Env2
    "warehouseb": {
        "starts": ((9, 1), (0, 1), (9, 8)),
        "goals": ((0, 5),),
        "object_starts": ((3, 1), (6, 1), (6, 8)),
        "object_goal": (9, 4),
        "hazard_zone_cells": None,
        "hazard_zone_bounds": (0, 6, 4, 8),
    },

    # Env3
    "warehousec": {
        "starts": ((9, 0), (0, 0), (9, 9)),
        # Single shared goal (length 1) is allowed
        "goals": ((0, 4),),
        "object_starts": ((8, 2), (4, 7), (4, 2)),
        "object_goal": (9, 5),
        "hazard_zone_cells": None,
        "hazard_zone_bounds": (2, 3, 3, 6),
    },

    # Env4
    "warehoused": {
        "starts": ((9, 0), (3, 5), (9, 9)),
        "goals": ((0, 5),),
        "object_starts": ((8, 4), (7, 1), (7, 7)),
        "object_goal": (9, 5),
        "hazard_zone_bounds": None,
        "hazard_zone_cells": [
            (3, 0), (3, 1), (3, 2), (3, 3),
            (4, 0), (4, 1), (4, 2), (4, 3),
        ],
    },

    # Env5
    "warehousee": {
        "starts": ((9, 0), (0, 9), (9, 9)),
        "goals": ((0, 5),),
        "object_starts": ((2, 4), (4, 6), (6, 4)),
        "object_goal": (9, 5),
        "hazard_zone_cells": None,
        "hazard_zone_bounds": (1, 0, 7, 2),
    },

    # Env6
    "warehousef": {
        "starts": ((9, 0), (0, 0), (9, 9)),
        "goals": ((0, 5),),
        "object_starts": ((5, 1), (7, 2), (7, 7)),
        "object_goal": (9, 5),
        "hazard_zone_cells": None,
        "hazard_zone_bounds": (3, 3, 5, 5),
    },
}

@dataclass
class MultiAgentGridConfig:
    device: str = "cuda"
    total_steps: int = 5_000_000
    rollout_len: int = 512
    update_epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5
    seed: int = 42
    log_interval: int = 1_000
    eval_every_steps: int = 100_000

    wandb_project: str = "mappo-gridworld"
    wandb_run_name: str = "case_study_1_2_v0"
    wandb_mode: str = "online"
    log_video: bool = False

    H: int = 10
    W: int = 10

    starts: Sequence[Coord] = ((9, 0), (0, 0), (9, 9))
    goals: Sequence[Coord] = ((0, 5),)

    obstacles: Iterable[Coord] = field(default_factory=list)
    obstacle_template_id: Optional[str] = "WarehouseA"  # case/format-insensitive

    hazard_zone_cells: Optional[Iterable[Coord]] = None
    hazard_zone_bounds: Optional[Tuple[int, int, int, int]] = None
    hazard_template_id: Optional[str] = "WarehouseA"  # case/format insensitive

    hazard_prob: float = 0.20
    hazard_loss: float = 120.0

    step_cost: float = -0.1
    goal_reward: float = +100.0
    catastrophe_reward: float = -120.0

    invalid_move_penalty: float = -0.1
    max_steps: int = 120

    n_agents: int = 3
    slip_prob: float = 0.05
    sample_active_hazard_each_episode: bool = False
    block_on_collision: bool = True

    critic_obs_mode: Literal["grid", "none"] = "grid"
    actor_obs_mode: Literal["coords", "ego"] = "ego"
    ego_k: int = 5  # odd

    has_objects: bool = True
    object_starts: Sequence[Coord] = ((6, 2), (6, 3), (6, 4))
    object_goal: Optional[Coord] = (9, 5)
    require_two_pushers: bool = False
    object_push_penalty: float = 0.05
    object_reward: float = 300.0

    use_push_step_cost: bool = True
    push_step_cost: float = 0.0
    push_fail_penalty: float = 0.0

    allow_multi_agent_on_goal: bool = True
    allow_objects_stack_on_object_goal: bool = True

    param_type: Literal["expected", "distributional"] = "expected"
    
    critic_target_ema: float = 0.01
    lr_actor: float = 1e-4
    lr_critic: float = 2e-4
    critic_weight_decay: float = 0.0
    qr_kappa: float = 1.0
    n_quantiles: int = 21
    mean_anchor_coef: float = 0.1
    exploration_eps: float = 0.05

    risk_enable: bool = True         # turn on risk-aware mixing
    risk_beta: float = 0.2           # 0.0..1.0 trade-off (performance vs safety)
    cvar_alpha: float = 0.10         # tail level for CVaR

    # federation
    num_communication_rounds: int = 1
    local_epochs_per_round: int = 1
    n_clients: int = 1
    client_weights: List[float] = field(default_factory=lambda: [1.0])

    # NEW: per client environment templates; if shorter than n_clients they repeat cyclically
    client_warehouse_templates: Sequence[str] = (
        "WarehouseA",
        "WarehouseB",
        "WarehouseC",
        "WarehouseD",
        "WarehouseE",
        "WarehouseF",
    )

    # fedrl
    enable_ae_aux: bool = True          # must be true to run and log AE
    fedrl_d_latent: int = 128           # optional; default 128
    fedrl_enc_lr: float = 1.0e-3        # optional
    fedrl_dec_lr: float = 5.0e-4        # optional
    fedrl_rec_lambda: float = 1.0e-2    # not used in the loss, only kept for compatibility
    fedrl_latent_hw: int = 5

    enabled: bool = True
    alpha: float = 0.5
    beta: float = 1.0
    radius_abs: float = 0.0
    radius_rel: float = 0.10

    lambda_cvar: float = 0.5     # try 0.02–0.05
    tau_cvar: float = 1.0         # optional temperature, 1.0–2.0

    agg_hazard_eps: float = 1e-3
    agg_w_max: float = 10.0

    def _nearest_free_noncorner(
        self,
        H: int,
        W: int,
        obstacles: set[Coord],
        occupied: set[Coord],
        start: Coord
    ) -> Coord:
        corners = {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}

        def neighbors(rc: Coord) -> List[Coord]:
            r, c = rc
            cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            return [(rr, cc) for rr, cc in cand if 0 <= rr < H and 0 <= cc < W]

        def is_free(rc: Coord) -> bool:
            return (rc not in obstacles) and (rc not in occupied)

        def has_three_accessible_sides(rc: Coord) -> bool:
            nb = neighbors(rc)
            free = [n for n in nb if is_free(n)]
            required = 3 if len(nb) >= 3 else len(nb)
            return len(free) >= required

        def is_ok(rc: Coord) -> bool:
            if not (0 <= rc[0] < H and 0 <= rc[1] < W):
                return False
            if rc in obstacles or rc in occupied:
                return False
            if rc in corners:
                return False
            return has_three_accessible_sides(rc)

        if is_ok(start):
            return start

        q = deque([start])
        seen = {start}
        while q:
            rc = q.popleft()
            if is_ok(rc):
                return rc
            r, c = rc
            for n in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                rr, cc = n
                if 0 <= rr < H and 0 <= cc < W and n not in seen:
                    seen.add(n)
                    q.append(n)

        raise RuntimeError("no free non-corner cell with 3 accessible sides found")

    def _resolve_hazard_zone_cells(self) -> List[Coord]:
        if self.hazard_zone_cells is not None:
            return [(int(r), int(c)) for (r, c) in self.hazard_zone_cells]
        if self.hazard_zone_bounds is not None:
            r0, c0, r1, c1 = self.hazard_zone_bounds
            r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
            return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        return []

    def validate(self):
        H, W = int(self.H), int(self.W)

        # Resolve obstacle layout from template id, if provided.
        tid = _canon(self.obstacle_template_id)
        if tid is not None:
            if tid not in _OBSTACLE_TEMPLATES:
                raise ValueError(
                    f"unknown obstacle template id '{self.obstacle_template_id}'. "
                    f"Known obstacle templates: {list(_OBSTACLE_TEMPLATES.keys())}"
                )
            # Obstacles from warehouse template
            self.obstacles = list(_OBSTACLE_TEMPLATES[tid])

            # If we have a full environment layout for this warehouse,
            # override starts, goals, objects, and hazard zone.
            if tid in _ENV_LAYOUT_TEMPLATES:
                env_spec = _ENV_LAYOUT_TEMPLATES[tid]

                if "starts" in env_spec:
                    self.starts = tuple(env_spec["starts"])  # type: ignore[arg-type]
                    # Keep n_agents consistent with starts
                    self.n_agents = len(self.starts)

                if "goals" in env_spec:
                    self.goals = tuple(env_spec["goals"])  # type: ignore[arg-type]

                if "object_starts" in env_spec:
                    self.object_starts = tuple(env_spec["object_starts"])  # type: ignore[arg-type]

                if "object_goal" in env_spec:
                    self.object_goal = env_spec["object_goal"]  # type: ignore[assignment]

                if "hazard_zone_cells" in env_spec:
                    self.hazard_zone_cells = env_spec["hazard_zone_cells"]  # type: ignore[assignment]

                if "hazard_zone_bounds" in env_spec:
                    self.hazard_zone_bounds = env_spec["hazard_zone_bounds"]  # type: ignore[assignment]

        # Resolve hazard layout from hazard_template_id only if no explicit hazard
        if self.hazard_zone_cells is None and self.hazard_zone_bounds is None:
            htid = _canon(getattr(self, "hazard_template_id", None))
            if htid is not None:
                if htid not in _HAZARD_TEMPLATES:
                    raise ValueError(
                        f"unknown hazard template id '{self.hazard_template_id}'. "
                        f"Known hazard templates: {list(_HAZARD_TEMPLATES.keys())}"
                    )

                spec = _HAZARD_TEMPLATES[htid]
                cells = spec.get("cells")
                bounds = spec.get("bounds")

                if cells is not None and bounds is not None:
                    raise ValueError(
                        f"hazard template '{self.hazard_template_id}' "
                        f"defines both cells and bounds"
                    )

                if cells is not None:
                    self.hazard_zone_cells = list(cells)
                    self.hazard_zone_bounds = None
                elif bounds is not None:
                    r0, c0, r1, c1 = bounds
                    self.hazard_zone_cells = None
                    self.hazard_zone_bounds = (int(r0), int(c0), int(r1), int(c1))
                else:
                    # template with no hazard is allowed
                    self.hazard_zone_cells = None
                    self.hazard_zone_bounds = None

        # Cast everything to ints and perform checks
        self.starts = tuple((int(r), int(c)) for r, c in self.starts)
        self.goals = tuple((int(r), int(c)) for r, c in self.goals)
        self.obstacles = list((int(r), int(c)) for (r, c) in self.obstacles)

        if self.has_objects:
            self.object_starts = tuple((int(r), int(c)) for r, c in self.object_starts)
            if self.object_goal is not None:
                r, c = self.object_goal
                self.object_goal = (int(r), int(c))

        if self.hazard_zone_cells is not None:
            self.hazard_zone_cells = list((int(r), int(c)) for r, c in self.hazard_zone_cells)

        if self.hazard_zone_bounds is not None:
            r0, c0, r1, c1 = self.hazard_zone_bounds
            self.hazard_zone_bounds = (int(r0), int(c0), int(r1), int(c1))

        # basic checks
        assert self.n_agents >= 1, "n_agents must be >= 1"
        assert len(self.starts) == self.n_agents, "starts length must equal n_agents"
        assert len(self.goals) in (1, self.n_agents), "goals must have length 1 or n_agents"

        obs = set(self.obstacles)
        for s in self.starts:
            r, c = s
            assert 0 <= r < H and 0 <= c < W, "start out of bounds"
            assert s not in obs, "start cannot be an obstacle"
        for g in (self.goals if len(self.goals) == self.n_agents else [self.goals[0]]):
            r, c = g
            assert 0 <= r < H and 0 <= c < W, "goal out of bounds"
            assert g not in obs, "goal cannot be an obstacle"

        hz_cells = set(self._resolve_hazard_zone_cells())
        for (r, c) in hz_cells:
            assert 0 <= r < H and 0 <= c < W, "hazard zone cell out of bounds"

        assert 0.0 <= self.hazard_prob <= 1.0, "hazard_prob must be in [0,1]"
        assert self.ego_k % 2 == 1 and self.ego_k >= 1, "ego_k must be odd and >= 1"

        if self.has_objects:
            assert self.object_goal is not None, "object_goal required when has_objects=True"
            ogr, ogc = self.object_goal
            assert 0 <= ogr < H and 0 <= ogc < W, "object_goal out of bounds"
            assert self.object_goal not in obs, "object_goal cannot be an obstacle"

            fixed_starts: List[Coord] = []
            for os in self.object_starts:
                os = self._nearest_free_noncorner(
                    H=H,
                    W=W,
                    obstacles=set(self.obstacles),
                    occupied=(set(self.starts)
                              | set(self.goals if len(self.goals) == self.n_agents else [self.goals[0]])
                              | set(fixed_starts)),
                    start=os,
                )
                fixed_starts.append(os)
            self.object_starts = tuple(fixed_starts)

    def make_client_cfg(self, client_idx: int) -> "MultiAgentGridConfig":
        """
        Create a client specific config with a chosen warehouse environment.

        Mapping:
          client_idx -> client_warehouse_templates[client_idx % len(client_warehouse_templates)]
        """
        from dataclasses import replace

        cfg_i: MultiAgentGridConfig = replace(self)

        templates = list(self.client_warehouse_templates)
        if templates:
            chosen = templates[client_idx % len(templates)]
            cfg_i.obstacle_template_id = chosen
            # default: align hazard template with obstacle template
            if getattr(cfg_i, "hazard_template_id", None) is not None:
                cfg_i.hazard_template_id = chosen

        cfg_i.validate()
        return cfg_i
