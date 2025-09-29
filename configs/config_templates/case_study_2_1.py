from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterable, Sequence, List, Dict, Literal
from collections import deque

Coord = Tuple[int, int]  # (row, col)

# --- Obstacle templates (canonical keys are lowercase alnum with no separators) ---
# We accept "WarehouseA", "warehouse_a", "WAREHOUSE-A", etc.
def _canon(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return "".join(ch for ch in s.lower() if ch.isalnum())

_OBSTACLE_TEMPLATES: Dict[str, Iterable[Coord]] = {
    "warehousea": [
        *[(r, 3) for r in range(2, 6)],  # vertical shelf 1
        *[(r, 6) for r in range(3, 7)],  # vertical shelf 2
    ],
    "empty": [],
}

@dataclass
class MultiAgentGridConfig:
    # --- trainer (unused by env but kept for completeness) ---
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
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    seed: int = 42
    log_interval: int = 1_000
    eval_every_steps: int = 100_000

    wandb_project: str = "mappo-gridworld"
    wandb_run_name: str = "case_study_1_2_v0"
    wandb_mode: str = "online"
    log_video: bool = False

    # --- map ---
    H: int = 10
    W: int = 10

    starts: Sequence[Coord] = ((9, 0), (0, 0), (9, 9))
    goals: Sequence[Coord] = ((0, 5),)

    obstacles: Iterable[Coord] = field(default_factory=list)
    obstacle_template_id: Optional[str] = "WarehouseA"  # case/format-insensitive

    # Hazard region (inclusive bounds). If cells provided, bounds ignored.
    hazard_zone_cells: Optional[Iterable[Coord]] = None
    hazard_zone_bounds: Optional[Tuple[int, int, int, int]] = (2, 6, 4, 8)

    # client-specific hazard severity
    hazard_prob: float = 0.20
    hazard_loss: float = 120.0

    # base costs/rewards
    step_cost: float = -0.1
    goal_reward: float = +100.0
    catastrophe_reward: float = -120.0

    # penalties / limits
    invalid_move_penalty: float = -0.1
    max_steps: int = 120

    # MARL and dynamics
    n_agents: int = 3
    slip_prob: float = 0.05
    sample_active_hazard_each_episode: bool = False
    block_on_collision: bool = True

    # Observations
    critic_obs_mode: Literal["grid", "none"] = "grid"
    actor_obs_mode: Literal["coords", "ego"] = "ego"
    ego_k: int = 5  # odd

    # Movable object (crate) and pushing rules
    has_objects: bool = True
    object_starts: Sequence[Coord] = ((6, 2), (6, 3), (6, 4))
    object_goal: Optional[Coord] = (9, 5)
    require_two_pushers: bool = False
    object_push_penalty: float = 0.05
    object_reward: float = 300.0

    use_push_step_cost: bool = True
    push_step_cost: float = 0.0
    push_fail_penalty: float = 0.0

    # Stacking options on goals
    allow_multi_agent_on_goal: bool = True
    allow_objects_stack_on_object_goal: bool = True

    # Distributional RL options
    param_type: Literal["expected", "distributional"] = "expected"

    # ----------------- helpers -----------------
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

    # ----------------- validation -----------------
    def validate(self):
        H, W = int(self.H), int(self.W)

        # apply obstacle template first (case/format-insensitive id)
        tid = _canon(self.obstacle_template_id)
        if tid is not None:
            if tid not in _OBSTACLE_TEMPLATES:
                raise ValueError(
                    f"unknown obstacle template id '{self.obstacle_template_id}'. "
                    f"Known: {list(_OBSTACLE_TEMPLATES.keys())}"
                )
            self.obstacles = list(_OBSTACLE_TEMPLATES[tid])

        # coerce numeric tuples/lists
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
