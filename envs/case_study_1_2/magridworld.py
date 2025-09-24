from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Iterable, Optional, Tuple, List, Sequence, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

Coord = Tuple[int, int]  # (row, col)

# -----------------------------------------------------------------------------
# Obstacle templates (aligned across clients)
# -----------------------------------------------------------------------------
OBSTACLE_TEMPLATES: Dict[str, Iterable[Coord]] = {
    "WarehouseA": [
        # vertical shelves
        *[(r, 3) for r in range(2, 6)],
        *[(r, 6) for r in range(3, 7)],
    ],
    "Empty": [],
}

# -----------------------------------------------------------------------------
# Multi-agent config
# -----------------------------------------------------------------------------
@dataclass
class MultiAgentGridConfig:
    H: int = 10
    W: int = 10

    # agents far apart by default
    starts: Sequence[Coord] = ((9, 0), (0, 0), (9, 9))
    # single shared goal near top right
    goals: Sequence[Coord] = ((0, 5),)

    # map layout
    obstacles: Iterable[Coord] = field(default_factory=list)
    obstacle_template_id: Optional[str] = "WarehouseA"  # default template

    # smaller hazard zone by default (3x3). indices are inclusive
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
    object_starts: Sequence[Coord] = ((6, 2), (6, 3), (6, 4))  # 3 objects by default
    object_goal: Optional[Coord] = (9, 5)
    require_two_pushers: bool = False
    object_push_penalty: float = 0.05  # shaping penalty per push, if any
    object_reward: float = 300.0    # NEW: one-time bonus per object delivered

    # --- pushing economics ---
    # Use a lower-magnitude penalty when a push succeeds (replaces step_cost for pushers on that step)
    use_push_step_cost: bool = True         # NEW: if True, use push_step_cost instead of step_cost for pushers
    push_step_cost: float = 0.0           # NEW: per-step cost for successful push (|push_step_cost| < |step_cost|)
    push_fail_penalty: float = 0.0        # NEW: extra penalty when a push was attempted but failed to move the object


    # Stacking options on goals
    allow_multi_agent_on_goal: bool = True     # agents may share the agent-goal cell(s)
    allow_objects_stack_on_object_goal: bool = True  # multiple objects may share the object-goal cell


    # RNG
    seed: Optional[int] = 42

    # ------------- validation and helpers -------------
    def _nearest_free_noncorner(
        self,
        H: int,
        W: int,
        obstacles: set[Coord],
        occupied: set[Coord],
        start: Coord
    ) -> Coord:
        """
        Find the nearest cell that:
        - is in-bounds,
        - is not an obstacle,
        - is not in `occupied`,
        - is not a map corner,
        - has 3 accessible sides:
            * interior: at least 3 of 4 neighbors are in-bounds and free
            * edge: all in-bounds neighbors (3) are free
        """
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
            # on edges nb has length 3; require all 3 free
            # in interior nb has length 4; require at least 3 free
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
            for n in neighbors(rc):
                if n not in seen:
                    seen.add(n)
                    q.append(n)

        raise RuntimeError("no free non-corner cell with 3 accessible sides found")

    def validate(self):
        assert self.n_agents >= 1, "n_agents must be >= 1"
        assert len(self.starts) == self.n_agents, "starts length must equal n_agents"
        assert len(self.goals) in (1, self.n_agents), "goals must have length 1 or n_agents"
        H, W = self.H, self.W

        # derive obstacles from template if provided
        if self.obstacle_template_id is not None:
            assert self.obstacle_template_id in OBSTACLE_TEMPLATES, "unknown obstacle template id"
            self.obstacles = list(OBSTACLE_TEMPLATES[self.obstacle_template_id])

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

            # treat agent starts and goals as initially occupied to avoid overlaps
            occupied0 = set(self.starts) | set(self.goals if len(self.goals)==self.n_agents else [self.goals[0]])

            fixed_starts: List[Coord] = []
            for os in self.object_starts:
                os = self._nearest_free_noncorner(
                    H=self.H,
                    W=self.W,
                    obstacles=set(self.obstacles),
                    occupied=(set(self.starts)
                            | set(self.goals if len(self.goals) == self.n_agents else [self.goals[0]])
                            | set(fixed_starts)),
                    start=os,
                )
                fixed_starts.append(os)
            self.object_starts = tuple(fixed_starts)

    def _resolve_hazard_zone_cells(self) -> List[Coord]:
        if self.hazard_zone_cells is not None:
            return list(self.hazard_zone_cells)
        if self.hazard_zone_bounds is not None:
            r0, c0, r1, c1 = self.hazard_zone_bounds
            return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        return []

# -----------------------------------------------------------------------------
# Multi-agent GridWorld with aligned hazard zone, object pushing, CTDE-friendly obs
# -----------------------------------------------------------------------------
class MultiAgentGridWorld:
    """
    Actions per agent: 0=Up, 1=Right, 2=Down, 3=Left.
    Step takes list[int] of length n_agents or dict[agent_id]->int.
    Observations:
        - actor_obs: list per agent (coords or k x k ego crop)
        - critic_obs: global (H, W, C) planes if enabled, else None
    """
    ACTIONS = [(-1,0),(0,+1),(+1,0),(0,-1),(0,0)]


    def __init__(self, cfg: MultiAgentGridConfig):
        cfg.validate()
        self.cfg = cfg
        self.rng: np.random.Generator = np.random.default_rng(cfg.seed)
        self._H, self._W = cfg.H, cfg.W
        self._obstacles = set(cfg.obstacles)
        self._t: int = 0

        # state
        self._pos: List[Coord] = list(cfg.starts)
        self._done_agents: List[bool] = [False] * cfg.n_agents
        self._hazard_cells_all: List[Coord] = cfg._resolve_hazard_zone_cells()
        self._hazard_active: Optional[Coord] = None
        self._obj_pos_list: List[Coord] = list(cfg.object_starts) if cfg.has_objects else []
        self._obj_delivered: List[bool] = [False] * len(self._obj_pos_list) if self.cfg.has_objects else []

        # --- NEW: choose a fixed active hazard once at construction if we're not resampling per episode ---
        if self._hazard_cells_all and not cfg.sample_active_hazard_each_episode:
            self._sample_active_hazard_cell()


    # ---------------- helpers ----------------
    def in_bounds(self, rc: Coord) -> bool:
        r, c = rc
        return 0 <= r < self._H and 0 <= c < self._W

    def is_obstacle(self, rc: Coord) -> bool:
        return rc in self._obstacles

    def is_goal(self, idx: int, rc: Coord) -> bool:
        g = self.cfg.goals[idx] if len(self.cfg.goals) == self.cfg.n_agents else self.cfg.goals[0]
        return rc == g

    def _occupied_static(self, rc: Coord) -> bool:
        # static occupancy: walls and shelves
        return (rc in self._obstacles)

    def _occupied_any(self, rc: Coord) -> bool:
        if rc in self._obstacles:
            return True
        if getattr(self.cfg, "allow_objects_stack_on_object_goal", False) and self.cfg.object_goal is not None:
            if rc == self.cfg.object_goal:
                # the goal cell never blocks additional objects; agents vs object on goal handled in propose-move logic
                return False
        # block if any object occupies rc
        if any(rc == opos for opos in getattr(self, "_obj_pos_list", [])):
            return True
        return False


    def _sample_active_hazard_cell(self):
        if self._hazard_cells_all:
            pick = self.rng.choice(np.array(self._hazard_cells_all))
            self._hazard_active = (int(pick[0]), int(pick[1]))
        else:
            self._hazard_active = None

    # ------------- observations -------------
    def _feature_stack_global(self) -> np.ndarray:
        """
        Global critic observation (H, W, C):
        0: obstacles
        1: hazard_zone
        2: hazard_active
        3: objects (one-hot if any object occupies a cell)
        4: agents (any agent)
        5: goals (any goal cell)
        """
        obstacles = np.zeros((self._H, self._W), dtype=np.float32)
        hz = np.zeros((self._H, self._W), dtype=np.float32)
        hz_act = np.zeros((self._H, self._W), dtype=np.float32)
        obj = np.zeros((self._H, self._W), dtype=np.float32)
        agents = np.zeros((self._H, self._W), dtype=np.float32)
        goals = np.zeros((self._H, self._W), dtype=np.float32)

        for (r, c) in self._obstacles:
            obstacles[r, c] = 1.0
        for (r, c) in self._hazard_cells_all:
            hz[r, c] = 1.0
        if self._hazard_active is not None:
            ar, ac = self._hazard_active
            hz_act[ar, ac] = 1.0
        for (orr, occ) in self._obj_pos_list:
            obj[orr, occ] = 1.0
        for (r, c) in self._pos:
            agents[r, c] = 1.0
        for g in (self.cfg.goals if len(self.cfg.goals) == self.cfg.n_agents else [self.cfg.goals[0]]):
            r, c = g
            goals[r, c] = 1.0

        return np.stack([obstacles, hz, hz_act, obj, agents, goals], axis=-1)


    def _ego_crop(self, center: Coord, planes: np.ndarray, k: int) -> np.ndarray:
        assert k % 2 == 1
        pad = k // 2
        H, W, C = planes.shape
        r0, c0 = center
        out = np.zeros((k, k, C), dtype=planes.dtype)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out[dr + pad, dc + pad, :] = planes[rr, cc, :]
        return out

    def _build_actor_obs(self) -> List[Union[Coord, np.ndarray]]:
        planes = self._feature_stack_global()
        obs_list: List[Union[Coord, np.ndarray]] = []
        for rc in self._pos:
            if self.cfg.actor_obs_mode == "coords":
                obs_list.append(rc)
            else:
                ego = self._ego_crop(rc, planes, self.cfg.ego_k)
                # append one more channel marking "self" at center
                marker = np.zeros((self.cfg.ego_k, self.cfg.ego_k, 1), dtype=ego.dtype)
                marker[self.cfg.ego_k // 2, self.cfg.ego_k // 2, 0] = 1.0
                obs_list.append(np.concatenate([ego, marker], axis=-1))
        return obs_list

    def _get_obs(self):
        actor_obs = self._build_actor_obs()
        critic_obs = self._feature_stack_global() if self.cfg.critic_obs_mode == "grid" else None
        return actor_obs, critic_obs

    # ---------------- core api ----------------
    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = list(self.cfg.starts)
        self._done_agents = [False] * self.cfg.n_agents
        self._obj_pos_list = list(self.cfg.object_starts) if self.cfg.has_objects else []
        self._obj_delivered = [False] * len(self._obj_pos_list) if self.cfg.has_objects else []

        # If you want the hazard fixed: keep sample_active_hazard_each_episode=False.
        # If True, we resample every episode. If False and none picked yet, pick once now.
        if self.cfg.sample_active_hazard_each_episode:
            self._sample_active_hazard_cell()
        elif self._hazard_active is None and self._hazard_cells_all:
            self._sample_active_hazard_cell()

        return self._get_obs(), {"t": self._t, "hazard_active": self._hazard_active}

    def step(self, actions: Union[List[int], Dict[int, int]]):
        if isinstance(actions, dict):
            act_list = [actions[i] for i in range(self.cfg.n_agents)]
        else:
            assert len(actions) == self.cfg.n_agents
            act_list = list(actions)

        self._t += 1
        info = {"t": self._t, "terminated_by": None}
        terminated = False
        truncated = False

        rewards_per_agent = [self.cfg.step_cost] * self.cfg.n_agents
        new_positions: List[Coord] = list(self._pos)

        # propose moves with slip
        a_eff_list: List[int] = [0] * self.cfg.n_agents
        for i, a in enumerate(act_list):
            if self._done_agents[i]:
                a_eff_list[i] = 0
                continue
            a_eff = a
            if self.cfg.slip_prob > 0.0 and self.rng.uniform() < self.cfg.slip_prob:
                a_eff = int(self.rng.integers(0, 5))   # was 0..4
            a_eff_list[i] = a_eff

            dr, dc = MultiAgentGridWorld.ACTIONS[a_eff]
            nr, nc = self._pos[i][0] + dr, self._pos[i][1] + dc
            cand = (nr, nc)

            blocked = (not self.in_bounds(cand)) or self._occupied_any(cand)

            if blocked:
                # If the block is an OBJECT, defer to push logic (no invalid penalty).
                if any(cand == opos for opos in self._obj_pos_list):
                    new_positions[i] = self._pos[i]
                else:
                    rewards_per_agent[i] += self.cfg.invalid_move_penalty
                    new_positions[i] = self._pos[i]
            else:
                new_positions[i] = cand


        if self.cfg.block_on_collision:
            # map proposed cell -> list of agents wanting it
            where: Dict[Coord, List[int]] = {}
            for i, cand in enumerate(new_positions):
                if self._done_agents[i]:
                    continue
                where.setdefault(cand, []).append(i)

            goal_cells = set(self.cfg.goals if len(self.cfg.goals) == self.cfg.n_agents
                            else [self.cfg.goals[0]])

            for cand, idxs in where.items():
                if len(idxs) <= 1:
                    continue

                # allow stacking only on agent goal cells if enabled
                if self.cfg.allow_multi_agent_on_goal and cand in goal_cells:
                    continue

                # prefer current occupant if someone is staying on the cell
                current_occupants = [i for i in idxs if self._pos[i] == cand]
                if current_occupants:
                    winner = min(current_occupants)       # deterministic tie-break
                else:
                    winner = min(idxs)                    # deterministic among movers

                # all losers bounce back to their old cells and get penalized
                for j in idxs:
                    if j == winner:
                        continue
                    rewards_per_agent[j] += self.cfg.invalid_move_penalty
                    new_positions[j] = self._pos[j]

            # prevent head-on swaps (A↔B)
            for i in range(self.cfg.n_agents):
                if self._done_agents[i]:
                    continue
                for j in range(i + 1, self.cfg.n_agents):
                    if self._done_agents[j]:
                        continue
                    if new_positions[i] == self._pos[j] and new_positions[j] == self._pos[i]:
                        # break tie deterministically; j loses
                        rewards_per_agent[j] += self.cfg.invalid_move_penalty
                        new_positions[j] = self._pos[j]

        # commit positions
        self._pos = new_positions

        # object pushing
        # --- object pushing (attempt/success detection, movement, and costs) ---
        if self.cfg.has_objects and self._obj_pos_list:
            obj_to_pushers: Dict[int, List[Tuple[int, Coord]]] = {}
            pushing_attempted: List[bool] = [False] * self.cfg.n_agents
            pushing_succeeded: List[bool] = [False] * self.cfg.n_agents

            # detect push attempts using effective actions (after slip)
            for i, a_eff in enumerate(a_eff_list):
                if self._done_agents[i]:
                    continue
                dr, dc = MultiAgentGridWorld.ACTIONS[a_eff]
                ahead = (self._pos[i][0] + dr, self._pos[i][1] + dc)
                for oi, opos in enumerate(self._obj_pos_list):
                    # optional: freeze objects already on the goal
                    if (self.cfg.object_goal is not None) and (opos == self.cfg.object_goal):
                        continue
                    if ahead == opos:
                        pushing_attempted[i] = True
                        obj_to_pushers.setdefault(oi, []).append((i, (dr, dc)))
                        break

            # resolve object movements
            for oi, pushers in obj_to_pushers.items():
                # optional: safety freeze here too
                if (self.cfg.object_goal is not None) and (self._obj_pos_list[oi] == self.cfg.object_goal):
                    continue

                allow_push = True
                if self.cfg.require_two_pushers and len(pushers) < 2:
                    allow_push = False
                if not allow_push:
                    continue

                # all pushers must agree on direction
                dirs = {d for _, d in pushers}
                if len(dirs) != 1:
                    continue

                (dr, dc) = next(iter(dirs))
                cur = self._obj_pos_list[oi]
                tgt = (cur[0] + dr, cur[1] + dc)

                # target must be in-bounds and free (stacking rule handled by _occupied_any)
                if (not self.in_bounds(tgt)) or (tgt in self._pos) or self._occupied_any(tgt):
                    continue

                # move object
                self._obj_pos_list[oi] = tgt

                # choose exactly one pusher to step into the object's previous cell
                prev = cur
                winner = min(p for p, _ in pushers)  # deterministic tie-break
                pushing_succeeded[winner] = True
                self._pos[winner] = prev

                # others attempted but didn't succeed
                for i, _ in pushers:
                    if i != winner:
                        pushing_succeeded[i] = False

            # (keep your pushing economics section as-is)


            # apply pushing economics
            if getattr(self.cfg, "use_push_step_cost", False):
                for i in range(self.cfg.n_agents):
                    if pushing_succeeded[i]:
                        # replace the base step_cost with a cheaper push_step_cost
                        rewards_per_agent[i] -= self.cfg.step_cost
                        rewards_per_agent[i] += self.cfg.push_step_cost
                    elif pushing_attempted[i]:
                        # optional extra penalty for failed shove
                        rewards_per_agent[i] += self.cfg.push_fail_penalty
            else:
                # legacy additive shaping (keep for backward compat)
                if getattr(self.cfg, "object_push_penalty", 0.0) != 0.0:
                    for i in range(self.cfg.n_agents):
                        if pushing_succeeded[i]:
                            rewards_per_agent[i] += self.cfg.object_push_penalty


        # --- one-time object delivery rewards ---
        objects_delivered_new = 0
        if self.cfg.has_objects and self.cfg.object_goal is not None and self._obj_pos_list:
            for oi, pos in enumerate(self._obj_pos_list):
                if not self._obj_delivered[oi] and pos == self.cfg.object_goal:
                    self._obj_delivered[oi] = True
                    objects_delivered_new += 1

            if objects_delivered_new > 0 and self.cfg.object_reward != 0.0:
                bonus = self.cfg.object_reward * objects_delivered_new
                # team reward: split equally across agents (keeps per-agent accounting simple)
                share = bonus / self.cfg.n_agents
                for i in range(self.cfg.n_agents):
                    rewards_per_agent[i] += share

                info["objects_delivered_new"] = objects_delivered_new

        # diagnostics
        info["objects_delivered_total"] = int(sum(self._obj_delivered)) if self._obj_delivered else 0


        # ----- catastrophe (unchanged) -----
        if self._hazard_active is not None:
            for i in range(self.cfg.n_agents):
                if self._done_agents[i]:
                    continue
                if self._pos[i] == self._hazard_active and self.rng.uniform() < self.cfg.hazard_prob:
                    for j in range(self.cfg.n_agents):
                        rewards_per_agent[j] += self.cfg.catastrophe_reward
                    terminated = True
                    info["terminated_by"] = "catastrophe"
                    break

        # ----- defer agent goal completion until all objects are currently on goal -----
        if self.cfg.has_objects and self.cfg.object_goal is not None and self._obj_pos_list:
            objects_all_on_goal_now = all(op == self.cfg.object_goal for op in self._obj_pos_list)
        else:
            objects_all_on_goal_now = True

        for i in range(self.cfg.n_agents):
            if self._done_agents[i]:
                continue
            if self.is_goal(i, self._pos[i]):
                if objects_all_on_goal_now:
                    rewards_per_agent[i] += self.cfg.goal_reward
                    self._done_agents[i] = True
                # else: standing on goal early gives no terminal reward; agent stays active

        # ----- success conditions -----
        all_agents_done = all(self._done_agents)
        if (not terminated) and objects_all_on_goal_now and all_agents_done:
            terminated = True
            info["terminated_by"] = "all_goals_and_objects"

        # time limit
        if (not terminated) and self._t >= self.cfg.max_steps:
            truncated = True
            info["terminated_by"] = "time_limit"

        obs = self._get_obs()
        reward_team = float(np.sum(rewards_per_agent))
        info["rewards_per_agent"] = rewards_per_agent
        info["object_pos_list"] = list(self._obj_pos_list)
        info["hazard_active"] = self._hazard_active
        return obs, reward_team, terminated, truncated, info

    # ---------------- rendering ----------------
    def render(self, mode: str = "human"):
        H, W = self._H, self._W
        grid = np.zeros((H, W), dtype=int)

        # obstacles
        for (r, c) in self._obstacles:
            grid[r, c] = 1

        # hazard zone and active hazard
        for (r, c) in self._hazard_cells_all:
            if grid[r, c] == 0:
                grid[r, c] = 2
        if self._hazard_active is not None:
            r, c = self._hazard_active
            grid[r, c] = 3

        # object goal then object
        if self.cfg.has_objects and self.cfg.object_goal is not None:
            gr, gc = self.cfg.object_goal
            grid[gr, gc] = 6
        if self.cfg.has_objects:
            for (orr, occ) in self._obj_pos_list:
                grid[orr, occ] = 7

        # goals
        goals = self.cfg.goals if len(self.cfg.goals) == self.cfg.n_agents else [self.cfg.goals[0]]
        for gr, gc in goals:
            grid[gr, gc] = 4

        # agents
        for (r, c) in self._pos:
            grid[r, c] = 5

        cmap = mcolors.ListedColormap([
            "white", "black", "salmon", "red",
            "green", "blue", "gold", "purple"
        ])
        bounds = [0,1,2,3,4,5,6,7,8]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                    labelbottom=False, labelleft=False)

        legend_patches = [
            mpatches.Patch(color="white", label="Empty"),
            mpatches.Patch(color="black", label="Obstacle"),
            mpatches.Patch(color="salmon", label="Hazard zone"),
            mpatches.Patch(color="red", label="Active hazard"),
            mpatches.Patch(color="green", label="Agent goal"),
            mpatches.Patch(color="blue", label="Agent"),
            mpatches.Patch(color="gold", label="Object goal"),
            mpatches.Patch(color="purple", label="Object"),
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

        if mode == "human":
            plt.show()
            return None

        elif mode == "rgb_array":
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3).copy()
            plt.close(fig)
            return img

        else:
            plt.close(fig)
            raise NotImplementedError(f"Unknown render mode: {mode}")

    def render_agent_views(self):
        """
        For each agent, show its k x k egocentric crop.
        Layers:
        - objects: solid
        - obstacles/hazard zone/active hazard/goals: light overlays
        - center crosshair marks the agent
        """
        actor_obs, _ = self._get_obs()
        n = self.cfg.n_agents
        k = self.cfg.ego_k

        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]

        for i, ego in enumerate(actor_obs):
            ax = axes[i]
            ax.axis("off")
            ax.set_title(f"agent {i} ego")

            if isinstance(ego, tuple):  # coords mode
                ax.text(0.5, 0.5, f"coords {ego}", ha="center", va="center")
                continue

            # Channel indices in your planes:
            # 0 obstacles, 1 hazard_zone, 2 hazard_active, 3 objects, 4 agents, 5 goals, 6 self-marker
            obstacles = ego[..., 0]
            hz       = ego[..., 1]
            hz_act   = ego[..., 2]
            objects  = ego[..., 3]
            # agents_channel = ego[..., 4]  # not shown to avoid "always bright center" confusion
            goals    = ego[..., 5]
            marker   = ego[..., 6]

            # Base: objects (so they are clearly visible)
            ax.imshow(objects, interpolation="nearest")

            # Light overlays
            ax.imshow(obstacles, cmap="Greys", alpha=0.25, interpolation="nearest")
            ax.imshow(hz,       cmap="Oranges", alpha=0.20, interpolation="nearest")
            ax.imshow(hz_act,   cmap="Reds",    alpha=0.35, interpolation="nearest")
            ax.imshow(goals,    cmap="Greens",  alpha=0.25, interpolation="nearest")

            # Center crosshair (self)
            mid = k // 2
            ax.plot([mid-0.5, mid+0.5], [mid, mid], lw=1.0, color="black")
            ax.plot([mid, mid], [mid-0.5, mid+0.5], lw=1.0, color="black")

            # Tiny legend
            patches = [
                mpatches.Patch(color="black", alpha=0.25, label="obstacle"),
                mpatches.Patch(color="orange", alpha=0.20, label="hazard zone"),
                mpatches.Patch(color="red", alpha=0.35, label="active hazard"),
                mpatches.Patch(color="green", alpha=0.25, label="agent goal"),
                mpatches.Patch(color="blue", alpha=1.00,  label="object (base)"),
            ]
            ax.legend(handles=patches, fontsize=8, loc="lower right")

        plt.show()


    def render_cumulative_observability(self):
        """
        Show the union of what agents can actually perceive right now, given:
        - ego window size (ego_k),
        - their current positions,
        - and the actor visibility toggles for hazard layers.
        """
        planes = self._feature_stack_global()  # world-frame planes (0..5 as documented)
        H, W = self._H, self._W
        k = self.cfg.ego_k
        pad = k // 2

        # Union-of-visibility mask in world coordinates
        visible = np.zeros((H, W), dtype=np.float32)
        for (r0, c0) in self._pos:
            rmin, rmax = max(0, r0 - pad), min(H - 1, r0 + pad)
            cmin, cmax = max(0, c0 - pad), min(W - 1, c0 + pad)
            visible[rmin:rmax + 1, cmin:cmax + 1] = 1.0

        # What agents could actually see under current toggles:
        vis_planes = planes.copy()
        # mask by visibility footprint
        vis_planes *= visible[..., None]

        # ----- rendering -----
        fig, ax = plt.subplots(figsize=(6, 6))

        # faint obstacles everywhere, but only vivid inside visible region (multiply by visible)
        ax.imshow(planes[..., 0] * visible, cmap="Greys", alpha=0.35, interpolation="nearest")

        # hazard zone (if included) and active hazard (if included), only where visible
        ax.imshow(vis_planes[..., 1], cmap="Oranges", alpha=0.25, interpolation="nearest")  # hazard zone
        ax.imshow(vis_planes[..., 2], cmap="Reds",    alpha=0.55, interpolation="nearest")  # active hazard

        # objects and goals that are visible to at least one agent
        ax.imshow(vis_planes[..., 3], cmap="Blues",   alpha=0.65, interpolation="nearest")  # objects
        ax.imshow(vis_planes[..., 5], cmap="Greens",  alpha=0.40, interpolation="nearest")  # goals

        # outline the visibility union for clarity
        ax.imshow(visible, cmap="viridis", alpha=0.12, interpolation="nearest")

        # grid styling
        ax.set_title("cumulative observability (what actors can actually see now)")
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        legend_patches = [
            mpatches.Patch(color="grey",   alpha=0.35, label="obstacles (visible portion)"),
            mpatches.Patch(color="orange", alpha=0.25, label="hazard zone (if included)"),
            mpatches.Patch(color="red",    alpha=0.55, label="active hazard (if included)"),
            mpatches.Patch(color="blue",   alpha=0.65, label="objects (visible)"),
            mpatches.Patch(color="green",  alpha=0.40, label="goals (visible)"),
            mpatches.Patch(color="purple", alpha=0.00, label=""),  # spacer
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.show()


# -----------------------------------------------------------------------------
# Factory for aligned, template-based client configs with sensible defaults
# -----------------------------------------------------------------------------
def make_aligned_client_cfg(
    n_agents: int = 3,
    H: int = 10,
    W: int = 10,
    obstacle_template_id: Optional[str] = "WarehouseA",
    # this input is ignored below because we compute a centered 2x2 zone
    hazard_bounds: Tuple[int, int, int, int] = (2, 6, 4, 8),
    starts: Optional[Sequence[Coord]] = None,
    goals: Optional[Sequence[Coord]] = None,
    hazard_prob: float = 0.05,
    hazard_loss: float = 120.0,
    seed: Optional[int] = 42,
) -> MultiAgentGridConfig:
    if starts is None:
        starts = [(H - 1, 0), (0, 0), (H - 1, W - 1)][:n_agents]
    if goals is None:
        goals = [(0, 5)]  # shared agent-goal

    # --- compute a centered 2x2 hazard zone (inclusive bounds) ---
    r0 = (H // 2) - 1
    r1 = (H // 2)
    c0 = (W // 2) - 1
    c1 = (W // 2)
    centered_2x2_bounds = (r0, c0, r1, c1)  # e.g., for 10x10 -> (4,4)-(5,5)

    # --- add 3 extra obstacles (in addition to the template) ---
    # chosen to avoid starts, goals, object, and the hazard 2x2 region
    extra_obstacles: List[Coord] = [(1, 1), (8, 8), (4, 1)]

    cfg = MultiAgentGridConfig(
        H=H,
        W=W,
        starts=starts,
        goals=goals,
        # keep any template shelves AND add three extra obstacles
        obstacles=extra_obstacles,
        obstacle_template_id=obstacle_template_id,

        # force the centered 2x2 hazard zone
        hazard_zone_cells=None,
        hazard_zone_bounds=centered_2x2_bounds,

        hazard_prob=hazard_prob,
        hazard_loss=hazard_loss,
        step_cost=-0.2,
        goal_reward=+100.0,
        catastrophe_reward=-hazard_loss,
        invalid_move_penalty=-0.1,
        max_steps=120,
        n_agents=n_agents,
        slip_prob=0.05,
        sample_active_hazard_each_episode=False,
        block_on_collision=True,
        critic_obs_mode="grid",
        actor_obs_mode="ego",
        ego_k=7,
        has_objects=True,
        object_starts=((6, 1), (2, 1), (5, 8)),  # 3 objects by default
        object_goal=(9, 5),
        require_two_pushers=False,
        object_push_penalty=0.0,
        seed=seed,
    )
    cfg.validate()
    return cfg

