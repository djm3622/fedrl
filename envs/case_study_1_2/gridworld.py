from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


Coord = Tuple[int, int]  # (row, col)


@dataclass
class GridConfig:
    H: int                                      # grid height (rows)
    W: int                                      # grid width (cols)
    start: Coord                                # start cell
    goal: Coord                                 # goal cell
    obstacles: Iterable[Coord] = field(default_factory=list)
    # Map of catastrophic cell -> termination probability in [0,1]
    catastrophes: Dict[Coord, float] = field(default_factory=dict)

    # Rewards / costs
    step_cost: float = -1.0                     # per-step cost
    goal_reward: float = +100.0                 # reward on reaching goal
    cat_reward: float = -100.0                  # reward if catastrophe triggers

    # Penalties / limits
    invalid_move_penalty: float = -2.0          # hitting obstacle or wall (agent stays in place)
    max_steps: int = 200                        # truncate episode after this many steps

    # RNG
    seed: Optional[int] = None                  # for reproducibility

    def validate(self):
        r0, c0 = self.start
        rg, cg = self.goal
        assert 0 <= r0 < self.H and 0 <= c0 < self.W, "start out of bounds"
        assert 0 <= rg < self.H and 0 <= cg < self.W, "goal out of bounds"
        obs = set(self.obstacles)
        assert self.start not in obs, "start cannot be an obstacle"
        assert self.goal not in obs, "goal cannot be an obstacle"
        for (r, c), p in self.catastrophes.items():
            assert 0 <= r < self.H and 0 <= c < self.W, "catastrophe cell out of bounds"
            assert 0.0 <= p <= 1.0, "catastrophe probability must be in [0,1]"


class GridWorld:
    """
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    Observations:
        - obs_mode="coords": (row, col)
        - obs_mode="grid": np.ndarray[H,W,C] with stacked feature maps
    """
    ACTIONS: List[Coord] = [(-1, 0), (0, +1), (+1, 0), (0, -1)]

    def __init__(self, cfg: GridConfig, obs_mode: str = "coords"):
        cfg.validate()
        self.cfg = cfg
        self.obs_mode = obs_mode
        self.rng: np.random.Generator = np.random.default_rng(cfg.seed)
        self._obstacles = set(cfg.obstacles)
        self._cat = dict(cfg.catastrophes)
        self._pos: Coord = cfg.start
        self._t: int = 0

    # ----------------------------
    # Helpers
    # ----------------------------
    def in_bounds(self, rc: Coord) -> bool:
        r, c = rc
        return 0 <= r < self.cfg.H and 0 <= c < self.cfg.W

    def is_obstacle(self, rc: Coord) -> bool:
        return rc in self._obstacles

    def is_goal(self, rc: Coord) -> bool:
        return rc == self.cfg.goal

    def is_catastrophic_cell(self, rc: Coord) -> bool:
        return rc in self._cat

    def catastrophe_prob(self, rc: Coord) -> float:
        return self._cat.get(rc, 0.0)

    def state_index(self, rc: Optional[Coord] = None) -> int:
        if rc is None:
            rc = self._pos
        r, c = rc
        return r * self.cfg.W + c

    def _make_grid_obs(self) -> np.ndarray:
        H, W = self.cfg.H, self.cfg.W
        agent = np.zeros((H, W), dtype=np.float32)
        goal = np.zeros((H, W), dtype=np.float32)
        obstacles = np.zeros((H, W), dtype=np.float32)
        cats = np.zeros((H, W), dtype=np.float32)
        cat_prob = np.zeros((H, W), dtype=np.float32)

        r, c = self._pos
        agent[r, c] = 1.0

        gr, gc = self.cfg.goal
        goal[gr, gc] = 1.0

        for (ro, co) in self._obstacles:
            obstacles[ro, co] = 1.0

        for (rc, cc), p in self._cat.items():
            cats[rc, cc] = 1.0
            cat_prob[rc, cc] = p

        # Stack channels: (H, W, C)
        return np.stack([agent, goal, obstacles, cats, cat_prob], axis=-1)

    def _get_obs(self):
        if self.obs_mode == "coords":
            return self._pos
        elif self.obs_mode == "grid":
            return self._make_grid_obs()
        else:
            raise ValueError(f"Unknown obs_mode={self.obs_mode}")

    # ----------------------------
    # Core API
    # ----------------------------
    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pos = self.cfg.start
        self._t = 0
        return self._get_obs(), {"t": self._t}

    def step(self, action: int):
        assert 0 <= action < 4, "action must be in {0,1,2,3}"
        self._t += 1

        dr, dc = GridWorld.ACTIONS[action]
        nr, nc = self._pos[0] + dr, self._pos[1] + dc
        new_pos = (nr, nc)

        reward = self.cfg.step_cost
        terminated = False
        truncated = False
        info = {"t": self._t, "terminated_by": None}

        if not self.in_bounds(new_pos) or self.is_obstacle(new_pos):
            reward += self.cfg.invalid_move_penalty
            new_pos = self._pos
        else:
            self._pos = new_pos

        if self.is_goal(self._pos):
            reward += self.cfg.goal_reward
            terminated = True
            info["terminated_by"] = "goal"
        elif self.is_catastrophic_cell(self._pos):
            p = self.catastrophe_prob(self._pos)
            if self.rng.uniform() < p:
                reward += self.cfg.cat_reward
                terminated = True
                info["terminated_by"] = "catastrophe"

        if self._t >= self.cfg.max_steps:
            truncated = True
            info["terminated_by"] = "time_limit"

        return self._get_obs(), reward, terminated, truncated, info

    # ----------------------------
    # Rendering
    # ----------------------------

    def render(self):
        H, W = self.cfg.H, self.cfg.W
        grid = np.zeros((H, W), dtype=int)

        # Encode features as integers
        for (r, c) in self._obstacles:
            grid[r, c] = 1  # obstacle
        for (r, c) in self._cat.keys():
            grid[r, c] = 2  # catastrophe
        gr, gc = self.cfg.goal
        grid[gr, gc] = 3  # goal
        r, c = self._pos
        grid[r, c] = 4  # agent

        # Colormap
        cmap = mcolors.ListedColormap(["white", "black", "red", "green", "blue"])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap=cmap, norm=norm)

        # Grid lines
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Legend
        legend_patches = [
            mpatches.Patch(color="white", label="Empty"),
            mpatches.Patch(color="black", label="Obstacle"),
            mpatches.Patch(color="red", label="Catastrophe"),
            mpatches.Patch(color="green", label="Goal"),
            mpatches.Patch(color="blue", label="Agent"),
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.show()

