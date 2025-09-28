from typing import Dict, Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from configs.config_templates.case_study_2_1 import MultiAgentGridConfig, Coord


class MultiAgentGridWorld:
    ACTIONS = [(-1,0),(0,+1),(+1,0),(0,-1),(0,0)]

    def __init__(self, cfg: MultiAgentGridConfig):
        self.cfg = cfg
        self.rng: np.random.Generator = np.random.default_rng(cfg.seed)
        self._H, self._W = cfg.H, cfg.W
        self._obstacles = set(cfg.obstacles)
        self._t: int = 0

        self._pos: List[Coord] = list(cfg.starts)
        self._done_agents: List[bool] = [False] * cfg.n_agents
        self._hazard_cells_all: List[Coord] = cfg._resolve_hazard_zone_cells()
        self._hazard_active: Optional[Coord] = None
        self._obj_pos_list: List[Coord] = list(cfg.object_starts) if cfg.has_objects else []
        self._obj_delivered: List[bool] = [False] * len(self._obj_pos_list) if self.cfg.has_objects else []

        if self._hazard_cells_all and not cfg.sample_active_hazard_each_episode:
            self._sample_active_hazard_cell()


    def in_bounds(self, rc: Coord) -> bool:
        r, c = rc
        return 0 <= r < self._H and 0 <= c < self._W

    def is_obstacle(self, rc: Coord) -> bool:
        return rc in self._obstacles

    def is_goal(self, idx: int, rc: Coord) -> bool:
        g = self.cfg.goals[idx] if len(self.cfg.goals) == self.cfg.n_agents else self.cfg.goals[0]
        return rc == g

    def _occupied_static(self, rc: Coord) -> bool:
        return (rc in self._obstacles)

    def _occupied_any(self, rc: Coord) -> bool:
        if rc in self._obstacles:
            return True
        if getattr(self.cfg, "allow_objects_stack_on_object_goal", False) and self.cfg.object_goal is not None:
            if rc == self.cfg.object_goal:
                return False

        if any(rc == opos for opos in getattr(self, "_obj_pos_list", [])):
            return True
        return False


    def _sample_active_hazard_cell(self):
        if self._hazard_cells_all:
            pick = self.rng.choice(np.array(self._hazard_cells_all))
            self._hazard_active = (int(pick[0]), int(pick[1]))
        else:
            self._hazard_active = None

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

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = list(self.cfg.starts)
        self._done_agents = [False] * self.cfg.n_agents
        self._obj_pos_list = list(self.cfg.object_starts) if self.cfg.has_objects else []
        self._obj_delivered = [False] * len(self._obj_pos_list) if self.cfg.has_objects else []

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
                if any(cand == opos for opos in self._obj_pos_list):
                    new_positions[i] = self._pos[i]
                else:
                    rewards_per_agent[i] += self.cfg.invalid_move_penalty
                    new_positions[i] = self._pos[i]
            else:
                new_positions[i] = cand


        if self.cfg.block_on_collision:
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

                if self.cfg.allow_multi_agent_on_goal and cand in goal_cells:
                    continue

                # prefer current occupant if someone is staying on the cell
                current_occupants = [i for i in idxs if self._pos[i] == cand]
                if current_occupants:
                    winner = min(current_occupants)     
                else:
                    winner = min(idxs)                   

                for j in idxs:
                    if j == winner:
                        continue
                    rewards_per_agent[j] += self.cfg.invalid_move_penalty
                    new_positions[j] = self._pos[j]

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

                    if (self.cfg.object_goal is not None) and (opos == self.cfg.object_goal):
                        continue
                    if ahead == opos:
                        pushing_attempted[i] = True
                        obj_to_pushers.setdefault(oi, []).append((i, (dr, dc)))
                        break

            # resolve object movements
            for oi, pushers in obj_to_pushers.items():
                if (self.cfg.object_goal is not None) and (self._obj_pos_list[oi] == self.cfg.object_goal):
                    continue

                allow_push = True
                if self.cfg.require_two_pushers and len(pushers) < 2:
                    allow_push = False
                if not allow_push:
                    continue

                dirs = {d for _, d in pushers}
                if len(dirs) != 1:
                    continue

                (dr, dc) = next(iter(dirs))
                cur = self._obj_pos_list[oi]
                tgt = (cur[0] + dr, cur[1] + dc)

                if (not self.in_bounds(tgt)) or (tgt in self._pos) or self._occupied_any(tgt):
                    continue

                # move object
                self._obj_pos_list[oi] = tgt

                prev = cur
                winner = min(p for p, _ in pushers)
                pushing_succeeded[winner] = True
                self._pos[winner] = prev

                for i, _ in pushers:
                    if i != winner:
                        pushing_succeeded[i] = False

            # apply pushing economics
            if getattr(self.cfg, "use_push_step_cost", False):
                for i in range(self.cfg.n_agents):
                    if pushing_succeeded[i]:
                        
                        rewards_per_agent[i] -= self.cfg.step_cost
                        rewards_per_agent[i] += self.cfg.push_step_cost
                    elif pushing_attempted[i]:
                        
                        rewards_per_agent[i] += self.cfg.push_fail_penalty
            else:
                if getattr(self.cfg, "object_push_penalty", 0.0) != 0.0:
                    for i in range(self.cfg.n_agents):
                        if pushing_succeeded[i]:
                            rewards_per_agent[i] += self.cfg.object_push_penalty


        # object delivery rewards 
        objects_delivered_new = 0
        if self.cfg.has_objects and self.cfg.object_goal is not None and self._obj_pos_list:
            for oi, pos in enumerate(self._obj_pos_list):
                if not self._obj_delivered[oi] and pos == self.cfg.object_goal:
                    self._obj_delivered[oi] = True
                    objects_delivered_new += 1

            if objects_delivered_new > 0 and self.cfg.object_reward != 0.0:
                bonus = self.cfg.object_reward * objects_delivered_new
                share = bonus / self.cfg.n_agents
                for i in range(self.cfg.n_agents):
                    rewards_per_agent[i] += share

                info["objects_delivered_new"] = objects_delivered_new

        # diagnostics
        info["objects_delivered_total"] = int(sum(self._obj_delivered)) if self._obj_delivered else 0


        # catastrophe 
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

        # success conditions
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

        # object goal then objects
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

        # agents (draw last)
        for (r, c) in self._pos:
            grid[r, c] = 5

        cmap = mcolors.ListedColormap([
            "white", "black", "salmon", "red",
            "green", "blue", "gold", "purple"
        ])
        bounds = [0,1,2,3,4,5,6,7,8]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # ---- Figure with a dedicated legend column (no overlay, no cropping) ----
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        # Wider figure: left 4/5 for map, right 1/5 for legend
        fig = plt.figure(figsize=(7.5, 6.0), dpi=120)  # keep size fixed for GIF stability
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")  # legend-only pane

        im = ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                    labelbottom=False, labelleft=False)

        legend_patches = [
            mpatches.Patch(color="white",  label="Empty"),
            mpatches.Patch(color="black",  label="Obstacle"),
            mpatches.Patch(color="salmon", label="Hazard zone"),
            mpatches.Patch(color="red",    label="Active hazard"),
            mpatches.Patch(color="green",  label="Agent goal"),
            mpatches.Patch(color="blue",   label="Agent"),
            mpatches.Patch(color="gold",   label="Object goal"),
            mpatches.Patch(color="purple", label="Object"),
        ]
        # Legend *inside its own axes*; nothing hangs outside the canvas
        ax_leg.legend(handles=legend_patches, loc="center left", frameon=True)

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

            obstacles = ego[..., 0]
            hz       = ego[..., 1]
            hz_act   = ego[..., 2]
            objects  = ego[..., 3]
            goals    = ego[..., 5]
            marker   = ego[..., 6]

            ax.imshow(objects, interpolation="nearest")

            ax.imshow(obstacles, cmap="Greys", alpha=0.25, interpolation="nearest")
            ax.imshow(hz,       cmap="Oranges", alpha=0.20, interpolation="nearest")
            ax.imshow(hz_act,   cmap="Reds",    alpha=0.35, interpolation="nearest")
            ax.imshow(goals,    cmap="Greens",  alpha=0.25, interpolation="nearest")

            mid = k // 2
            ax.plot([mid-0.5, mid+0.5], [mid, mid], lw=1.0, color="black")
            ax.plot([mid, mid], [mid-0.5, mid+0.5], lw=1.0, color="black")

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
        planes = self._feature_stack_global()
        H, W = self._H, self._W
        k = self.cfg.ego_k
        pad = k // 2

        # union-of-visibility mask in world coordinates
        visible = np.zeros((H, W), dtype=np.float32)
        for (r0, c0) in self._pos:
            rmin, rmax = max(0, r0 - pad), min(H - 1, r0 + pad)
            cmin, cmax = max(0, c0 - pad), min(W - 1, c0 + pad)
            visible[rmin:rmax + 1, cmin:cmax + 1] = 1.0

        vis_planes = planes.copy()
        vis_planes *= visible[..., None]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(planes[..., 0] * visible, cmap="Greys", alpha=0.35, interpolation="nearest")

        ax.imshow(vis_planes[..., 1], cmap="Oranges", alpha=0.25, interpolation="nearest")  # hazard zone
        ax.imshow(vis_planes[..., 2], cmap="Reds",    alpha=0.55, interpolation="nearest")  # active hazard

        ax.imshow(vis_planes[..., 3], cmap="Blues",   alpha=0.65, interpolation="nearest")  # objects
        ax.imshow(vis_planes[..., 5], cmap="Greens",  alpha=0.40, interpolation="nearest")  # goals

        ax.imshow(visible, cmap="viridis", alpha=0.12, interpolation="nearest")

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
