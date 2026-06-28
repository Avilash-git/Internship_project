
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ppo_repair import (Config, SyntheticProbMap, UNetProbMap, MapGenerator,
                        guided_astar)
from adaptive_astar import ProbabilityGuidedAdaptiveAStar

Cell = Tuple[int, int]


# ======================================================================
# CONFIG
# ======================================================================
@dataclass
class DynamicConfigV9(Config):
   # ---- obstacle counts per episode ----

    # Existing single temporary obstacles
    n_temp_min: int = 4
    n_temp_max: int = 6

    # NEW: Convoy temporary obstacles
    n_convoy_min: int = 1
    n_convoy_max: int = 2

    # Each convoy contains this many obstacle members
    convoy_length_min: int = 3
    convoy_length_max: int = 5
    convoy_spacing: int = 1          # cells between members
    convoy_speed: int = 1   

    # Existing permanent obstacles
    n_perm_min: int = 1
    n_perm_max: int = 2

    # (Stage 2)
    # Background movers
    n_background_min: int = 3
    n_background_max: int = 5

    background_traj_min: int = 8
    background_traj_max: int = 15 

    # ---- temporary trajectory geometry ----
    walk_along_min: int = 3        # cells walked ALONG the path (sustained block)
    walk_along_max: int = 5

    # ---- observation sizes ----
    vel_history_len: int = 5
    pos_history_len: int = 5
    lookahead_K: int = 10
    max_expected_obstacles: int = 4

    # ---- reward (bare costs; see header) ----
    step_cost: float = 0.05            # magnitude; applied as -step_cost
    block_wait_penalty: float = 0.20   # magnitude; (legacy flat penalty, unused
                                       #   when use_step_penalty=True below)
    fixed_repair_cost: float = 1.50    # magnitude; flat repair toll (raised 1.20->1.50
                                       #   to lean the agent toward WAITING through
                                       #   temporaries; early-repair penalty untouched
                                       #   so permanent handling stays balanced)
    no_path_penalty: float = 20.0
    goal_reward: float = 50.0
    collision_penalty: float = 10.0

    # ---- STEP-FUNCTION wait penalty (sharp blocked_duration knife) ----
    # Cheap while inside the linger window (<= walk_along_max steps) so the agent
    # is patient on temporaries; brutal past it so a never-clearing permanent block
    # forces a REPAIR almost immediately. linger_grace is bound to walk_along_max
    # in __init__ so the crossover always sits exactly at the edge of the linger
    # range (change the linger length and the boundary follows automatically).
    use_step_penalty: bool = True
    wait_penalty_cheap: float = 0.10   # magnitude, blocked_duration <= grace
    wait_penalty_steep: float = 0.35   # magnitude, blocked_duration >  grace
                                       # (0.80 was too brutal -> created an
                                       #  "always REPAIR" attractor; 0.35 still
                                       #  rises past the cheap window so permanents
                                       #  -> repair, but doesn't terrify the agent
                                       #  away from waiting through temporaries.)

    # ---- ACTION OVERRIDE + EARLY-REPAIR PENALTY (force use of blocked_duration) ----
    # Override: choosing REPAIR when nothing is blocking is useless (no detour to
    # find), so it is converted to a follow-path step and given a tiny slap so the
    # policy learns to stop suggesting it.
    override_repair_when_clear: bool = True
    useless_repair_penalty: float = 0.10
    # Early-repair penalty: punish "panic-repairing" before the agent has waited
    # through the cheap linger window. This forces the timer to rise -> the agent
    # must read blocked_duration to decide -> patience becomes financially optimal.
    # NOTE: this is engineered (reward-shaped) patience, not emergent; framed
    # honestly as a temporal prior, it validates the sensory-motor loop.
    use_early_repair_penalty: bool = True
    early_repair_threshold: int = 3    # was 4; lowered so permanents become
                                       #   "normal-price" to repair sooner (the
                                       #   threshold-4 version delayed repair too
                                       #   long and crashed permanent success).
    early_repair_penalty: float = 0.50 # was 1.00; softened to relieve the squeeze
                                       #   on permanents (1.00 -> 64% perm success,
                                       #   47% mixed; too scared to repair). 0.50
                                       #   still discourages bd=0 panic-repair on
                                       #   temporaries without freezing permanents.

    # ---- episode bounds ----
    max_wait: int = 35
    max_steps: int = 500

    # ---- generation safety ----
    max_gen_attempts: int = 120
    min_path_len: int = 14
    remap_max_dist: int = 5            # dormant obstacle dropped if its crossing
                                       #   cell is >this from any new-path cell


# ======================================================================
# DYNAMIC OBSTACLE
# ======================================================================
@dataclass
class DynObstacle:
    """One moving obstacle. `category` is INTERNAL ONLY (never in the obs).

       temporary : walks its whole trajectory then leaves (pos -> None)
       permanent : walks its trajectory, then HALTS on the final cell forever
    """
    category: str                       # 'temporary' | 'permanent'| 'background'
    trajectory: List[Cell]              # ordered cells it will visit
    launch_at_index: int                # agent path index that triggers launch
    halted_behavior: bool = False       # True for permanent (freeze at end)
    # runtime
    traj_pos: int = -1                  # -1 = dormant; else index into trajectory
    halted: bool = False                # permanent obstacle that has frozen
    pos: Optional[Cell] = None          # current occupied cell
    prev_pos: Optional[Cell] = None     # occupied cell on the PREVIOUS step

    # ---- lifecycle ----
    def is_active(self) -> bool:
        """FIX 1: a halted permanent obstacle is ALWAYS active (a fixture),
           even though it has 'finished' its trajectory."""
        if self.halted:
            return True
        return self.pos is not None

    def has_reached_end(self) -> bool:
        return self.traj_pos >= len(self.trajectory) - 1

    def current_cell(self) -> Optional[Cell]:
        """The cell this obstacle occupies for occupancy purposes."""
        if self.halted:
            return self.trajectory[-1]
        return self.pos

    def launch_if_ready(self, agent_index: int):
        if self.traj_pos == -1 and agent_index >= self.launch_at_index:
            self.traj_pos = 0
            self.prev_pos = self.trajectory[0]
            self.pos = self.trajectory[0]

    def advance(self):
        """Move ONE trajectory cell per call (after launch).
           temporary: walks to the end then leaves (pos -> None).
           permanent: walks to the end then freezes (halted=True)."""
        if self.traj_pos == -1:           # dormant
            return
        if self.halted:                   # frozen permanent: prev == pos
            self.prev_pos = self.pos
            return
        self.prev_pos = self.pos          # remember for velocity
        if self.traj_pos + 1 < len(self.trajectory):
            self.traj_pos += 1
            self.pos = self.trajectory[self.traj_pos]
            if self.halted_behavior and self.traj_pos == len(self.trajectory) - 1:
                self.halted = True        # reached final cell -> freeze (permanent)
        else:
            # ran off the end of the trajectory
            if self.halted_behavior:
                self.halted = True
            else:
                self.pos = None           # temporary leaves the board

    def velocity(self) -> Tuple[float, float]:
        """OBSERVED velocity = current - previous (0,0 if dormant/gone/frozen)."""
        cur = self.current_cell()
        if cur is None or self.prev_pos is None:
            return (0.0, 0.0)
        return (float(cur[0] - self.prev_pos[0]),
                float(cur[1] - self.prev_pos[1]))


# ======================================================================
# OBSTACLE GENERATION  (FIX 3: perpendicular offset, dynamic perp on exit)
# ======================================================================
def _in_bounds_free(cell, grid):
    r, c = cell
    n = grid.shape[0]
    return 0 <= r < n and 0 <= c < n and grid[r, c] == 1


def generate_temporary_obstacle(cfg, grid, path, rng):
    """
    Temporary obstacle (LINGER): crosses perpendicular onto a path cell, STOPS
    DEAD on it for 3-5 steps (velocity -> 0,0 during the pause), then continues
    straight through to the other side (velocity > 0 again).

    WHY linger (not continuous motion): if a temporary obstacle always moved, a
    trivial rule "velocity != 0 -> WAIT" would solve everything and PPO would be
    pointless. By stopping, the temporary obstacle's local features (velocity
    history, blocked_now) become IDENTICAL to a permanent blocker's for the first
    few steps:

        step 3 of a block:
          temporary lingerer : vel_hist [(1,0),(0,0),(0,0)], blocked_duration 3
          permanent blocker  : vel_hist [(1,0),(0,0),(0,0)], blocked_duration 3

    The agent CANNOT decide from velocity alone -> it must use blocked_duration
    trend + global detour context to break the tie. That is exactly the POMDP that
    justifies RL over an if-else baseline.

    Clean "cross -> stop -> continue" shape (no fold-backs): approach 2 cells in,
    linger on the crossing cell, exit 2 cells out the OTHER side.
    """
    L = len(path)
    lo = int(0.20 * L)
    hi = int(0.60 * L)
    if hi <= lo:
        return None
    crossing_idx = int(rng.integers(lo, hi))
    if crossing_idx + 1 >= L:
        return None
    crossing_cell = path[crossing_idx]
    linger_steps = int(rng.integers(cfg.walk_along_min, cfg.walk_along_max + 1))  # 3-5

    # perpendicular approach vector from the LOCAL path heading
    p_start = path[crossing_idx]
    p_next = path[crossing_idx + 1]
    path_dir = (p_next[0] - p_start[0], p_next[1] - p_start[1])
    perp = (-path_dir[1], path_dir[0])

    traj: List[Cell] = []
    r0, c0 = crossing_cell

    # PHASE A: approach perpendicular (2 cells out -> 1 cell out)
    traj.append((r0 + perp[0] * 2, c0 + perp[1] * 2))
    traj.append((r0 + perp[0] * 1, c0 + perp[1] * 1))

    # PHASE B: LINGER on the crossing cell for N steps (velocity becomes 0,0)
    for _ in range(linger_steps):
        traj.append(crossing_cell)

    # PHASE C: continue straight through to the OTHER side (velocity > 0 again)
    traj.append((r0 - perp[0] * 1, c0 - perp[1] * 1))
    traj.append((r0 - perp[0] * 2, c0 - perp[1] * 2))

    # boundary + wall check
    for cell in traj:
        if not _in_bounds_free(cell, grid):
            return None

    launch_idx = max(0, crossing_idx - 2)

    return DynObstacle(category='temporary', trajectory=traj,
                       launch_at_index=launch_idx, halted_behavior=False)


def generate_convoy_obstacles(cfg, grid, path, rng):
    """
    Generate one moving convoy represented as multiple synchronized
    temporary DynObstacle objects.

    The convoy behaves like a moving train:
        Member 1
            ↓
        Member 2
            ↓
        Member 3
            ↓
        ...

    Every member follows the SAME trajectory.
    They launch at different times so the convoy maintains formation.
    """

    L = len(path)
    lo = int(0.20 * L)
    hi = int(0.60 * L)

    if hi <= lo:
        return None

    crossing_idx = int(rng.integers(lo, hi))

    if crossing_idx + 1 >= L:
        return None

    crossing_cell = path[crossing_idx]

    # -------------------------------------------------------------
    # Compute local path direction (same logic as temporary obstacle)
    # -------------------------------------------------------------
    p_start = path[crossing_idx]
    p_next = path[crossing_idx + 1]

    path_dir = (
        p_next[0] - p_start[0],
        p_next[1] - p_start[1]
    )

    perp = (
        -path_dir[1],
        path_dir[0]
    )

    r0, c0 = crossing_cell

    # -------------------------------------------------------------
    # Leader trajectory
    #
    # Approach
    # Cross
    # Leave
    # -------------------------------------------------------------
    leader_traj = [
        (r0 + perp[0] * 2, c0 + perp[1] * 2),
        (r0 + perp[0] * 1, c0 + perp[1] * 1),
        crossing_cell,
        (r0 - perp[0] * 1, c0 - perp[1] * 1),
        (r0 - perp[0] * 2, c0 - perp[1] * 2),
    ]

    # Boundary / obstacle validation
    for cell in leader_traj:
        if not _in_bounds_free(cell, grid):
            return None

    convoy_length = int(
        rng.integers(
            cfg.convoy_length_min,
            cfg.convoy_length_max + 1
        )
    )

    launch_idx = max(0, crossing_idx - 2)

    convoy = []

    for member in range(convoy_length):

        # Build a shifted trajectory so every member follows
        # the leader while maintaining formation.
        shifted_traj = []

        # Stay at the first position while waiting for the leader
        for _ in range(member * cfg.convoy_spacing):
            shifted_traj.append(leader_traj[0])

        # Follow the leader trajectory
        shifted_traj.extend(leader_traj)

        convoy.append(
            DynObstacle(
                category="temporary",
                trajectory=shifted_traj,
                launch_at_index=launch_idx,      # everyone launches together
                halted_behavior=False
            ))


    return convoy

def generate_background_obstacle(cfg, grid, path, rng):
    """
    Background movers create dynamic traffic.

    They are NOT intentionally generated to block the robot.
    They simply patrol through free space, making the environment
    appear alive.
    """

    n = grid.shape[0]

    # ---------------------------------------------------------
    # Choose a random free starting cell that is NOT near path
    # ---------------------------------------------------------
    attempts = 50

    while attempts > 0:
        attempts -= 1

        r = int(rng.integers(1, n - 1))
        c = int(rng.integers(1, n - 1))

        if grid[r, c] == 0:
            continue

        # keep background movers away from planned path
        near_path = False
        for pr, pc in path:
            if abs(pr - r) + abs(pc - c) <= 3:
                near_path = True
                break

        if near_path:
            continue

        start = (r, c)
        break
    else:
        return None

    # ---------------------------------------------------------
    # Random motion direction
    # ---------------------------------------------------------
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    dr, dc = directions[int(rng.integers(len(directions)))]

    traj_length = int(
        rng.integers(
            cfg.background_traj_min,
            cfg.background_traj_max + 1
        )
    )

    trajectory = []

    r, c = start

    for _ in range(traj_length):

        if not _in_bounds_free((r, c), grid):
            break

        trajectory.append((r, c))

        nr = r + dr
        nc = c + dc

        # Bounce from walls
        if not _in_bounds_free((nr, nc), grid):
            dr = -dr
            dc = -dc

            nr = r + dr
            nc = c + dc

            if not _in_bounds_free((nr, nc), grid):
                break

        r = nr
        c = nc

    if len(trajectory) < 4:
        return None

    return DynObstacle(
        category="background",
        trajectory=trajectory,
        launch_at_index=0,
        halted_behavior=False
    )
def generate_permanent_obstacle(cfg, grid, path, rng):
    """
    Permanent obstacle: approaches from the side (entry perpendicular), reaches
    a path cell, and HALTS there forever.
    """
    L = len(path)
    lo = int(0.20 * L)
    hi = int(0.70 * L)
    if hi <= lo:
        return None
    crossing_idx = int(rng.integers(lo, hi))
    if crossing_idx + 1 >= L:
        return None

    p_start = path[crossing_idx]
    p_next = path[crossing_idx + 1]
    entry_dir = (p_next[0] - p_start[0], p_next[1] - p_start[1])
    perp_entry = (-entry_dir[1], entry_dir[0])

    traj: List[Cell] = []
    r0, c0 = p_start
    traj.append((r0 + perp_entry[0] * 2, c0 + perp_entry[1] * 2))   # approach
    traj.append((r0 + perp_entry[0] * 1, c0 + perp_entry[1] * 1))   # approach
    traj.append(p_start)                                            # reach & halt

    for cell in traj:
        if not _in_bounds_free(cell, grid):
            return None

    launch_idx = max(0, crossing_idx - 2)

    return DynObstacle(category='permanent', trajectory=traj,
                       launch_at_index=launch_idx, halted_behavior=True)


# ======================================================================
# ENVIRONMENT
# ======================================================================
class DynamicObstacleEnvV9(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: DynamicConfigV9 = None, prob_provider=None):
        super().__init__()
        self.cfg = cfg or DynamicConfigV9()
        self.prob_provider = prob_provider or SyntheticProbMap(self.cfg)
        self.map_gen = MapGenerator(self.cfg)
        n = self.cfg.grid_size
        H = self.cfg.vel_history_len
        P = self.cfg.pos_history_len

        self.action_space = spaces.Discrete(2)   # 0 = WAIT, 1 = REPAIR
        vec_dim = H * 2 + P * 2 + 1 + 1 + 1 + 1   # = 24
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (6, n, n), np.float32),
            "vec":  spaces.Box(0.0, 1.0, (vec_dim,), np.float32),
        })

        # step-function wait penalty: the cheap window ends exactly at the edge of
        # the linger range, so temporaries (linger <= walk_along_max) stay cheap to
        # wait out while permanents cross into the steep zone almost immediately.
        self.linger_grace = self.cfg.walk_along_max

        self._vel_hist: List[Tuple[float, float]] = [(0.0, 0.0)] * H
        self._pos_hist: List[Tuple[float, float]] = [(0.0, 0.0)] * P

    # ---------- occupancy (FIX 1: halted permanents stay as fixtures) ----------
    def _active_cells(self) -> set:
        active = set()
        for o in self.obstacles:
            if o.is_active():
                cell = o.current_cell()
                if cell is not None:
                    active.add(cell)
        return active

    def _prev_cells(self) -> set:
        cells = set()
        for o in self.obstacles:
            if o.traj_pos != -1 and o.prev_pos is not None:
                cells.add(o.prev_pos)
        return cells

    def _occ_with_active(self):
        """Grid with every active obstacle cell blocked (for REPAIR planning)."""
        grid = self.grid.copy()
        for c in self._active_cells():
            grid[c] = 0
        return grid

    def _permanent_block_cells(self) -> set:
        """Cells permanent obstacles will occupy forever (for solvability)."""
        cells = set()
        for o in self.obstacles:
            if o.category == 'permanent':
                cells.add(o.trajectory[-1])
        return cells

    def _next_cell(self):
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return None

    def _blocking_obstacle(self):
        nxt = self._next_cell()
        if nxt is None:
            return None
        for o in self.obstacles:
            if o.is_active() and o.current_cell() == nxt:
                return o
        return None

    def _is_blocked(self) -> bool:
        return self._blocking_obstacle() is not None

    # ---------- lookahead along the planned path ----------
    def _nearest_blocking_ahead(self):
        K = self.cfg.lookahead_K
        active = {o.current_cell(): o for o in self.obstacles
                  if o.is_active() and o.current_cell() is not None}
        for j in range(1, K + 1):
            idx = self.path_index + j
            if idx >= len(self.path):
                break
            cell = self.path[idx]
            if cell in active:
                return active[cell], j
        return None, K + 1

    def _ahead_counts(self):
        K = self.cfg.lookahead_K
        ahead = set()
        for j in range(1, K + 1):
            idx = self.path_index + j
            if idx >= len(self.path):
                break
            ahead.add(self.path[idx])
        total = moving = 0
        for o in self.obstacles:
            if o.is_active() and o.current_cell() in ahead:
                total += 1
                vx, vy = o.velocity()
                if abs(vx) + abs(vy) > 0.0:
                    moving += 1
        return total, moving

    # ---------- observation ----------
    def _push_hist(self):
        """Push the nearest-blocking obstacle's velocity AND position into the
           rolling histories. If nothing is blocking ahead, push zeros / agent-
           relative neutral so the vectors stay well-defined."""
        nb, _ = self._nearest_blocking_ahead()
        if nb is not None:
            self._vel_hist.append(nb.velocity())
            cur = nb.current_cell()
            self._pos_hist.append((float(cur[0]), float(cur[1])))
        else:
            self._vel_hist.append((0.0, 0.0))
            self._pos_hist.append((0.0, 0.0))
        if len(self._vel_hist) > self.cfg.vel_history_len:
            self._vel_hist.pop(0)
        if len(self._pos_hist) > self.cfg.pos_history_len:
            self._pos_hist.pop(0)

    def _build_obs(self):
        n = self.cfg.grid_size
        a = np.zeros((n, n), np.float32); a[self.agent] = 1.0
        gl = np.zeros((n, n), np.float32); gl[self.goal] = 1.0
        cur = np.zeros((n, n), np.float32)
        for c in self._active_cells():
            cur[c] = 1.0
        prev = np.zeros((n, n), np.float32)
        for c in self._prev_cells():
            prev[c] = 1.0
        grid = np.stack([self.grid.astype(np.float32), self.prob_map,
                         a, gl, cur, prev]).astype(np.float32)

        # velocity history (normalised: components in {-1,0,1} -> {0,0.5,1})
        vh = []
        for (vx, vy) in self._vel_hist:
            vh.append((vx + 1.0) / 2.0)
            vh.append((vy + 1.0) / 2.0)
        # position history (normalised by grid size)
        ph = []
        for (px, py) in self._pos_hist:
            ph.append(px / float(n))
            ph.append(py / float(n))

        bd_norm = min(self.blocked_duration / float(self.cfg.max_wait), 1.0)
        _, dist = self._nearest_blocking_ahead()
        dist_norm = min((dist - 1) / float(self.cfg.lookahead_K), 1.0)
        total_ahead, moving_ahead = self._ahead_counts()
        total_norm = min(total_ahead / float(self.cfg.max_expected_obstacles), 1.0)
        moving_norm = min(moving_ahead / float(self.cfg.max_expected_obstacles), 1.0)

        vec = np.array(vh + ph + [bd_norm, dist_norm, total_norm, moving_norm],
                       dtype=np.float32)
        return {"grid": grid, "vec": vec}

    # ---------- FIX 2: remap dormant obstacles after a repair ----------
    def _remap_dormant_obstacles(self, new_path):
        """After REPAIR the path changes, so dormant obstacles' launch_at_index
           (relative to the OLD path) is meaningless. Re-anchor each dormant
           obstacle to the NEW path by its crossing (final-trajectory) cell. If
           that cell is far from any new-path cell, the obstacle is irrelevant on
           the new route -> retire it."""
        for o in self.obstacles:
            if o.traj_pos != -1:
                continue                      # already launched; leave it alone
            crossing_cell = o.trajectory[-1]
            if crossing_cell in new_path:
                cross_idx = new_path.index(crossing_cell)
            else:
                dists = [abs(p[0] - crossing_cell[0]) + abs(p[1] - crossing_cell[1])
                         for p in new_path]
                m = min(dists)
                if m > self.cfg.remap_max_dist:
                    o.traj_pos = len(o.trajectory)   # retire: never launches
                    o.halted = False
                    o.pos = None
                    continue
                cross_idx = dists.index(m)
            approach_steps = max(1, len(o.trajectory) - 1)
            o.launch_at_index = max(0, cross_idx - 2)

    # ---------- episode generation ----------
    def _new_episode(self, rng):
        cfg = self.cfg
        for _ in range(cfg.max_gen_attempts):
            grid, start, goal = self.map_gen.generate(rng)
            prob = self.prob_provider(grid, start, goal)
            res = guided_astar(grid, start, goal, prob, self.cfg.lam)
            if not res["success"]:
                continue
            path = res["path"]
            if len(path) < cfg.min_path_len:
                continue

            obstacles: List[DynObstacle] = []
            n_temp = int(rng.integers(cfg.n_temp_min, cfg.n_temp_max + 1))
            n_perm = int(rng.integers(cfg.n_perm_min, cfg.n_perm_max + 1))
            n_convoy = int(
                    rng.integers(
                        cfg.n_convoy_min,
                        cfg.n_convoy_max + 1
                    )
                )
            n_background = int(
                    rng.integers(
                        cfg.n_background_min,
                        cfg.n_background_max + 1
                    )
                )

            for _ in range(n_temp * 10 + 10):
                if sum(o.category == 'temporary' for o in obstacles) >= n_temp:
                    break
                o = generate_temporary_obstacle(cfg, grid, path, rng)
                if o is not None and self._traj_cells_free(o, obstacles):
                    obstacles.append(o)

            for _ in range(n_convoy * 10 + 10):

                existing_convoys = sum(
                    1 for o in obstacles
                    if getattr(o, "is_convoy_member", False)
                )

                if existing_convoys >= n_convoy:
                    break

                convoy = generate_convoy_obstacles(
                    cfg,
                    grid,
                    path,
                    rng
                )

                if convoy is None:
                    continue

                valid = True

                for member in convoy:
                    if not self._traj_cells_free(member, obstacles):
                        valid = False
                        break

                if not valid:
                    continue

                for member in convoy:
                    member.is_convoy_member = True
                    obstacles.append(member)

            # ---------------------------------------------------------
            # Generate background movers
            # ---------------------------------------------------------
            for _ in range(n_background * 10 + 10):

                if sum(
                    o.category == "background"
                    for o in obstacles
                ) >= n_background:
                    break

                o = generate_background_obstacle(
                    cfg,
                    grid,
                    path,
                    rng
                )

                if o is None:
                    continue

                if self._traj_cells_free(o, obstacles):
                    obstacles.append(o)

            for _ in range(n_perm * 10 + 10):
                if sum(o.category == 'permanent' for o in obstacles) >= n_perm:
                    break
                o = generate_permanent_obstacle(cfg, grid, path, rng)
                if o is not None and self._traj_cells_free(o, obstacles):
                    obstacles.append(o)

            # require we actually reached the intended minimums
            got_temp = sum(
                    o.category == "temporary"
                    and not getattr(o, "is_convoy_member", False)
                    for o in obstacles
                )
            got_convoy = sum(
                    1 for o in obstacles
                    if getattr(o, "is_convoy_member", False)
                )
            got_background = sum(
                    o.category == "background"
                    for o in obstacles
                )
            got_perm = sum(o.category == 'permanent' for o in obstacles)
            if (
                    got_temp < cfg.n_temp_min
                    or got_perm < cfg.n_perm_min
                    or got_convoy < cfg.n_convoy_min
                    or got_background < cfg.n_background_min
                ):
                continue

            # SOLVABILITY: with every permanent block removed, a route must exist.
            test = grid.copy()
            for c in self._perm_cells_of(obstacles):
                test[c] = 0
            chk = guided_astar(test, start, goal, prob, cfg.lam)
            if chk["success"]:
                return grid, start, goal, prob, path, obstacles
        raise RuntimeError("could not generate solvable dynamic map (v8)")

    @staticmethod
    def _perm_cells_of(obstacles):
        return {o.trajectory[-1] for o in obstacles if o.category == 'permanent'}

    @staticmethod
    def _traj_cells_free(new_obs, existing):
        """Reject a new obstacle whose trajectory cells overlap an already-placed
           obstacle's trajectory (keeps obstacles from stacking / colliding)."""
        occupied = set()
        for o in existing:
            occupied.update(o.trajectory)
        for c in new_obs.trajectory:
            if c in occupied:
                return False
        return True

    # ---------- reset ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self._np_random
        (self.grid, self.start, self.goal, self.prob_map,
         self.path, self.obstacles) = self._new_episode(rng)

        self.agent = self.start
        self.path_index = 0
        self.blocked_duration = 0
        self.wait_counter = 0
        self.steps = 0
        self.n_repairs = 0
        self.steps_advanced = 0
        self.n_collisions = 0
        self.repair_expansions = []
        self._vel_hist = [(0.0, 0.0)] * self.cfg.vel_history_len
        self._pos_hist = [(0.0, 0.0)] * self.cfg.pos_history_len

        # launch any obstacle whose trigger index is already <= 0
        for o in self.obstacles:
            o.launch_if_ready(self.path_index)
        return self._build_obs(), {}

    # ---------- world dynamics ----------
    def _advance_world(self):
        for o in self.obstacles:
            o.launch_if_ready(self.path_index)
        for o in self.obstacles:
            o.advance()

    # ---------- one step ----------
    def step(self, action):
        cfg = self.cfg
        self.steps += 1
        reward = -cfg.step_cost            # baseline per-step cost
        terminated = truncated = False
        info = {}

        # --- ACTION OVERRIDE: REPAIR with a clear path is useless (no detour to
        #     find). Convert to a follow-path step and apply a tiny slap so the
        #     policy stops proposing it. After this, action==1 only ever runs the
        #     repair branch when the agent is genuinely blocked.
        if (cfg.override_repair_when_clear and action == 1
                and not self._is_blocked()):
            action = 0
            reward -= cfg.useless_repair_penalty

        if action == 1:                                  # ---------- REPAIR ----------
            self.n_repairs += 1
            grid = self._occ_with_active()
            res = guided_astar(grid, self.agent, self.goal, self.prob_map, self.cfg.lam)
            self.repair_expansions.append(res["expanded_nodes"])
            reward -= cfg.fixed_repair_cost
            # EARLY-REPAIR PENALTY: punish giving up before the agent has waited
            # through the cheap linger window. Forces blocked_duration to rise so
            # the agent must read the timer to decide (engineered patience).
            if (cfg.use_early_repair_penalty
                    and self.blocked_duration < cfg.early_repair_threshold):
                reward -= cfg.early_repair_penalty
            self.blocked_duration = 0
            self.wait_counter = 0
            if not res["success"]:
                reward -= cfg.no_path_penalty
                terminated = True
                info["result"] = "no_path"
                self._advance_world()
                self._push_hist()
                self._fill_info(info)
                return self._build_obs(), reward, terminated, truncated, info
            # adopt new path and RE-MAP dormant obstacles to it (FIX 2)
            self.path = res["path"]
            self._remap_dormant_obstacles(self.path)
            self.path_index = (self.path.index(self.agent)
                               if self.agent in self.path else 0)
            # FIX 4: after replanning, ADVANCE one cell along the new path if the
            # next cell is clear. Without this, REPAIR never moves the agent, so a
            # policy that repeatedly picks REPAIR sits in place forever (replan ->
            # stay -> replan -> stay) until timeout. Repair = "replan AND step onto
            # the new route", which is the intended semantics and removes the
            # degenerate in-place-repair loop.
            nxt = self._next_cell()
            if nxt is not None and nxt not in self._active_cells():
                self.agent = nxt
                self.path_index += 1
                self.steps_advanced += 1

        else:                                            # ---------- WAIT / follow ----------
            nxt = self._next_cell()
            blocking = self._blocking_obstacle()
            if blocking is not None:
                # next cell occupied -> cannot step; wait, accumulate penalty.
                self.blocked_duration += 1
                self.wait_counter += 1
                if cfg.use_step_penalty:
                    # STEP-FUNCTION: cheap inside the linger window (patience on
                    # temporaries), brutal past it (forces REPAIR on permanents).
                    if self.blocked_duration <= self.linger_grace:
                        reward -= cfg.wait_penalty_cheap
                    else:
                        reward -= cfg.wait_penalty_steep
                else:
                    reward -= cfg.block_wait_penalty       # legacy flat penalty
            elif nxt is not None:
                if nxt in self._active_cells():
                    # safety guard (shouldn't trigger since blocking is None)
                    self.n_collisions += 1
                    reward -= cfg.collision_penalty
                    self.blocked_duration += 1
                    self.wait_counter += 1
                else:
                    self.agent = nxt
                    self.path_index += 1
                    self.blocked_duration = 0
                    self.wait_counter = 0
                    self.steps_advanced += 1

        # advance the moving world AFTER the agent acts
        self._advance_world()

        # collision check: did an obstacle move ONTO the agent?
        if self.agent in self._active_cells():
            self.n_collisions += 1
            reward -= cfg.collision_penalty

        self._push_hist()

        # terminal checks
        if self.agent == self.goal:
            reward += cfg.goal_reward
            terminated = True
            info["result"] = "success"
        elif self.wait_counter >= cfg.max_wait:
            reward -= cfg.no_path_penalty
            truncated = True
            info["result"] = "stuck"
        elif self.steps >= cfg.max_steps:
            truncated = True
            info["result"] = "timeout"

        self._fill_info(info)
        return self._build_obs(), reward, terminated, truncated, info

    def _fill_info(self, info):
        n_temp = sum(o.category == 'temporary' for o in self.obstacles)
        n_perm = sum(o.category == 'permanent' for o in self.obstacles)
        n_convoy = sum(
            getattr(o, "is_convoy_member", False)
            for o in self.obstacles
        )

        n_background = sum(
            o.category == "background"
            for o in self.obstacles
        )
        info.update(n_repairs=self.n_repairs,
                    steps_advanced=self.steps_advanced,
                    blocked_duration=self.blocked_duration,
                    n_collisions=self.n_collisions,
                    n_obstacles=len(self.obstacles),
                    n_temporary=n_temp, n_permanent=n_perm,n_convoy=n_convoy,n_background=n_background)