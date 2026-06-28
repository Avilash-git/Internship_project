"""
PPO WAIT-vs-REPAIR supervisor on top of the frozen HybridAttentionUNet + guided A*.

Conventions taken DIRECTLY from attention_agent__13_.ipynb:
  - grid: 1 = FREE, 0 = OBSTACLE
  - U-Net input: 3 channels [grid, start_mask, goal_mask]  (create_input_tensor)
  - model: HybridAttentionUNet, sigmoid already in forward -> output IS P[n] in [0,1]
  - planner: guided_astar, f = g + h + lambda*(1 - P[n]), lambda=10, 4-connected
  - weights file: best_attention_astar_unet.pth

PPO actions:  0 = WAIT (follow path: advance if clear, else hold, wait_counter += 1)
              1 = REPAIR (call guided_astar from current pos, same P[n], goal unchanged)

Agent NEVER sees the obstacle. Only signals it sees: blocked flag + wait_counter.
Reward = negative real-time-to-goal: every step costs the same (waiting == moving),
so a REPAIR detour prices ITSELF in extra steps; planner compute is a tiny term.
=> temporary block: few wait steps cheaper than the detour  -> WAIT wins
   permanent block: endless waiting vs finite detour          -> REPAIR wins
"""

import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Optional

from adaptive_astar import ProbabilityGuidedAdaptiveAStar

Cell = Tuple[int, int]


# ======================================================================
# CONFIG
# ======================================================================
@dataclass
class Config:
    grid_size: int = 40
    obstacle_density: float = 0.25
    min_manhattan: int = 30
    lam: float = 10.0                 # lambda_weight in guided A*

    # episode behavior mixture (static walls ALWAYS present; this is obstacle type)
    p_permanent: float = 0.55         # parks on path forever -> REPAIR correct
    p_temporary: float = 0.40         # blocks D steps then clears -> WAIT correct
    p_none: float = 0.05              # never blocks the path
    temp_block_min: int = 1
    temp_block_max: int = 4

    # reward (negative real time to goal)
    step_time_cost: float = 1.0       # cost of ONE timestep (waiting OR moving)
    per_node_time_s: float = 4.2e-6   # ~0.39ms / 92 nodes  (compute time per A* node)
    planner_dt_s: float = 0.1         # real seconds represented by one timestep
    goal_reward: float = 50.0
    stuck_penalty: float = 20.0
    no_path_penalty: float = 20.0
    fixed_repair_penalty: float = 4.0  # LOWERED 5->4: makes REPAIR a bit cheaper so the agent
                                       # repairs permanent blocks more readily (collapsing the bad
                                       # "wait-to-cap" basin behind unstable 75%-vs-20% runs), while
                                       # staying ABOVE the ~5-step cost of waiting out a temporary
                                       # (temp_block_max=5, free within grace) so temporary -> WAIT
                                       # is preserved. Verified by value arithmetic.
                                       # decision-time repair cost; > temp_block_max (4) so
                                       # temporary -> WAIT, but << cost of waiting out a
                                       # permanent block (hits the cap) so permanent -> REPAIR.
                                       # the DETOUR itself is priced naturally by the per-step
                                       # cost as the agent walks the longer rerouted path.

    max_wait: int = 35                # blocked this long w/o repair -> stuck/fail. kept well
                                      # above temp_block_max (4) so temporary blocks can be
                                      # waited out, but low enough that the cost of NOT repairing
                                      # a permanent block is felt soon -> permanent learns to REPAIR.
    max_steps: int = 400


# ======================================================================
# guided A*  (verbatim cost from the notebook; grid 1=free/0=obstacle)
# returns dict: success, path, expanded_nodes, planning_time(ms)
# ======================================================================
def guided_astar(grid, start, goal, prob_map, lambda_weight=10.0):
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    expanded = 0

    while open_set:
        _, cur = heapq.heappop(open_set)
        expanded += 1
        if cur == goal:
            path = []
            node = cur
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return {"success": True, "path": path,
                    "expanded_nodes": expanded}
        for dx, dy in directions:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 0:           # obstacle
                continue
            nb = (nx, ny)
            tg = g_score[cur] + 1
            if nb not in g_score or tg < g_score[nb]:
                g_score[nb] = tg
                p = float(prob_map[nx, ny])
                f = tg + h(nb, goal) + lambda_weight * (1.0 - p)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = cur
    return {"success": False, "path": [], "expanded_nodes": expanded}


# ======================================================================
# PROBABILITY MAP PROVIDERS
# ======================================================================
class SyntheticProbMap:
    """Fallback so the PPO pipeline runs WITHOUT the trained .pth.
       Blurs the guided/plain A* path into a [0,1] map. grid: 1=free/0=obstacle."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, grid, start, goal) -> np.ndarray:
        n = grid.shape[0]
        flat = np.ones((n, n), np.float32)
        res = guided_astar(grid, start, goal, flat, self.cfg.lam)
        prob = np.full((n, n), 0.05, np.float32)
        if not res["success"]:
            return np.where(grid == 1, 0.5, 0.0).astype(np.float32)
        for (r, c) in res["path"]:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        prob[rr, cc] = max(prob[rr, cc], 0.99 - 0.18 * (abs(dr) + abs(dc)))
        prob[grid == 0] = 0.0
        return prob


# ----- the real model, classes copied verbatim from the notebook -----
def _build_unet_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HybridResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        def forward(self, x):
            identity = self.skip(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + identity)

    class HybridResidualBottleneck(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 5, padding=2)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        def forward(self, x):
            identity = self.skip(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + identity)

    class HybridAttentionGate(nn.Module):
        def __init__(self, g_ch, x_ch, inter_ch):
            super().__init__()
            self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
            self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
            self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())
        def forward(self, g, x):
            g1 = self.Wg(g); x1 = self.Wx(x)
            if g1.shape[2:] != x1.shape[2:]:
                g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            psi = F.relu(g1 + x1)
            alpha = self.psi(psi)
            return x * alpha

    class HybridAttentionUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = HybridResidualBlock(3, 32);  self.pool1 = nn.MaxPool2d(2)
            self.enc2 = HybridResidualBlock(32, 64); self.pool2 = nn.MaxPool2d(2)
            self.enc3 = HybridResidualBlock(64, 128)
            self.bottleneck = HybridResidualBottleneck(128, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.att3 = HybridAttentionGate(128, 128, 64)
            self.dec3 = HybridResidualBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.att2 = HybridAttentionGate(64, 64, 32)
            self.dec2 = HybridResidualBlock(128, 64)
            self.att1 = HybridAttentionGate(64, 32, 16)
            self.dec1 = HybridResidualBlock(64 + 32, 32)
            self.final = nn.Conv2d(32, 1, 1)
        def forward(self, x):
            e1 = self.enc1(x); p1 = self.pool1(e1)
            e2 = self.enc2(p1); p2 = self.pool2(e2)
            e3 = self.enc3(p2)
            b = self.bottleneck(e3)
            d3 = self.up3(b)
            e3a = self.att3(d3, e3)
            if e3a.shape[2:] != d3.shape[2:]:
                e3a = F.interpolate(e3a, size=d3.shape[2:], mode='bilinear', align_corners=False)
            d3 = self.dec3(torch.cat([d3, e3a], 1))
            d2 = self.up2(d3)
            e2a = self.att2(d2, e2)
            if e2a.shape[2:] != d2.shape[2:]:
                e2a = F.interpolate(e2a, size=d2.shape[2:], mode='bilinear', align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2a], 1))
            e1a = self.att1(d2, e1)
            if e1a.shape[2:] != d2.shape[2:]:
                e1a = F.interpolate(e1a, size=d2.shape[2:], mode='bilinear', align_corners=False)
            d1 = self.dec1(torch.cat([d2, e1a], 1))
            return torch.sigmoid(self.final(d1))

    return HybridAttentionUNet


class UNetProbMap:
    """Loads the frozen HybridAttentionUNet from best_attention_astar_unet.pth and
       produces P[n] exactly as the notebook does. grid: 1=free/0=obstacle."""
    def __init__(self, cfg: Config, weights_path: str = "best_attention_astar_unet.pth",
                 device: str = None, model=None):
        import torch
        self.cfg = cfg
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            UNet = _build_unet_classes()
            model = UNet()
            state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    def __call__(self, grid, start, goal) -> np.ndarray:
        torch = self.torch
        s_mask = np.zeros_like(grid, np.float32); s_mask[start] = 1.0
        g_mask = np.zeros_like(grid, np.float32); g_mask[goal] = 1.0
        x = np.stack([grid.astype(np.float32), s_mask, g_mask], 0)[None]  # (1,3,H,W)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x).to(self.device))
        return pred[0, 0].cpu().numpy().astype(np.float32)


# ======================================================================
# MAP GENERATOR  (matches notebook: density .25, corners, manhattan>=30)
# ======================================================================
class MapGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, rng: np.random.Generator):
        n = self.cfg.grid_size
        for _ in range(500):
            grid = np.ones((n, n), np.uint8)
            grid[rng.random((n, n)) < self.cfg.obstacle_density] = 0
            start = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
            goal = (int(rng.integers(n - 3, n)), int(rng.integers(n - 3, n)))
            if grid[start] != 1 or grid[goal] != 1:
                continue
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < self.cfg.min_manhattan:
                continue
            res = guided_astar(grid, start, goal,
                               np.ones((n, n), np.float32), self.cfg.lam)
            if res["success"]:
                return grid, start, goal
        raise RuntimeError("could not generate solvable map")


# ======================================================================
# DYNAMIC OBSTACLE  (agent never sees it; only its blocking effect)
#   block cell on the agent path; permanent = forever, temporary = D steps
#   then clears and never blocks again (monotonic, no reverse).
# ======================================================================
@dataclass
class Obstacle:
    behavior: str
    block_cell: Optional[Cell]
    block_duration: int
    triggered: bool = False
    counter: int = 0
    cleared: bool = False

    def occupied_cell(self):
        if self.behavior == 'none' or self.cleared:
            return None
        return self.block_cell

    def notify_blocked_step(self):
        if self.behavior == 'none' or self.cleared:
            return
        self.triggered = True
        self.counter += 1
        if self.behavior == 'temporary' and self.counter >= self.block_duration:
            self.cleared = True


def make_obstacle(cfg, path, rng):
    r = rng.random()
    if r < cfg.p_permanent:
        behavior = 'permanent'
    elif r < cfg.p_permanent + cfg.p_temporary:
        behavior = 'temporary'
    else:
        behavior = 'none'
    if behavior == 'none' or len(path) < 6:
        return Obstacle('none', None, 0)
    idx = int(rng.integers(int(0.3 * len(path)), int(0.75 * len(path))))
    bc = path[idx]
    if behavior == 'permanent':
        return Obstacle('permanent', bc, 10 ** 9)
    D = int(rng.integers(cfg.temp_block_min, cfg.temp_block_max + 1))
    return Obstacle('temporary', bc, D)


# ======================================================================
# ENVIRONMENT
# ======================================================================
class PathRepairEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config = None, prob_provider=None, fixed_scenario=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.prob_provider = prob_provider or SyntheticProbMap(self.cfg)
        self.planner = ProbabilityGuidedAdaptiveAStar(self.cfg.lam)
        self.map_gen = MapGenerator(self.cfg)
        self.fixed_scenario = fixed_scenario
        n = self.cfg.grid_size
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (4, n, n), np.float32),   # free, prob, agent, goal
            "vec": spaces.Box(0.0, 1.0, (4,), np.float32),         # blocked, wait, dist, next_p
        })

    # ---- helpers ----
    def _occ_with_obstacle(self):
        grid = self.grid.copy()
        cell = self.obstacle.occupied_cell()
        if cell is not None and self.obstacle.triggered:
            grid[cell] = 0          # obstacle = blocked
        return grid

    def _next_cell(self):
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return None

    def _is_blocked(self):
        nxt = self._next_cell()
        if nxt is None:
            return False
        cell = self.obstacle.occupied_cell()
        return cell is not None and nxt == cell

    def _build_obs(self):
        n = self.cfg.grid_size
        a = np.zeros((n, n), np.float32); a[self.agent] = 1.0
        gl = np.zeros((n, n), np.float32); gl[self.goal] = 1.0
        grid = np.stack([self.grid.astype(np.float32), self.prob_map, a, gl])
        blocked = 1.0 if self._is_blocked() else 0.0
        wait_norm = min(self.wait_counter / self.cfg.max_wait, 1.0)
        dist = abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])
        dist_norm = dist / (2 * n)
        nxt = self._next_cell()
        next_p = float(self.prob_map[nxt]) if nxt is not None else 0.0
        vec = np.array([blocked, wait_norm, dist_norm, next_p], np.float32)
        return {"grid": grid.astype(np.float32), "vec": vec}

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self._np_random
        if self.fixed_scenario is not None:
            sc = self.fixed_scenario
            self.grid = sc["grid"].copy(); self.start = sc["start"]; self.goal = sc["goal"]
            self.prob_map = sc["prob"].copy(); self.path = list(sc["path"])
            self.obstacle = Obstacle(**sc["obstacle_kwargs"])
        else:
            self.grid, self.start, self.goal = self.map_gen.generate(rng)
            self.prob_map = self.prob_provider(self.grid, self.start, self.goal)
            self.planner.reset_cache()          # new map / new P[n] -> fresh h-cache
            res = self.planner.search(self.grid, self.start, self.goal, self.prob_map)
            if not res["success"]:
                return self.reset(seed=int(rng.integers(1 << 30)))
            self.path = res["path"]
            self.obstacle = make_obstacle(self.cfg, self.path, rng)
        self.agent = self.start
        self.path_index = 0
        self.wait_counter = 0
        self.steps = 0
        self.n_repairs = 0
        self.n_waits = 0
        self.steps_advanced = 0      # WAIT actions that moved forward (true movement)
        self.blocked_wait_steps = 0  # WAIT actions that held still because blocked (true waiting)
        self.repair_expansions = []
        return self._build_obs(), {}

    def step(self, action):
        cfg = self.cfg
        self.steps += 1
        reward = -cfg.step_time_cost          # every timestep costs the same
        terminated = truncated = False
        info = {}
        blocked_before = self._is_blocked()

        if action == 1:                        # REPAIR
            self.n_repairs += 1
            grid = self._occ_with_obstacle()
            # replan from CURRENT agent pos, goal unchanged, SAME prob map (no U-Net re-run).
            # the planner reuses its cached h-values from this episode -> fewer expansions.
            res = self.planner.search(grid, self.agent, self.goal, self.prob_map)
            self.repair_expansions.append(res["expanded_nodes"])
            # repair cost = fixed decision-time overhead (breaks sloppy near-zero-detour
            #   repairs; felt immediately at the decision). The DETOUR is NOT charged here
            #   again -- it is priced naturally by the per-step cost as the agent walks the
            #   longer rerouted path, so charging it now too would double-count.
            # + tiny compute term (~0.39ms, negligible, kept for honesty).
            reward -= cfg.fixed_repair_penalty
            reward -= res["expanded_nodes"] * cfg.per_node_time_s / cfg.planner_dt_s
            self.wait_counter = 0
            if not res["success"]:
                reward -= cfg.no_path_penalty
                terminated = True; info["result"] = "no_path"
                return self._build_obs(), reward, terminated, truncated, info
            self.path = res["path"]; self.path_index = 0
        else:                                  # WAIT = follow path
            self.n_waits += 1
            if blocked_before:
                self.wait_counter += 1
                self.blocked_wait_steps += 1     # true waiting (held still, blocked)
                self.obstacle.notify_blocked_step()
            else:
                nxt = self._next_cell()
                if nxt is not None:
                    self.agent = nxt
                    self.path_index += 1
                    self.wait_counter = 0
                    self.steps_advanced += 1      # true forward movement

        if self.agent == self.goal:
            reward += cfg.goal_reward; terminated = True; info["result"] = "success"
        elif self.wait_counter >= cfg.max_wait:
            reward -= cfg.stuck_penalty; truncated = True; info["result"] = "stuck"
        elif self.steps >= cfg.max_steps:
            truncated = True; info["result"] = "timeout"

        info.update(n_repairs=self.n_repairs, n_waits=self.n_waits,
                    steps_advanced=self.steps_advanced,
                    blocked_wait_steps=self.blocked_wait_steps,
                    behavior=self.obstacle.behavior)
        return self._build_obs(), reward, terminated, truncated, info
