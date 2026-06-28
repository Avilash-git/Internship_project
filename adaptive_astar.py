"""
Probability-Guided Adaptive A*
==============================
Standard Adaptive A* (Koenig & Likhachev) run on a U-Net probability-weighted
cost metric.

  movement cost into cell n :  step_cost(n) = 1 + lambda*(1 - P[n])
  g_new                     :  g_current + step_cost
  f                         :  g_new + h          <- standard form, P is in g
  heuristic update (cached) :  h[s] = g(goal) - g(s)   for every expanded s
  next-search heuristic     :  h(s) = max(Manhattan(s,goal), cached_h[s])

Why this is sound:
  - every step costs >= 1, so Manhattan stays an admissible/consistent lower
    bound on the prob-weighted cost -> Adaptive A*'s update is valid.
  - P is part of g (a real static cost), so g(goal)-g(s) is a true
    prob-weighted goal-distance estimate. The update formula is NOT modified;
    probability flows into the cached heuristic automatically through g.
  - across replans P is fixed and obstacles only get ADDED (cost increases,
    never decreases), which is exactly the condition basic Adaptive A* needs.

Grid convention (matches the notebook): 1 = FREE, 0 = OBSTACLE.
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict

Cell = Tuple[int, int]
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ProbabilityGuidedAdaptiveAStar:
    def __init__(self, lam: float = 10.0):
        self.lam = lam
        self.h_cache: Dict[Cell, float] = {}   # learned heuristics, persist across replans
        self.last_expansions: int = 0
        self.last_path_cost: float = 0.0

    # ---- episode boundary: call once per new map / new P[n] ----
    def reset_cache(self):
        self.h_cache = {}

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _h(self, s: Cell, goal: Cell) -> float:
        # use the more-informed cached value when available
        return max(self._manhattan(s, goal), self.h_cache.get(s, 0.0))

    def search(self, grid: np.ndarray, start: Cell, goal: Cell,
               prob_map: np.ndarray) -> dict:
        """
        grid: 1=free, 0=obstacle. prob_map: [0,1] from the frozen U-Net.
        Returns dict(success, path, expanded_nodes, path_cost).
        """
        rows, cols = grid.shape
        lam = self.lam

        if grid[start] == 0 or grid[goal] == 0:
            self.last_expansions = 0
            self.last_path_cost = 0.0
            return {"success": False, "path": [], "expanded_nodes": 0, "path_cost": 0.0}

        open_set = []
        g_score: Dict[Cell, float] = {start: 0.0}
        came_from: Dict[Cell, Cell] = {}
        f0 = g_score[start] + self._h(start, goal)
        heapq.heappush(open_set, (f0, start))
        closed = set()
        expanded = 0

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            closed.add(cur)
            expanded += 1

            if cur == goal:
                break

            cg = g_score[cur]
            for dx, dy in _DIRS:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if grid[nx, ny] == 0:            # obstacle
                    continue
                nb = (nx, ny)
                if nb in closed:
                    continue
                # ---- probability folded into the MOVEMENT COST (g) ----
                step_cost = 1.0 + lam * (1.0 - float(prob_map[nx, ny]))
                ng = cg + step_cost
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb] = ng
                    came_from[nb] = cur
                    f = ng + self._h(nb, goal)   # standard f = g + h
                    heapq.heappush(open_set, (f, nb))

        self.last_expansions = expanded

        if goal not in g_score:
            self.last_path_cost = 0.0
            return {"success": False, "path": [], "expanded_nodes": expanded, "path_cost": 0.0}

        # ---- Adaptive A* heuristic update (formula UNCHANGED) ----
        # h[s] = g(goal) - g(s) for every expanded cell; keep the max so cached
        # heuristics are monotonically non-decreasing (stay consistent under the
        # cost increases caused by added obstacles across replans).
        g_goal = g_score[goal]
        for s in closed:
            self.h_cache[s] = max(self.h_cache.get(s, 0.0), g_goal - g_score[s])

        # reconstruct path
        path: List[Cell] = []
        node = goal
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        path.reverse()

        self.last_path_cost = g_goal
        return {"success": True, "path": path,
                "expanded_nodes": expanded, "path_cost": g_goal}
