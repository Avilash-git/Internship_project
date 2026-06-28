"""
visualize_expansion.py
======================
Two figures for the paper:

(A) MAP with dynamic-obstacle START positions
    - black  = static walls
    - red    = dynamic obstacle starting cells
    - green  = start,  blue = goal,  grey line = the U-Net+A* path

(B) NODE-EXPANSION FOOTPRINT over 5 maps, side by side:
    - LEFT  : Classic A* guided by prob map         (the base-paper planner)
    - RIGHT : Adaptive A* with prob map (ours)
    Each shaded cell = a cell the search EXPANDED. The shaded AREA is the
    node-expansion count made visible -- a smaller/tighter footprint means
    fewer node expansions (the efficiency the paper is about).

These figures DO NOT change any of your model files. The two A* variants below
are self-contained, instrumented copies that ALSO record which cells they expand
(the plain count is identical to your real planners; we just also collect the
cells so they can be drawn).

USAGE (Colab, `prov` in scope):
    from visualize_expansion import show_start_positions, show_expansion_footprints
    show_start_positions(provider=prov, seed=7000)
    show_expansion_footprints(provider=prov, seeds=[7000,7001,7002,7003,7004])
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ppo_dynamic_v9 import DynamicConfigV9, DynamicObstacleEnvV9


# ======================================================================
# instrumented A* variants -- identical search to your planners, but they
# ALSO return the SET of expanded cells so we can draw the footprint.
# ======================================================================
def classic_guided_astar_cells(grid, start, goal, prob_map, lam=10.0):
    """Classic prob-guided A* (no caching) -- base-paper planner.
       Returns (success, path, expanded_cells)."""
    rows, cols = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0.0, start)]
    came_from = {}
    g = {start: 0.0}
    expanded_cells = []

    while open_set:
        _, cur = heapq.heappop(open_set)
        expanded_cells.append(cur)
        if cur == goal:
            path = []
            node = cur
            while node in came_from:
                path.append(node); node = came_from[node]
            path.append(start); path.reverse()
            return True, path, expanded_cells
        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 0:
                continue
            nb = (nx, ny)
            tg = g[cur] + 1
            if nb not in g or tg < g[nb]:
                g[nb] = tg
                p = float(prob_map[nx, ny])
                f = tg + h(nb, goal) + lam * (1.0 - p)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = cur
    return False, [], expanded_cells


def adaptive_guided_astar_cells(grid, start, goal, prob_map, h_cache, lam=10.0):
    """Prob-guided ADAPTIVE A* (ours): uses a cached heuristic h_cache (dict) that
       lowers expansions on repeated searches to the same goal. Returns
       (success, path, expanded_cells) and UPDATES h_cache in place."""
    rows, cols = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def h(s):
        # adaptive heuristic: max(manhattan, cached) -- cache makes it sharper
        base = abs(s[0] - goal[0]) + abs(s[1] - goal[1])
        return max(base, h_cache.get(s, 0.0))

    open_set = [(0.0, start)]
    came_from = {}
    g = {start: 0.0}
    expanded_cells = []

    while open_set:
        _, cur = heapq.heappop(open_set)
        expanded_cells.append(cur)
        if cur == goal:
            path = []
            node = cur
            while node in came_from:
                path.append(node); node = came_from[node]
            path.append(start); path.reverse()
            # adaptive update: h[s] = g(goal) - g(s) for expanded cells
            g_goal = g[goal]
            for s in expanded_cells:
                if s in g:
                    h_cache[s] = max(h_cache.get(s, 0.0), g_goal - g[s])
            return True, path, expanded_cells
        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 0:
                continue
            nb = (nx, ny)
            tg = g[cur] + 1
            if nb not in g or tg < g[nb]:
                g[nb] = tg
                p = float(prob_map[nx, ny])
                f = tg + h(nb) + lam * (1.0 - p)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = cur
    return False, [], expanded_cells


# ======================================================================
# FIGURE A : dynamic obstacle START positions
# ======================================================================
def show_start_positions(provider=None, seed=7000, cfg=None):
    cfg = cfg or DynamicConfigV9()
    env = DynamicObstacleEnvV9(cfg, prob_provider=provider)
    env.reset(seed=seed)

    n = cfg.grid_size
    img = np.ones((n, n, 3))                      # white background
    # static walls -> black
    walls = (env.grid == 0)
    img[walls] = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, origin="upper")

    # dynamic obstacle START cells -> red squares
    for o in env.obstacles:
        sr, sc = o.trajectory[0]
        ax.scatter(sc, sr, c="red", s=120, marker="s",
                   edgecolors="darkred", zorder=3)

    # start (green) and goal (blue)
    ax.scatter(env.start[1], env.start[0], c="lime", s=160, marker="o",
               edgecolors="black", zorder=4, label="start")
    ax.scatter(env.goal[1], env.goal[0], c="blue", s=160, marker="*",
               edgecolors="black", zorder=4, label="goal")

    # the planned path -> grey line
    if len(env.path) > 1:
        pr = [p[0] for p in env.path]
        pc = [p[1] for p in env.path]
        ax.plot(pc, pr, c="grey", lw=1.5, alpha=0.7, zorder=2, label="A* path")

    ax.set_title(f"Map (seed {seed})  |  black = static walls, "
                 f"red = dynamic obstacle START positions")
    ax.legend(loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    n_temp = sum(o.category == "temporary" for o in env.obstacles)
    n_perm = sum(o.category == "permanent" for o in env.obstacles)
    print(f"seed {seed}: {n_temp} temporary + {n_perm} permanent dynamic obstacles "
          f"(red squares mark their starting cells)")


# ======================================================================
# FIGURE B : node-expansion FOOTPRINT, 5 maps, classic vs ours
# ======================================================================
def show_expansion_footprints(provider=None, seeds=None, cfg=None):
    cfg = cfg or DynamicConfigV9()
    seeds = seeds or [7000, 7001, 7002, 7003, 7004]
    n = cfg.grid_size

    fig, axes = plt.subplots(len(seeds), 2, figsize=(9, 4.2 * len(seeds)))
    if len(seeds) == 1:
        axes = axes.reshape(1, 2)

    for row, seed in enumerate(seeds):
        env = DynamicObstacleEnvV9(cfg, prob_provider=provider)
        env.reset(seed=seed)
        grid = env._occ_with_active()            # grid with obstacles blocked
        start, goal, prob, lam = env.start, env.goal, env.prob_map, cfg.lam

        # CLASSIC expansion footprint
        _, c_path, c_cells = classic_guided_astar_cells(grid, start, goal, prob, lam)
        # OURS (adaptive) expansion footprint -- fresh cache for a fair single search
        _, o_path, o_cells = adaptive_guided_astar_cells(grid, start, goal, prob,
                                                          {}, lam)

        for col, (title, cells, path, color) in enumerate([
            ("Classic A* (prob-guided)\nbase paper", c_cells, c_path, "orange"),
            ("Adaptive A* + prob map\nours",         o_cells, o_path, "deepskyblue"),
        ]):
            ax = axes[row, col]
            img = np.ones((n, n, 3))
            img[env.grid == 0] = [0, 0, 0]                      # walls black
            ax.imshow(img, origin="upper")
            # expanded cells -> translucent shade (the footprint)
            if cells:
                er = [c[0] for c in cells]; ec = [c[1] for c in cells]
                ax.scatter(ec, er, c=color, s=18, alpha=0.35, marker="s", zorder=1)
            # path -> dark line
            if len(path) > 1:
                pr = [p[0] for p in path]; pc = [p[1] for p in path]
                ax.plot(pc, pr, c="black", lw=1.8, zorder=2)
            # start/goal
            ax.scatter(start[1], start[0], c="lime", s=90, marker="o",
                       edgecolors="black", zorder=3)
            ax.scatter(goal[1], goal[0], c="blue", s=110, marker="*",
                       edgecolors="black", zorder=3)
            ax.set_title(f"seed {seed}  |  {title}\nnodes expanded = {len(cells)}",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # honest numeric summary
    print("Node-expansion footprint (single search per map, obstacles in place):")
    for seed in seeds:
        env = DynamicObstacleEnvV9(cfg, prob_provider=provider)
        env.reset(seed=seed)
        grid = env._occ_with_active()
        _, _, c_cells = classic_guided_astar_cells(grid, env.start, env.goal,
                                                   env.prob_map, cfg.lam)
        _, _, o_cells = adaptive_guided_astar_cells(grid, env.start, env.goal,
                                                    env.prob_map, {}, cfg.lam)
        print(f"  seed {seed}: classic {len(c_cells):4d} nodes   |   "
              f"ours {len(o_cells):4d} nodes")
