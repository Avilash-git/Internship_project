"""
compare_classic_vs_ours.py
==========================
HEADLINE COMPARISON for the paper:

    Classic A* guided by prob map        (the BASE-PAPER planner)
                vs
    Adaptive A* + PPO with prob map      (OUR model)

Metric = NODE EXPANSIONS (lower = better), the base paper's efficiency metric.

WHY THIS IS THE RIGHT COMPARISON
--------------------------------
* Classic guided A*  : plain A* (NO adaptive heuristic caching) guided by the
  U-Net probability map. In a dynamic world it has no "wait" concept, so it must
  REPLAN ON EVERY BLOCK. Each replan is a full classic-A* search.  -> baseline
* Adaptive A* + PPO  : OUR system. Probability-Guided *Adaptive* A* (reuses the
  heuristic across replans, so later searches are cheaper) PLUS a PPO supervisor
  that decides WAIT vs REPAIR, so it skips replans on transient obstacles.

So OUR model has TWO sources of node-expansion saving over the classic baseline:
  (1) adaptive caching makes each replan cheaper, and
  (2) PPO avoids unnecessary replans entirely (waits through temporaries).

The classic baseline calls guided_astar() directly (no caching); our model runs
through the env, which uses ProbabilityGuidedAdaptiveAStar. Both report
expanded_nodes, so the comparison is apples-to-apples on the headline metric.

SCENARIOS (10 environments each by default):
  STATIC   : no moving obstacles (just the U-Net + A* path, one plan)
  DYNAMIC  : mixed temporaries + permanents (the full wait/repair problem)

USAGE (Colab, model trained & saved, `prov` in scope):
    from stable_baselines3 import PPO
    model = PPO.load("ppo_dynamic_v8")
    from compare_classic_vs_ours import run_comparison
    run_comparison(model, provider=prov, n=10)
"""

import numpy as np
from ppo_repair import guided_astar
from ppo_dynamic_v8 import DynamicConfigV8, DynamicObstacleEnvV8


# ======================================================================
# CLASSIC guided-A* baseline: replan-on-block using PLAIN guided_astar
# (no adaptive caching). This mirrors what the base-paper planner must
# do in a dynamic world -- a fresh classic search on every block.
# ======================================================================
def _run_classic_repair_on_block(env, seed):
    """Drive the SAME env, but on every block do a CLASSIC guided_astar replan
       (not the env's adaptive planner). We count classic A* expanded_nodes."""
    obs, info = env.reset(seed=seed)
    total_nodes = 0
    n_replans = 0
    lam = env.cfg.lam

    for _ in range(env.cfg.max_steps + 5):
        if env._is_blocked():
            # CLASSIC replan: plain guided A* on the current occupancy
            grid = env._occ_with_active()
            res = guided_astar(grid, env.agent, env.goal, env.prob_map, lam)
            total_nodes += res["expanded_nodes"]
            n_replans += 1
            # apply the same REPAIR action so the env state advances identically
            obs, r, term, trunc, info = env.step(1)
        else:
            obs, r, term, trunc, info = env.step(0)
        if term or trunc:
            break

    success = info.get("result") == "success"
    return success, n_replans, total_nodes


# ======================================================================
# OUR model: Adaptive A* + PPO (env uses ProbabilityGuidedAdaptiveAStar)
# ======================================================================
def _run_ours(model, env, seed):
    obs, info = env.reset(seed=seed)
    for _ in range(env.cfg.max_steps + 5):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        if term or trunc:
            break
    success = info.get("result") == "success"
    n_repairs = info.get("n_repairs", 0)
    total_nodes = int(sum(env.repair_expansions))   # adaptive A* expansions
    return success, n_repairs, total_nodes


# ======================================================================
# one scenario table
# ======================================================================
def _run_scenario(cfg, label, model, provider, seeds):
    # two SEPARATE env instances with the SAME cfg so seeds line up but the
    # adaptive planner's cache is never shared with the classic runs
    env_classic = DynamicObstacleEnvV8(cfg, prob_provider=provider)
    env_ours = DynamicObstacleEnvV8(cfg, prob_provider=provider)

    classic = [_run_classic_repair_on_block(env_classic, s) for s in seeds]
    ours = [_run_ours(model, env_ours, s) for s in seeds]

    def agg(res):
        succ = 100.0 * np.mean([a for a, _, _ in res])
        calls = np.mean([c for _, c, _ in res])
        nodes = np.mean([n for _, _, n in res])
        return succ, calls, nodes

    c_succ, c_calls, c_nodes = agg(classic)
    o_succ, o_calls, o_nodes = agg(ours)

    print("=" * 78)
    print(f"{label}   ({len(seeds)} environments)")
    print("=" * 78)
    print(f"{'method':<34}{'success':>9}{'A*_calls':>11}{'node_expansions':>17}")
    print("-" * 78)
    print(f"{'Classic A* (prob-guided)':<34}{c_succ:>8.0f}%{c_calls:>11.2f}"
          f"{c_nodes:>17.1f}")
    print(f"{'Adaptive A* + PPO (ours)':<34}{o_succ:>8.0f}%{o_calls:>11.2f}"
          f"{o_nodes:>17.1f}")
    print("-" * 78)

    if c_nodes > 0:
        saving = 100.0 * (c_nodes - o_nodes) / c_nodes
        verdict = "FEWER nodes (good)" if o_nodes < c_nodes else "MORE nodes (worse)"
        print(f"Node-expansion saving of OURS vs Classic: {saving:+.1f}%  ({verdict})")
    if o_succ < c_succ - 5:
        print(f"  CAUTION: ours success ({o_succ:.0f}%) is below classic "
              f"({c_succ:.0f}%) -- a node saving from failing early is not a real win.")
    print("=" * 78)
    print()
    return {"classic": (c_succ, c_calls, c_nodes),
            "ours": (o_succ, o_calls, o_nodes)}


# ======================================================================
# full comparison: STATIC + DYNAMIC, 10 environments each
# ======================================================================
def run_comparison(model, provider=None, n=10):
    seeds = list(range(7000, 7000 + n))

    print("\n" + "#" * 78)
    print("# NODE-EXPANSION COMPARISON  --  Classic guided A*  vs  Adaptive A* + PPO")
    print("# metric: node expansions per environment (lower = better)")
    print("#" * 78 + "\n")

    # ---- STATIC: no moving obstacles (single plan, no replanning needed) ----
    cs = DynamicConfigV8()
    cs.n_temp_min = 0; cs.n_temp_max = 0
    cs.n_perm_min = 0; cs.n_perm_max = 0
    static = _run_scenario(cs, "STATIC  (no moving obstacles)", model, provider, seeds)

    # ---- DYNAMIC: full mixed problem (temporaries + permanents) ----
    cd = DynamicConfigV8()   # defaults: 2-3 temp, 1-2 perm
    dynamic = _run_scenario(cd, "DYNAMIC  (temporaries + permanents)",
                            model, provider, seeds)

    # ---- honest overall summary ----
    print("#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    sc, so = static["classic"], static["ours"]
    dc, do = dynamic["classic"], dynamic["ours"]
    print(f"STATIC  : classic {sc[2]:.1f} nodes  vs  ours {so[2]:.1f} nodes")
    print(f"DYNAMIC : classic {dc[2]:.1f} nodes  vs  ours {do[2]:.1f} nodes")
    print()
    if do[2] < dc[2] and do[0] >= dc[0] - 5:
        print("=> On DYNAMIC, our Adaptive A* + PPO expands FEWER nodes than classic")
        print("   guided A* at comparable success. This is the efficiency contribution:")
        print("   adaptive caching + learned wait/repair reduce planning computation in")
        print("   dynamic environments.")
    else:
        print("=> On DYNAMIC, ours does NOT beat classic on node expansions. Read the")
        print("   numbers honestly and report what they show rather than the hoped result.")
    print("#" * 78)
