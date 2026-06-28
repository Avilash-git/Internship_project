"""
compare_node_expansions.py
==========================
NODE-EXPANSION comparison for the dynamic-obstacle wait/repair study.

The base paper's metric is NODE EXPANSIONS (how many A* nodes the planner
expands). This script measures that metric across the policies, reusing the
ALREADY-TRAINED model saved to ppo_dynamic_v9.zip -- it does NOT retrain.

WHAT IS COMPARED (each row is a different "supervisor" deciding when to replan;
the underlying planner is the SAME Probability-Guided Adaptive A* in all cases,
so the comparison isolates the effect of the WAIT-vs-REPAIR decision on total
node expansions):

  repair-on-block   : replan on EVERY block (the naive dynamic baseline -- this
                      is what guided A* alone must do, since A* has no "wait")
  if-else velocity  : a hand-crafted heuristic (wait if blocker is moving,
                      else repair) -- the simple rule PPO must beat
  PPO (trained)     : the learned supervisor (your model)

For each policy we report, per episode:
  success rate
  avg total NODE EXPANSIONS  (sum of expanded_nodes over all repairs)  <-- HEADLINE
  avg repairs                (number of A* calls)
  avg reward

HONEST NOTE: node expansions = (repairs) x (nodes per repair). Fewer repairs
usually means fewer node expansions, BUT Adaptive A* caches its heuristic across
repairs in an episode, so LATER repairs are cheaper. This script measures the
REAL summed expansions on the cached planner, so it reflects the true cost, not
an estimate. Whether PPO beats the heuristics on THIS metric is exactly what the
numbers below decide -- read them honestly.

USAGE (Colab, after the model is trained & saved, with `prov` in scope):
    from stable_baselines3 import PPO
    model = PPO.load("ppo_dynamic_v9")          # reuse the saved model
    from compare_node_expansions import compare_all
    compare_all(model, provider=prov)           # mixed + per-category tables
"""

import numpy as np
from ppo_dynamic_v9 import DynamicConfigV9, DynamicObstacleEnvV9
from train_dynamic_v9 import (_always_wait, _repair_on_block, _ifelse_velocity)


# ----------------------------------------------------------------------
# runners that ALSO return total node expansions for the episode
# (env.repair_expansions is the list of expanded_nodes per repair call)
# ----------------------------------------------------------------------
def _run_scripted_nodes(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(env.cfg.max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(env))
        total_reward += r
        if term or trunc:
            break
    result = info.get("result", "?")
    n_repairs = info.get("n_repairs", 0)
    total_nodes = int(sum(env.repair_expansions))   # <-- the headline metric
    return result, total_reward, n_repairs, total_nodes


def _run_model_nodes(model, env, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(env.cfg.max_steps + 5):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        total_reward += r
        if term or trunc:
            break
    result = info.get("result", "?")
    n_repairs = info.get("n_repairs", 0)
    total_nodes = int(sum(env.repair_expansions))
    return result, total_reward, n_repairs, total_nodes


# ----------------------------------------------------------------------
# one comparison table on a given config
# ----------------------------------------------------------------------
def _compare_on(cfg, label, model, provider, seeds):
    env = DynamicObstacleEnvV9(cfg, prob_provider=provider)

    policies = [
        ("repair-on-block",  lambda s: _run_scripted_nodes(env, _repair_on_block, s)),
        ("if-else velocity", lambda s: _run_scripted_nodes(env, _ifelse_velocity, s)),
        ("PPO (trained)",    lambda s: _run_model_nodes(model, env, s)),
    ]

    print("=" * 74)
    print(f"{label}")
    print("=" * 74)
    print(f"{'policy':<18}{'success':>9}{'avg_nodes':>12}{'avg_repairs':>13}"
          f"{'avg_reward':>12}")
    print("-" * 74)

    rows = {}
    for name, runner in policies:
        res = [runner(s) for s in seeds]
        succ = sum(1 for r, _, _, _ in res if r == "success")
        avg_reward = np.mean([t for _, t, _, _ in res])
        avg_rep = np.mean([rp for _, _, rp, _ in res])
        avg_nodes = np.mean([nd for _, _, _, nd in res])
        rows[name] = (succ, avg_nodes, avg_rep, avg_reward)
        print(f"{name:<18}{succ:>6}/{len(seeds)}{avg_nodes:>12.1f}"
              f"{avg_rep:>13.2f}{avg_reward:>12.2f}")

    print("-" * 74)
    # honest verdict on the HEADLINE metric (node expansions) at comparable success
    base = rows["repair-on-block"]
    ifel = rows["if-else velocity"]
    ppo = rows["PPO (trained)"]

    def pct_fewer(a, b):
        return 100.0 * (b - a) / b if b > 0 else 0.0

    print("NODE-EXPANSION verdict (lower is better; headline metric):")
    print(f"  PPO vs repair-on-block : {ppo[1]:.0f} vs {base[1]:.0f} nodes "
          f"-> PPO uses {pct_fewer(ppo[1], base[1]):+.1f}% "
          f"({'FEWER, good' if ppo[1] < base[1] else 'MORE, worse'})")
    print(f"  PPO vs if-else velocity: {ppo[1]:.0f} vs {ifel[1]:.0f} nodes "
          f"-> PPO uses {pct_fewer(ppo[1], ifel[1]):+.1f}% "
          f"({'FEWER, good' if ppo[1] < ifel[1] else 'MORE, worse'})")
    # success guard: a node win only counts if success is comparable
    if ppo[0] < base[0] - 5:
        print(f"  CAUTION: PPO success ({ppo[0]}) is notably below "
              f"repair-on-block ({base[0]}); a node saving from failing "
              f"early does not count as a real win.")
    print("=" * 74)
    print()
    return rows


# ----------------------------------------------------------------------
# full comparison: mixed (headline) + per-category diagnostics
# ----------------------------------------------------------------------
def compare_all(model, provider=None, n=100):
    seeds = list(range(9000, 9000 + n))

    print("\n" + "#" * 74)
    print("# NODE-EXPANSION COMPARISON  (base-paper metric: lower nodes = better)")
    print("# same Adaptive A* planner everywhere; only the WAIT/REPAIR decider differs")
    print("#" * 74 + "\n")

    # MIXED (the real scenario -- temporaries + permanents together) -> HEADLINE
    cm = DynamicConfigV9()
    mixed = _compare_on(cm, "MIXED  (temporaries + permanents -- the real test)",
                        model, provider, seeds)

    # TEMPORARY-ONLY (where waiting CAN save replans -> where PPO can win on nodes)
    ct = DynamicConfigV9(); ct.n_perm_min = 0; ct.n_perm_max = 0
    ct.n_temp_min = 2; ct.n_temp_max = 3
    temp = _compare_on(ct, "TEMPORARY-ONLY  (waiting avoids replans -> PPO's best case)",
                       model, provider, seeds)

    # PERMANENT-ONLY (everyone must repair -> little room to differ on nodes)
    cp = DynamicConfigV9(); cp.n_temp_min = 0; cp.n_temp_max = 0
    cp.n_perm_min = 1; cp.n_perm_max = 2
    perm = _compare_on(cp, "PERMANENT-ONLY  (all must repair -> little node difference)",
                       model, provider, seeds)

    # ---- honest overall summary ----
    print("#" * 74)
    print("# HONEST SUMMARY")
    print("#" * 74)
    pm = mixed["PPO (trained)"]; bm = mixed["repair-on-block"]; im = mixed["if-else velocity"]
    print(f"On MIXED, PPO expands {pm[1]:.0f} nodes vs repair-on-block {bm[1]:.0f} "
          f"and if-else {im[1]:.0f}.")
    if pm[1] < bm[1] and pm[1] < im[1] and pm[0] >= bm[0] - 5:
        print("=> PPO uses the FEWEST node expansions at comparable success. This is a")
        print("   genuine efficiency contribution over BOTH the naive baseline and the")
        print("   hand-crafted heuristic -- the result worth reporting.")
    elif pm[1] < bm[1] and pm[0] >= bm[0] - 5:
        print("=> PPO beats the NAIVE baseline (repair-on-block) on node expansions, but")
        print("   NOT the if-else heuristic. Honest framing: 'RL reduces replanning cost")
        print("   versus naive replan-on-block; matches a hand-crafted heuristic.'")
    else:
        print("=> PPO does NOT win on node expansions on mixed. Honest framing: 'the")
        print("   learned policy matches heuristic baselines; the wait/repair decision is")
        print("   simple enough that hand-crafted rules are competitive.' Consider the")
        print("   temporary-only column -- that is where any RL node-saving shows up.")
    print("#" * 74)
