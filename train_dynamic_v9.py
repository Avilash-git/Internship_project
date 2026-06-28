"""
train_dynamic_V9.py
===================
Train + evaluate the PPO WAIT-vs-REPAIR agent on the V9 dynamic-obstacle env.

Drop this next to ppo_dynamic_v9.py, ppo_repair.py, adaptive_astar.py.

Quick start (Colab):
    from train_dynamic_V9 import train, verify, evaluate
    verify(provider=None)                 # sanity-check the env first (minutes)
    model = train(total_timesteps=1_000_000, provider=None)   # SyntheticProbMap
    evaluate(model, provider=None)        # reward + repair-count metrics

With the real frozen U-Net:
    from ppo_repair import UNetProbMap
    from ppo_dynamic_v9 import DynamicConfigV9
    cfg = DynamicConfigV9()
    prov = UNetProbMap(cfg, "best_attention_astar_unet.pth")
    verify(provider=prov)
    model = train(total_timesteps=1_000_000, provider=prov)
    evaluate(model, provider=prov)
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ppo_dynamic_v9 import DynamicConfigV9, DynamicObstacleEnvV9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
# ======================================================================
# feature extractor: CNN over the 6x40x40 grid + MLP merge with the vec
# ======================================================================
class GridVecExtractorV9(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space["grid"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        vec_dim = observation_space["vec"].shape[0]
        self.merge = nn.Sequential(nn.Linear(n_flat + vec_dim, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.merge(torch.cat([self.cnn(obs["grid"]), obs["vec"]], dim=1))


def _make_env(cfg, provider, seed=0):
    def _init():
        env = Monitor(DynamicObstacleEnvV9(cfg, prob_provider=provider))
        env.reset(seed=seed)
        return env
    return _init


# ======================================================================
# TRAIN
# ======================================================================
def train(cfg=None, total_timesteps=1_000_000, save_path="ppo_dynamic_v9",
          provider=None, n_envs=4, verbose=1 , device=device):
    cfg = cfg or DynamicConfigV9()
    venv = DummyVecEnv([_make_env(cfg, provider, seed=i) for i in range(n_envs)])
    policy_kwargs = dict(
        features_extractor_class=GridVecExtractorV9,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[64, 64],
    )
    model = PPO("MultiInputPolicy", venv, policy_kwargs=policy_kwargs,
                n_steps=512, batch_size=256, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.01, learning_rate=3e-4, verbose=verbose,device=device)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"saved -> {save_path}.zip")
    return model


# ======================================================================
# scripted baselines (for verify + evaluate comparison)
# ======================================================================
def _always_wait(env):
    return 0

def _repair_on_block(env):
    return 1 if env._is_blocked() else 0

def _ifelse_velocity(env):
    if not env._is_blocked():
        return 0
    nb, _ = env._nearest_blocking_ahead()
    if nb is None:
        return 0
    vx, vy = nb.velocity()
    return 0 if (abs(vx) + abs(vy)) > 0 else 1


def _run_scripted(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(env.cfg.max_steps + 5):
        obs, r, term, trunc, info = env.step(policy(env))
        total += r
        if term or trunc:
            return info.get("result", "?"), total, info.get("n_repairs", 0)
    return "timeout", total, 0


def _run_model(model, env, seed):
    obs, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(env.cfg.max_steps + 5):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        total += r
        if term or trunc:
            return info.get("result", "?"), total, info.get("n_repairs", 0)
    return "timeout", total, 0


# ======================================================================
# VERIFY (run BEFORE training)
# ======================================================================
def verify(provider=None, n=60):
    print("=" * 60)
    print("V9 ENVIRONMENT VERIFICATION")
    print("=" * 60)
    seeds = list(range(2000, 2000 + n))

    # temporary-only: WAIT should win
    ct = DynamicConfigV9(); ct.n_perm_min = 0; ct.n_perm_max = 0
    ct.n_temp_min = 2; ct.n_temp_max = 3
    et = DynamicObstacleEnvV9(ct, prob_provider=provider)
    w_t = sum(_run_scripted(et, _always_wait, s)[0] == "success" for s in seeds)

    # permanent-only: REPAIR should win, WAIT should fail
    cp = DynamicConfigV9(); cp.n_temp_min = 0; cp.n_temp_max = 0
    cp.n_perm_min = 1; cp.n_perm_max = 2
    ep = DynamicObstacleEnvV9(cp, prob_provider=provider)
    w_p = sum(_run_scripted(ep, _always_wait, s)[0] == "success" for s in seeds)
    r_p = sum(_run_scripted(ep, _repair_on_block, s)[0] == "success" for s in seeds)

    print(f"temporary-only + always-WAIT  : {w_t}/{n}  (want HIGH)")
    print(f"permanent-only + always-WAIT  : {w_p}/{n}  (want LOW)")
    print(f"permanent-only + always-REPAIR: {r_p}/{n}  (want HIGH)")
    ok = (w_t > 0.6 * n and w_p < 0.4 * n and r_p > 0.6 * n)
    print("VERDICT:", "PASS - real decision exists" if ok else "CHECK - env may be off")

    # if-else baseline fooled by lingers? (reward gap on temporary-only)
    wait_rewards = [_run_scripted(et, _always_wait, s)[1] for s in seeds]
    ifel = [_run_scripted(et, _ifelse_velocity, s) for s in seeds]
    ifel_rewards = [t for _, t, _ in ifel]
    ifel_repairs = [n_ for _, _, n_ in ifel]
    print(f"\ntemporary-only reward:  always-WAIT {np.mean(wait_rewards):.2f} | "
          f"if-else {np.mean(ifel_rewards):.2f} | "
          f"if-else wastes {np.mean(ifel_repairs):.2f} repairs/ep")
    print("  (positive gap = room for PPO to beat if-else by waiting through lingers)")
    print("=" * 60)


# ======================================================================
# EVALUATE (run AFTER training) -- reward & repair-count are the headline
# ======================================================================
def evaluate(model, provider=None, n=100):
    print("=" * 60)
    print("V9 EVALUATION  (reward & repairs are the real metrics)")
    print("=" * 60)
    seeds = list(range(9000, 9000 + n))
    cfg = DynamicConfigV9()
    env = DynamicObstacleEnvV9(cfg, prob_provider=provider)

    rows = []
    for name, runner in [("always-WAIT", lambda s: _run_scripted(env, _always_wait, s)),
                         ("repair-on-block", lambda s: _run_scripted(env, _repair_on_block, s)),
                         ("if-else velocity", lambda s: _run_scripted(env, _ifelse_velocity, s)),
                         ("PPO (trained)", lambda s: _run_model(model, env, s))]:
        res = [runner(s) for s in seeds]
        succ = sum(1 for r, _, _ in res if r == "success")
        avg_r = np.mean([t for _, t, _ in res])
        avg_rep = np.mean([rp for _, _, rp in res])
        rows.append((name, succ, avg_r, avg_rep))

    print(f"{'policy':<20}{'success':>10}{'avg_reward':>14}{'avg_repairs':>14}")
    for name, succ, avg_r, avg_rep in rows:
        print(f"{name:<20}{succ:>7}/{n}{avg_r:>14.2f}{avg_rep:>14.2f}")
    print("\nPPO should match/beat success AND get higher reward / fewer wasted repairs")
    print("than repair-on-block and if-else (it waits through lingering temporaries).")
    print("=" * 60)
    return rows
