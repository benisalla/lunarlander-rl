"""04: the continuous LunarLander (2 throttles in [-1, 1]) with DDPG, TD3, SAC or PPO.

    python examples/04_train_continuous.py                  # SAC, 20k steps, about 3-4 minutes
    python examples/04_train_continuous.py --method td3
    python examples/04_train_continuous.py --method ddpg
    python examples/04_train_continuous.py --method ppo-continuous --steps 100000

The first ``learning_starts`` steps are random actions that fill the replay buffer; after that
the agent updates every ``train_freq`` steps. For SAC it prints how the entropy temperature
alpha moved: it shrinks as the critic becomes trustworthy and exploration matters less.
"""

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import make_config, train  # noqa: E402
from lunarlander_rl.viz import plot_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--method", default="sac", choices=["ddpg", "td3", "sac", "ppo-continuous"])
p.add_argument("--steps", type=int)
a = p.parse_args()

steps = a.steps or (100_000 if a.method == "ppo-continuous" else 20_000)
over = dict(total_steps=steps, eval_every=max(5_000, steps // 8), eval_episodes=5)
if a.method == "ppo-continuous":
    over.update(n_envs=8, n_steps=512)
cfg = make_config(a.method, **over)
agent, summary = train(cfg, f"examples/output/{a.method}")

m = json.load(open(f"examples/output/{a.method}/metrics.json"))
if "alpha" in m["losses"]:
    al = m["losses"]["alpha"]
    print(f"SAC temperature alpha: {al[0][1]:.3f} at step {al[0][0]:,} -> {al[-1][1]:.3f} at step {al[-1][0]:,}")
print("curve:", plot_training(m, f"examples/output/{a.method}_curve.png"))
