"""03: the policy-gradient family on the discrete LunarLander: REINFORCE, A2C, PPO.

Same command for the three; they differ only in how the advantage is estimated and how many
gradient steps each batch of experience gets (see docs/architecture.md).

    python examples/03_train_policy_gradient.py                     # PPO, 100k steps, about 2-3 minutes
    python examples/03_train_policy_gradient.py --method a2c
    python examples/03_train_policy_gradient.py --method reinforce --steps 100000

Prints the PPO diagnostics logged during training (clip fraction, approximate KL, entropy).
"""

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import make_config, train  # noqa: E402
from lunarlander_rl.viz import plot_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--method", default="ppo", choices=["reinforce", "a2c", "ppo"])
p.add_argument("--steps", type=int, default=100_000)
a = p.parse_args()

over = dict(total_steps=a.steps, eval_every=10_000, eval_episodes=5)
if a.method == "ppo":
    over.update(n_envs=8, n_steps=512)  # 4096 transitions per update: more updates in a short run
cfg = make_config(a.method, **over)
agent, summary = train(cfg, f"examples/output/{a.method}")

m = json.load(open(f"examples/output/{a.method}/metrics.json"))
for name, series in m["losses"].items():
    print(f"  {name:10s} first {series[0][1]:9.4f}   last {series[-1][1]:9.4f}")
print("curve:", plot_training(m, f"examples/output/{a.method}_curve.png"))
