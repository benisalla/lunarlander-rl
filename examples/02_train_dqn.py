"""02: train a DQN-family agent on the discrete LunarLander, then evaluate it.

One command, a few CPU minutes. The run is saved to examples/output/<method>/ (config.json,
model.pt, metrics.json) and its learning curve to examples/output/<method>_curve.png.

    python examples/02_train_dqn.py                          # DQN, 30k steps, about 2 minutes
    python examples/02_train_dqn.py --method double-dqn
    python examples/02_train_dqn.py --method dueling-dqn --steps 80000

30k steps is enough to see the return climb from about -200 (crashing) to positive values
(hovering, then landing); the README's numbers use the 150k-step preset.
"""

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import evaluate, make_config, train  # noqa: E402
from lunarlander_rl.viz import plot_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--method", default="dqn", choices=["dqn", "double-dqn", "dueling-dqn"])
p.add_argument("--steps", type=int, default=30_000)
a = p.parse_args()

cfg = make_config(a.method, total_steps=a.steps, eval_every=5_000, eval_episodes=5)
print(f"{a.method}: double={cfg.double} dueling={cfg.dueling} per={cfg.per} n_step={cfg.n_step}, "
      f"{cfg.total_steps:,} steps, epsilon {cfg.eps_start} -> {cfg.eps_end}")
agent, summary = train(cfg, f"examples/output/{a.method}")

rets = evaluate(agent, continuous=False, n_episodes=10, seed=123)
print(f"greedy policy on 10 new episodes: {rets.mean():.1f} +- {rets.std():.1f}")
m = json.load(open(f"examples/output/{a.method}/metrics.json"))
print("curve:", plot_training(m, f"examples/output/{a.method}_curve.png"))
