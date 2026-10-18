"""05: run several methods with the same budget, then build the comparison table and curves.

    python examples/05_compare_methods.py                                 # 4 methods x 10k steps, about 5 minutes
    python examples/05_compare_methods.py --methods dqn ppo td3 sac --steps 50000
    python examples/05_compare_methods.py --all --steps 5000              # every method runs (smoke)

This is exactly what produced the README tables, with smaller budgets: every run goes to
examples/output/compare/<method>/ and lunarlander_rl.compare turns them into a markdown table
(examples/output/compare/results.md) and figures (curves_*.png).
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import METHODS, make_config, train  # noqa: E402
from lunarlander_rl.compare import main as compare  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--methods", nargs="+", default=["dqn", "ppo", "td3", "sac"])
p.add_argument("--all", action="store_true")
p.add_argument("--steps", type=int, default=10_000)
a = p.parse_args()

root = "examples/output/compare"
for m in (list(METHODS) if a.all else a.methods):
    over = dict(total_steps=a.steps, eval_every=max(2_500, a.steps // 4), eval_episodes=3, final_eval_episodes=5,
                learning_starts=min(5_000, a.steps // 5))
    if METHODS[m].algo == "ppo":
        over.update(n_envs=8, n_steps=min(512, max(64, a.steps // 32)))
    train(make_config(m, **over), f"{root}/{m}")
compare([root, "--out", root])
