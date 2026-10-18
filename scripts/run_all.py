"""Train several methods one after another (one process, one thread) into runs/<method>.

    python scripts/run_all.py dqn double-dqn dueling-dqn --stop-when-solved
    python scripts/run_all.py --all --steps 20000            # quick pass over every method

Then ``lunarlander-compare runs --out assets`` builds the table and the figures.
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lunarlander_rl import METHODS, make_config, train  # noqa: E402
from lunarlander_rl.config import parse_override  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("methods", nargs="*")
p.add_argument("--all", action="store_true")
p.add_argument("--runs", default="runs")
p.add_argument("--steps", type=int)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--stop-when-solved", action="store_true")
p.add_argument("--set", action="append", default=[])
a = p.parse_args()
methods = list(METHODS) if a.all else a.methods
for m in methods:
    over = dict(parse_override(s) for s in a.set)
    over.update(seed=a.seed, stop_when_solved=a.stop_when_solved)
    if a.steps:
        over["total_steps"] = a.steps
    train(make_config(m, **over), f"{a.runs}/{m}")
