"""Build everything in assets/ from runs/: results table, curves, per-run training plots, GIFs.

    python scripts/make_assets.py                    # GIFs for every run
    python scripts/make_assets.py --gifs ppo sac     # GIFs only for these
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import load_agent  # noqa: E402
from lunarlander_rl.compare import load_runs, main as compare  # noqa: E402
from lunarlander_rl.play import record_episode, save_gif  # noqa: E402
from lunarlander_rl.viz import plot_training  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--out", default="assets")
p.add_argument("--gifs", nargs="*")
p.add_argument("--gif-seed", type=int, default=3)
a = p.parse_args()

compare([a.runs, "--out", a.out])
runs = load_runs(a.runs)
for name, m in runs.items():
    plot_training(m, f"{a.out}/train_{name}.png", f"{name}: training episodes and greedy evaluations")
for name in (a.gifs if a.gifs is not None else runs):
    agent, cfg = load_agent(f"{a.runs}/{name}")
    frames, ret = record_episode(agent, cfg.continuous, seed=a.gif_seed)
    print(name, "GIF return", round(ret, 1), save_gif(frames, f"{a.out}/{name}.gif", every=3, scale=0.4))
