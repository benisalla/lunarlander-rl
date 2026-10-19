"""Regenerate the README's before/after and "solved" GIFs from the saved checkpoints in runs/.

    python scripts/make_gifs.py                     # assets/before_after_ppo.gif, ..._ppo-continuous.gif, solved_grid.gif
    python scripts/make_gifs.py --seed 5 --scale 0.4

Before/after: the same episode seed played by an untrained network of the same method (left)
and by the trained checkpoint (right). Grid: one panel per method whose run reached a greedy
evaluation >= 200 and kept a final 20-episode score >= 150. A missing checkpoint is trained
first with the method's preset (hold the shared lock for that: it takes minutes).
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from lunarlander_rl import load_agent, make_config, train  # noqa: E402
from lunarlander_rl.play import before_after, record_episode, save_panels_gif  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--out", default="assets")
p.add_argument("--seed", type=int, default=3)
p.add_argument("--scale", type=float, default=0.5)
p.add_argument("--every", type=int, default=3)
a = p.parse_args()
runs, out = Path(a.runs), Path(a.out)


def ensure(method: str) -> Path:
    d = runs / method
    if not (d / "model.pt").exists():
        print(f"{d} missing: training {method} with its preset (--stop-when-solved)")
        train(make_config(method, stop_when_solved=True), d)
    return d


for method in ("ppo", "ppo-continuous"):
    path, r0, r1 = before_after(ensure(method), out / f"before_after_{method}.gif", seed=a.seed,
                                every=a.every, scale=a.scale)
    print(f"{path}: before {r0:.1f}, after {r1:.1f}, {path.stat().st_size / 1e6:.2f} MB")

solved = []
for f in sorted(runs.glob("*/metrics.json")):
    s = json.loads(f.read_text())["summary"]
    if s.get("solved_at") and s["final_mean"] >= 150:
        solved.append(f.parent.name)
order = ["dqn", "double-dqn", "dueling-dqn", "reinforce", "a2c", "ppo", "sac-discrete", "ddpg", "td3", "sac", "ppo-continuous"]
solved.sort(key=lambda m: order.index(m) if m in order else 99)
episodes = []
for name in solved:
    agent, cfg = load_agent(runs / name)
    frames, ret = record_episode(agent, cfg.continuous, seed=a.seed)
    episodes.append((frames, f"{name}: return {ret:.0f}"))
path = save_panels_gif(episodes, out / "solved_grid.gif", ncols=min(3, len(episodes)), every=a.every, scale=a.scale)
print(f"{path}: {', '.join(solved)}, {path.stat().st_size / 1e6:.2f} MB")
