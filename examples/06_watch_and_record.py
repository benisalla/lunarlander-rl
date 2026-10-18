"""06: load a saved run, watch it land, compare it with the baselines, save a GIF.

    python examples/06_watch_and_record.py                         # examples/output/dqn (from 02)
    python examples/06_watch_and_record.py runs/ppo                # any run directory

Writes examples/output/<method>.gif; the README GIFs come from the full runs in runs/.
"""

import sys
from pathlib import Path

from lunarlander_rl import evaluate, load_agent
from lunarlander_rl.evaluate import BASELINES
from lunarlander_rl.play import record_episode, save_gif

run = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/output/dqn")
if not (run / "model.pt").exists():
    sys.exit(f"{run} has no model.pt: train first (python examples/02_train_dqn.py)")
agent, cfg = load_agent(run)
print(f"loaded {cfg.method} ({'continuous' if cfg.continuous else 'discrete'} env) from {run}")

rows = [(cfg.method, evaluate(agent, cfg.continuous, 10, seed=7))]
rows += [(name, evaluate(None, cfg.continuous, 10, seed=7, policy=pol)) for name, pol in BASELINES.items()]
for name, rets in rows:
    print(f"  {name:15s} {rets.mean():8.1f} +- {rets.std():6.1f}   landed (>= 200): {(rets >= 200).mean():.0%}")

frames, ret = record_episode(agent, cfg.continuous, seed=7)
print("GIF:", save_gif(frames, Path("examples/output") / f"{cfg.method}.gif"), f"(return {ret:.1f}, {len(frames)} frames)")
