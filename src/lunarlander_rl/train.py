"""Train any method on LunarLander and save the run.

    lunarlander-train --method ppo                       # preset budget
    lunarlander-train --method dqn --steps 50000         # shorter
    lunarlander-train --method dueling-dqn --set n_step=1 --set per=false
    lunarlander-train --list                             # every method and its one-line idea

A run directory holds ``config.json``, ``model.pt`` (network weights) and ``metrics.json``
(training episodes, periodic evaluations, losses and the summary used by lunarlander-compare).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from .algorithms import build_agent
from .config import METHODS, Config, make_config, parse_override
from .envs import make_env, make_vec_env, set_seed
from .evaluate import evaluate
from .metrics import Tracker


def train(cfg: Config, out_dir: str | Path | None = None, verbose: bool = True):
    """Train one agent with ``cfg``; returns (agent, summary dict). Saves to ``out_dir`` if given."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    agent = build_agent(cfg)
    tracker = Tracker(cfg, agent, verbose=verbose)
    if agent.vectorized:
        env = make_vec_env(cfg.n_envs, cfg.continuous, cfg.seed)
    else:
        env = make_env(cfg.continuous, cfg.seed)
    t0 = time.time()
    agent.learn(env, tracker)
    env.close()
    train_time = time.time() - t0
    final = evaluate(agent, cfg.continuous, cfg.final_eval_episodes, seed=20_000 + cfg.seed)
    summary = tracker.summary(final)
    summary["wall_time"] = train_time
    if verbose:
        solved = f"solved at {summary['solved_at']:,} steps" if summary["solved_at"] else "not solved"
        print(f"[{cfg.method}] done in {train_time:.0f}s: final {summary['final_mean']:.1f} +- "
              f"{summary['final_std']:.1f} over {len(final)} episodes, {solved}", flush=True)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg.save(out / "config.json")
        torch.save({"method": cfg.method, "state": agent.state_dict()}, out / "model.pt")
        tracker.save(out / "metrics.json", final, wall_time=train_time)
    return agent, summary


def load_agent(run_dir: str | Path):
    """Rebuild the agent of a saved run (weights included); returns (agent, cfg)."""
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.json")
    agent = build_agent(cfg)
    ckpt = torch.load(run_dir / "model.pt", map_location=cfg.device, weights_only=False)
    agent.load_state_dict(ckpt["state"])
    return agent, cfg


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train an RL agent on LunarLander.")
    p.add_argument("--method", default="dqn", choices=sorted(METHODS))
    p.add_argument("--steps", type=int, help="environment steps (default: the preset's budget)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", help="run directory (default runs/<method>)")
    p.add_argument("--stop-when-solved", action="store_true", help="stop at the first eval with mean >= 200")
    p.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock cap for training")
    p.add_argument("--eval-every", type=int, help="env steps between greedy evaluations")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any Config field")
    p.add_argument("--list", action="store_true", help="list the methods and exit")
    a = p.parse_args(argv)

    if a.list:
        for name, preset in METHODS.items():
            env = "continuous" if preset.continuous else "discrete"
            steps = preset.overrides.get("total_steps", Config().total_steps)
            print(f"{name:15s} {env:10s} {steps:>8,d} steps  {preset.summary}")
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    overrides = dict(parse_override(s) for s in a.set)
    overrides.update(seed=a.seed, threads=a.threads, stop_when_solved=a.stop_when_solved, max_minutes=a.max_minutes)
    if a.steps:
        overrides["total_steps"] = a.steps
    if a.eval_every:
        overrides["eval_every"] = a.eval_every
    cfg = make_config(a.method, **overrides)
    train(cfg, a.out or f"runs/{a.method}")


if __name__ == "__main__":
    main()
