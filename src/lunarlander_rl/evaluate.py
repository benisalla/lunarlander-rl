"""Greedy evaluation: the same protocol for every method.

``evaluate`` plays N full episodes with the deterministic policy (argmax Q, argmax pi, the
Gaussian mean, or tanh(mean) for SAC), from fixed reset seeds, and returns the N returns.
LunarLander counts as solved at a mean of 200.

CLI::

    lunarlander-eval runs/ppo --episodes 100
    lunarlander-eval --baseline heuristic        # gymnasium's hand-written controller
    lunarlander-eval --baseline random --continuous
"""

from __future__ import annotations

import argparse

import numpy as np

from .envs import SOLVED_SCORE, heuristic_action, make_env


def evaluate(agent, continuous: bool, n_episodes: int = 10, seed: int = 10_000, policy=None) -> np.ndarray:
    """Return an array of ``n_episodes`` undiscounted returns.

    ``policy(env, obs) -> action`` replaces the agent (used for the random / heuristic baselines).
    """
    env = make_env(continuous)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            action = policy(env, obs) if policy is not None else agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        returns.append(ret)
    env.close()
    return np.array(returns)


BASELINES = {
    "random": lambda env, obs: env.action_space.sample(),
    "heuristic": heuristic_action,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained run (or a baseline) on LunarLander.")
    p.add_argument("run_dir", nargs="?", help="directory written by lunarlander-train")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--baseline", choices=sorted(BASELINES), help="evaluate a reference policy instead")
    p.add_argument("--continuous", action="store_true", help="(baselines) use the continuous env")
    a = p.parse_args(argv)

    if a.baseline:
        rets = evaluate(None, a.continuous, a.episodes, a.seed, policy=BASELINES[a.baseline])
        name = f"{a.baseline} ({'continuous' if a.continuous else 'discrete'})"
    elif a.run_dir:
        from .train import load_agent

        agent, cfg = load_agent(a.run_dir)
        rets = evaluate(agent, cfg.continuous, a.episodes, a.seed)
        name = cfg.method
    else:
        p.error("give a run directory or --baseline")
    solved = "solved" if rets.mean() >= SOLVED_SCORE else "not solved"
    print(f"{name}: mean return {rets.mean():.1f} +- {rets.std():.1f} over {len(rets)} episodes "
          f"(min {rets.min():.1f}, max {rets.max():.1f}, {np.mean(rets >= 200):.0%} episodes >= 200) -> {solved}")
