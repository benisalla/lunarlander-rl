"""01: what the agent sees and how it is rewarded, before any learning.

Prints the observation / action spaces, plays one episode with random actions and one with
gymnasium's hand-written heuristic controller, checks step by step that the reward equals the
change of the shaping potential minus fuel (``envs.shaping_potential``), and saves a GIF of the
heuristic landing to examples/output/heuristic.gif. Runs in a few seconds:

    python examples/01_explore_the_env.py
"""

from pathlib import Path

import numpy as np

from lunarlander_rl.envs import heuristic_action, make_env, shaping_potential
from lunarlander_rl.evaluate import BASELINES, evaluate
from lunarlander_rl.play import record_episode, save_gif

OUT = Path("examples/output")

env = make_env(continuous=False)
print("observation space:", env.observation_space)
print("  [x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact]")
print("discrete actions  :", env.action_space, "-> 0 noop, 1 left engine, 2 main engine, 3 right engine")
print("continuous actions:", make_env(continuous=True).action_space, "-> [main throttle, lateral throttle]")

# one heuristic episode, decomposing every reward into shaping + fuel + terminal bonus
obs, _ = env.reset(seed=0)
total, shaping_sum, fuel_sum, t, done = 0.0, 0.0, 0.0, 0, False
while not done:
    a = heuristic_action(env, obs)
    nxt, r, term, trunc, _ = env.step(a)
    dphi = shaping_potential(nxt) - shaping_potential(obs)
    fuel = 0.30 * (a == 2) + 0.03 * (a in (1, 3))
    if not term and t > 0:  # first step has no previous potential; terminal step is +-100 instead
        assert abs(r - (dphi - fuel)) < 1e-3, (r, dphi, fuel)
        shaping_sum += dphi
        fuel_sum += fuel
    total, obs, t, done = total + r, nxt, t + 1, term or trunc
print(f"\nheuristic episode: return {total:.1f} in {t} steps "
      f"(shaping {shaping_sum:+.1f}, fuel {-fuel_sum:.1f}, last reward {r:+.0f} = landing bonus)")
print("reward = Phi(s') - Phi(s) - fuel checked on every step: OK")

for name in ("random", "heuristic"):
    rets = evaluate(None, False, 10, policy=BASELINES[name])
    print(f"{name:9s} policy over 10 episodes: {rets.mean():7.1f} +- {rets.std():.1f}")

frames, ret = record_episode(None, False, seed=1, policy=BASELINES["heuristic"])
print("GIF:", save_gif(frames, OUT / "heuristic.gif"), f"(return {ret:.1f})")
