"""The LunarLander environments (gymnasium + Box2D) and a few helpers around them.

* ``LunarLander-v3``: observation = 8 floats (x, y, vx, vy, angle, angular velocity, left leg
  contact, right leg contact); 4 discrete actions (noop, left engine, main engine, right engine).
* ``LunarLander-v3`` with ``continuous=True``: the same observation, action = 2 floats in [-1, 1]
  (main engine throttle, lateral engine throttle).

Episodes end on a crash/landing (``terminated``) or after 1000 steps (``truncated``). The
problem counts as solved when the mean return is 200 or more.
"""

from __future__ import annotations

import os
import random

import gymnasium as gym
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering for GIFs

ENV_ID = "LunarLander-v3"
SOLVED_SCORE = 200.0
MAX_EPISODE_STEPS = 1000


def make_env(continuous: bool = False, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, continuous=continuous, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_vec_env(n_envs: int, continuous: bool = False, seed: int = 0) -> gym.vector.SyncVectorEnv:
    """``n_envs`` copies stepped in lock-step (one process, no IPC; LunarLander is cheap).

    ``SAME_STEP`` autoreset: when an episode ends, ``step`` already returns the first observation
    of the next episode, and the real last observation is in ``info["final_obs"]``.
    """
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(ENV_ID, continuous=continuous) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    return envs


def obs_dim_and_actions(continuous: bool) -> tuple[int, int]:
    """(8, 4) for the discrete env, (8, 2) for the continuous one: the sizes the networks need."""
    return 8, (2 if continuous else 4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def heuristic_action(env: gym.Env, obs: np.ndarray):
    """The hand-written PID controller that ships with gymnasium (a useful reference policy)."""
    from gymnasium.envs.box2d.lunar_lander import heuristic

    return heuristic(env.unwrapped, obs)


def shaping_potential(obs: np.ndarray) -> float:
    """The potential Phi(s) that LunarLander's per-step reward is built from.

    reward_t = Phi(s_{t+1}) - Phi(s_t) - 0.30 * main_power - 0.03 * side_power,
    plus -100 on a crash / flying off screen or +100 when the lander comes to rest. Phi is high
    near the pad (0, 0), at low speed, upright, with legs touching the ground:

        Phi(s) = -100 |(x, y)| - 100 |(vx, vy)| - 100 |angle| + 10 leg_left + 10 leg_right
    """
    x, y, vx, vy, angle, _, leg_l, leg_r = obs
    return float(-100 * np.hypot(x, y) - 100 * np.hypot(vx, vy) - 100 * abs(angle) + 10 * leg_l + 10 * leg_r)
