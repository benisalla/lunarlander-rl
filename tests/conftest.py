"""Shared fixtures: a contextual bandit with LunarLander's shapes, for fast "does it learn" checks.

Observation: 8 random floats. Discrete: reward 1 if the action is argmax(obs[:4]), else 0.
Continuous: reward = -||a - clip(obs[:2] / 2, -1, 1)||^2. Every episode lasts one step, so the
optimal policy is a pure function of the observation and is learnt in a few thousand steps.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)


class ContextBandit(gym.Env):
    def __init__(self, continuous: bool = False):
        self.continuous = continuous
        self.observation_space = gym.spaces.Box(-5, 5, (8,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32) if continuous else gym.spaces.Discrete(4)
        self._obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self.np_random.normal(size=8).astype(np.float32)
        return self._obs, {}

    def reward(self, obs, action) -> float:
        if self.continuous:
            target = np.clip(obs[:2] / 2, -1, 1)
            return -float(np.sum((np.asarray(action) - target) ** 2))
        return float(int(action) == int(np.argmax(obs[:4])))

    def step(self, action):
        r = self.reward(self._obs, action)
        return self._obs, r, True, False, {}


def bandit_score(agent, continuous: bool, n: int = 300) -> float:
    env, rng = ContextBandit(continuous), np.random.default_rng(123)
    total = 0.0
    for _ in range(n):
        obs = rng.normal(size=8).astype(np.float32)
        total += env.reward(obs, agent.act(obs))
    return total / n


@pytest.fixture
def bandit():
    return ContextBandit


@pytest.fixture
def score():
    return bandit_score
