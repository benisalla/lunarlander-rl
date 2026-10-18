"""One file per algorithm. ``build_agent(cfg)`` picks the class named by ``cfg.algo``."""

from __future__ import annotations

from ..config import Config
from ..envs import obs_dim_and_actions
from .a2c import A2CAgent
from .base import Agent
from .ddpg import DDPGAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .reinforce import ReinforceAgent
from .sac import SACAgent
from .sac_discrete import SACDiscreteAgent
from .td3 import TD3Agent

ALGORITHMS: dict[str, type[Agent]] = {
    "dqn": DQNAgent,
    "reinforce": ReinforceAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
    "sac_discrete": SACDiscreteAgent,
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
}


def build_agent(cfg: Config) -> Agent:
    obs_dim, n_actions = obs_dim_and_actions(cfg.continuous)
    return ALGORITHMS[cfg.algo](cfg, obs_dim, n_actions)


__all__ = ["ALGORITHMS", "Agent", "build_agent", "A2CAgent", "DDPGAgent", "DQNAgent", "PPOAgent",
           "ReinforceAgent", "SACAgent", "SACDiscreteAgent", "TD3Agent"]
