"""lunarlander_rl: every classic deep-RL method on LunarLander, one small readable code base.

    from lunarlander_rl import make_config, train
    agent, summary = train(make_config("ppo", total_steps=200_000), "runs/ppo")
"""

from .algorithms import ALGORITHMS, build_agent
from .config import METHODS, Config, make_config
from .envs import SOLVED_SCORE, make_env, make_vec_env
from .evaluate import evaluate
from .train import load_agent, train

__all__ = ["ALGORITHMS", "METHODS", "SOLVED_SCORE", "Config", "build_agent", "evaluate", "load_agent",
           "make_config", "make_env", "make_vec_env", "train"]
__version__ = "0.1.0"
