"""What every agent shares: tensor conversion, greedy action selection, save / load."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import Config


class Agent:
    """Subclasses build their networks in ``__init__``, list the ones to save in ``self.nets``,
    implement ``greedy(obs_tensor)`` (deterministic policy) and ``learn(env, tracker)``.

    ``vectorized = True`` means ``learn`` gets a ``gymnasium.vector`` env with ``cfg.n_envs``
    copies (A2C, PPO); otherwise it gets a single env.
    """

    vectorized = False

    def __init__(self, cfg: Config, obs_dim: int, n_actions: int):
        self.cfg, self.obs_dim, self.n_actions = cfg, obs_dim, n_actions
        self.device = torch.device(cfg.device)
        self.nets: dict[str, nn.Module] = {}

    def t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True):
        """Action for one observation, in the format the env expects (int or float array)."""
        a = self.greedy(self.t(obs).unsqueeze(0))[0]
        return a.cpu().numpy().astype(np.float32) if self.cfg.continuous else int(a)

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def learn(self, env, tracker) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.nets.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            self.nets[k].load_state_dict(v)


def clip_grads(params, max_norm: float) -> None:
    if max_norm and max_norm > 0:
        nn.utils.clip_grad_norm_(params, max_norm)
