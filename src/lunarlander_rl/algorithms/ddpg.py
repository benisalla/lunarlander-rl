"""DDPG: deep deterministic policy gradient (continuous actions only).

Critic: regress Q(s, a) on the one-step target computed with the slow target networks

    y = r + g (1 - done) Q'(s', mu'(s'))          loss_Q = mean (Q(s, a) - y)^2

Actor: follow the critic's gradient with respect to the action (deterministic policy gradient)

    loss_mu = - mean Q(s, mu(s))

Targets track the online nets with Polyak averaging, theta' <- (1 - tau) theta' + tau theta.
Exploration adds Gaussian noise N(0, explore_noise) to mu(s) and clips to [-1, 1].
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..networks import DeterministicActor, QCritic, soft_update
from .offpolicy import OffPolicyAgent


class DDPGAgent(OffPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        assert cfg.continuous, "DDPG needs the continuous LunarLander"
        super().__init__(cfg, obs_dim, n_actions)
        self.actor = DeterministicActor(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.critic = QCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.actor_t = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_t = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.nets = {"actor": self.actor, "critic": self.critic}

    def greedy(self, obs):
        return self.actor(obs)

    def explore(self, obs):
        a = self.act(obs)
        return np.clip(a + self.cfg.explore_noise * np.random.randn(self.n_actions), -1, 1).astype(np.float32)

    def update(self, b) -> dict:
        with torch.no_grad():
            y = b["rew"] + b["discount"] * (1 - b["done"]) * self.critic_t(b["next_obs"], self.actor_t(b["next_obs"]))
        q_loss = (self.critic(b["obs"], b["act"]) - y).pow(2).mean()
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        pi_loss = -self.critic(b["obs"], self.actor(b["obs"])).mean()
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic_t, self.critic, self.cfg.tau)
        soft_update(self.actor_t, self.actor, self.cfg.tau)
        return dict(q_loss=q_loss.item(), pi_loss=pi_loss.item())
