"""TD3: twin delayed DDPG. Three fixes to DDPG's Q overestimation, each one line below.

1. Twin critics, pessimistic target:  y = r + g (1 - done) min(Q1'(s', a'), Q2'(s', a'))
2. Target policy smoothing:           a' = clip(mu'(s') + clip(N(0, target_noise), -c, c), -1, 1)
3. Delayed actor:                     the actor and the targets move once every ``policy_delay``
                                      critic steps, so the critic is closer to fitted when the
                                      actor follows its gradient  loss_mu = - mean Q1(s, mu(s)).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..networks import DeterministicActor, QCritic, soft_update
from .offpolicy import OffPolicyAgent


class TD3Agent(OffPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        assert cfg.continuous, "TD3 needs the continuous LunarLander"
        super().__init__(cfg, obs_dim, n_actions)
        self.actor = DeterministicActor(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q1 = QCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q2 = QCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.actor_t = copy.deepcopy(self.actor).requires_grad_(False)
        self.q1_t = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_t = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)
        self.nets = {"actor": self.actor, "q1": self.q1, "q2": self.q2}
        self._last_pi_loss = 0.0

    def greedy(self, obs):
        return self.actor(obs)

    def explore(self, obs):
        a = self.act(obs)
        return np.clip(a + self.cfg.explore_noise * np.random.randn(self.n_actions), -1, 1).astype(np.float32)

    def update(self, b) -> dict:
        cfg = self.cfg
        with torch.no_grad():
            noise = (torch.randn_like(b["act"]) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            a_next = (self.actor_t(b["next_obs"]) + noise).clamp(-1, 1)
            q_next = torch.min(self.q1_t(b["next_obs"], a_next), self.q2_t(b["next_obs"], a_next))
            y = b["rew"] + b["discount"] * (1 - b["done"]) * q_next
        q_loss = (self.q1(b["obs"], b["act"]) - y).pow(2).mean() + (self.q2(b["obs"], b["act"]) - y).pow(2).mean()
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        if self.n_updates % cfg.policy_delay == 0:
            pi_loss = -self.q1(b["obs"], self.actor(b["obs"])).mean()
            self.actor_opt.zero_grad()
            pi_loss.backward()
            self.actor_opt.step()
            self._last_pi_loss = pi_loss.item()
            for t, s in ((self.actor_t, self.actor), (self.q1_t, self.q1), (self.q2_t, self.q2)):
                soft_update(t, s, cfg.tau)
        return dict(q_loss=q_loss.item(), pi_loss=self._last_pi_loss)
