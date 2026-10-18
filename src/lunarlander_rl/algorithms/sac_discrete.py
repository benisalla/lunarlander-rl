"""Discrete SAC (Christodoulou, 2019): soft actor-critic with a categorical policy.

With 4 actions the expectation over pi can be computed exactly instead of sampled:

    V'(s')  = sum_a' pi(a'|s') [ min(Q1'(s', a'), Q2'(s', a')) - alpha log pi(a'|s') ]
    critic: y = r + g (1 - done) V'(s')      loss_Q = mean (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2
    actor:  loss_pi = mean_s sum_a pi(a|s) [ alpha log pi(a|s) - min(Q1(s, a), Q2(s, a)) ]
    alpha:  loss_a  = mean_s [ log alpha * (H[pi(.|s)] - target_entropy) ]

The critics output one Q per action (like DQN), so no action input is needed. The target
entropy is a fraction ``target_entropy_scale`` of the maximum entropy log(4): the policy is
asked to stay that random, and alpha adjusts to make it so.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from ..networks import QNet, mlp, soft_update
from .offpolicy import OffPolicyAgent


class SACDiscreteAgent(OffPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        assert not cfg.continuous, "sac-discrete is for the discrete LunarLander; use sac for the continuous one"
        super().__init__(cfg, obs_dim, n_actions)
        self.actor = mlp(obs_dim, n_actions, cfg.hidden).to(self.device)  # logits
        self.q1 = QNet(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q2 = QNet(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q1_t = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_t = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.log_alpha = torch.tensor(float(cfg.alpha), device=self.device).log().requires_grad_(cfg.auto_alpha)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = cfg.target_entropy_scale * math.log(n_actions)
        self.nets = {"actor": self.actor, "q1": self.q1, "q2": self.q2}

    def greedy(self, obs):
        return self.actor(obs).argmax(-1)

    @torch.no_grad()
    def explore(self, obs):
        logits = self.actor(self.t(obs).unsqueeze(0))[0]
        return int(torch.distributions.Categorical(logits=logits).sample())

    def update(self, b) -> dict:
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            logp_next = F.log_softmax(self.actor(b["next_obs"]), -1)
            q_next = torch.min(self.q1_t(b["next_obs"]), self.q2_t(b["next_obs"]))
            v_next = (logp_next.exp() * (q_next - alpha * logp_next)).sum(-1)
            y = b["rew"] + b["discount"] * (1 - b["done"]) * v_next
        a = b["act"].unsqueeze(1)
        q1, q2 = self.q1(b["obs"]).gather(1, a).squeeze(1), self.q2(b["obs"]).gather(1, a).squeeze(1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        logp = F.log_softmax(self.actor(b["obs"]), -1)
        probs = logp.exp()
        with torch.no_grad():
            q = torch.min(self.q1(b["obs"]), self.q2(b["obs"]))
        pi_loss = (probs * (alpha * logp - q)).sum(-1).mean()
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()

        entropy = -(probs * logp).sum(-1).detach()
        if self.cfg.auto_alpha:
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        soft_update(self.q1_t, self.q1, self.cfg.tau)
        soft_update(self.q2_t, self.q2, self.cfg.tau)
        return dict(q_loss=q_loss.item(), pi_loss=pi_loss.item(), alpha=alpha.item(), entropy=entropy.mean().item())

    def state_dict(self):
        return {**super().state_dict(), "log_alpha": self.log_alpha.detach().clone()}

    def load_state_dict(self, state):
        state = dict(state)
        la = state.pop("log_alpha", None)
        super().load_state_dict(state)
        if la is not None:
            self.log_alpha.data.copy_(la)
