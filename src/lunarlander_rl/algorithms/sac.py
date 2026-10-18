"""SAC: soft actor-critic for continuous actions (tanh-squashed Gaussian policy).

Maximise reward + alpha * entropy. With a ~ pi(.|s) sampled by reparameterisation:

    critic:  y = r + g (1 - done) [ min(Q1'(s', a'), Q2'(s', a')) - alpha log pi(a'|s') ],  a' ~ pi(.|s')
             loss_Q = mean (Q1(s, a) - y)^2 + (Q2(s, a) - y)^2
    actor:   loss_pi = mean [ alpha log pi(a~|s) - min(Q1(s, a~), Q2(s, a~)) ],   a~ = tanh(mu + sigma eps)
    alpha:   loss_a  = - mean [ log alpha * (log pi(a~|s) + target_entropy) ],  target_entropy = -dim(A)

The temperature alpha rises when the policy is less random than the target entropy and falls
when it is more random, so exploration is tuned automatically. Twin critics and Polyak targets
are the same trick as in TD3; there is no target actor (the policy itself is stochastic).
"""

from __future__ import annotations

import copy

import torch

from ..networks import QCritic, TanhGaussianActor, soft_update
from .offpolicy import OffPolicyAgent


class SACAgent(OffPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        assert cfg.continuous, "this SAC is for the continuous LunarLander; use sac-discrete for the discrete one"
        super().__init__(cfg, obs_dim, n_actions)
        self.actor = TanhGaussianActor(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q1 = QCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q2 = QCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q1_t = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_t = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(float(cfg.alpha), device=self.device).log().requires_grad_(cfg.auto_alpha)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.actor_lr)
        self.target_entropy = -float(n_actions)
        self.nets = {"actor": self.actor, "q1": self.q1, "q2": self.q2}

    def greedy(self, obs):
        return self.actor.greedy(obs)

    @torch.no_grad()
    def explore(self, obs):
        return self.actor.sample(self.t(obs).unsqueeze(0))[0][0].cpu().numpy()

    def update(self, b) -> dict:
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(b["next_obs"])
            q_next = torch.min(self.q1_t(b["next_obs"], a_next), self.q2_t(b["next_obs"], a_next))
            y = b["rew"] + b["discount"] * (1 - b["done"]) * (q_next - alpha * logp_next)
        q_loss = (self.q1(b["obs"], b["act"]) - y).pow(2).mean() + (self.q2(b["obs"], b["act"]) - y).pow(2).mean()
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor.sample(b["obs"])
        q = torch.min(self.q1(b["obs"], a), self.q2(b["obs"], a))
        pi_loss = (alpha * logp - q).mean()
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()

        if self.cfg.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        soft_update(self.q1_t, self.q1, self.cfg.tau)
        soft_update(self.q2_t, self.q2, self.cfg.tau)
        return dict(q_loss=q_loss.item(), pi_loss=pi_loss.item(), alpha=alpha.item(), entropy=-logp.mean().item())

    def state_dict(self):
        return {**super().state_dict(), "log_alpha": self.log_alpha.detach().clone()}

    def load_state_dict(self, state):
        state = dict(state)
        la = state.pop("log_alpha", None)
        super().load_state_dict(state)
        if la is not None:
            self.log_alpha.data.copy_(la)
