"""REINFORCE with a learned value baseline (Monte-Carlo policy gradient).

Play ``episodes_per_update`` whole episodes with the current policy, then for every step t:

    G_t   = r_t + g r_{t+1} + g^2 r_{t+2} + ...        (reward-to-go, computed backwards)
    A_t   = G_t - V(s_t)                                (baseline: same expectation, lower variance)
    loss  = - mean_t [ log pi(a_t|s_t) * A_t ]  - ent_coef * H[pi(.|s_t)]
          + vf_coef * mean_t (V(s_t) - G_t)^2

The gradient of the first term is the policy-gradient theorem estimate. Nothing bootstraps:
G_t is the real sampled return, which is unbiased but noisy, hence the baseline and the
advantage normalisation. Works with the discrete (Categorical) and continuous (Gaussian) env.
"""

from __future__ import annotations

import numpy as np
import torch

from ..networks import CategoricalActor, GaussianActor, ValueNet
from .base import Agent, clip_grads


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    out, g = np.zeros(len(rewards), np.float32), 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


class ReinforceAgent(Agent):
    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        Actor = GaussianActor if cfg.continuous else CategoricalActor
        self.actor = Actor(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.critic = ValueNet(obs_dim, cfg.hidden).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.nets = {"actor": self.actor, "critic": self.critic}

    def greedy(self, obs):
        return self.actor.greedy(obs)

    def update(self, obs, actions, returns) -> dict:
        obs, returns = self.t(obs), self.t(returns)
        actions = self.t(actions, torch.float32 if self.cfg.continuous else torch.int64)
        dist = self.actor.dist(obs)
        values = self.critic(obs)
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pi_loss = -(dist.log_prob(actions) * adv).mean()
        v_loss = (values - returns).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        clip_grads(self.actor.parameters(), self.cfg.max_grad_norm)  # separately: see rollout.py
        clip_grads(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return dict(pi_loss=pi_loss.item(), v_loss=v_loss.item(), entropy=entropy.item())

    def learn(self, env, tracker):
        cfg = self.cfg
        env.reset(seed=cfg.seed)
        while not tracker.done:
            batch_obs, batch_act, batch_ret = [], [], []
            for _ in range(cfg.episodes_per_update):
                obs, _ = env.reset()
                rewards, done = [], False
                while not done:
                    with torch.no_grad():
                        a = self.actor.dist(self.t(obs).unsqueeze(0)).sample()[0].cpu().numpy()
                    env_a = np.clip(a, -1, 1) if cfg.continuous else int(a)
                    next_obs, r, term, trunc, _ = env.step(env_a)
                    batch_obs.append(obs)
                    batch_act.append(a)
                    rewards.append(r)
                    obs, done = next_obs, term or trunc
                    tracker.add_steps(1)
                tracker.add_episode(sum(rewards), len(rewards))
                batch_ret.append(discounted_returns(rewards, cfg.gamma))
                if tracker.done:
                    break
            stats = self.update(np.array(batch_obs), np.array(batch_act), np.concatenate(batch_ret))
            for k, v in stats.items():
                tracker.log(k, v)
