"""DQN and its variants: Double DQN, Dueling DQN, prioritized replay, n-step returns.

One gradient step on a replay batch (s, a, R, s', done, discount):

    y      = R + discount * (1 - done) * Q_target(s', a*)
    a*     = argmax_a Q_target(s', a)          (DQN: the target net picks and scores)
    a*     = argmax_a Q_online(s', a)          (Double DQN: online picks, target scores -> less overestimation)
    loss   = mean_i  w_i * Huber(Q_online(s_i, a_i) - y_i)

``w_i`` is 1 for uniform replay and the importance weight for prioritized replay, whose
priorities are then set to |y_i - Q(s_i, a_i)|. With n-step replay, R is the discounted sum of
n rewards and discount = gamma^n (see ``buffers.py``). The target network is a copy of the
online one, refreshed every ``target_update`` gradient steps, so y does not move while we
regress towards it. Exploration is epsilon-greedy with epsilon decayed linearly.
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from ..buffers import PrioritizedReplayBuffer, ReplayBuffer
from ..networks import DuelingQNet, QNet
from .base import Agent, clip_grads


class DQNAgent(Agent):
    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        Net = DuelingQNet if cfg.dueling else QNet
        self.q = Net(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        Buffer = PrioritizedReplayBuffer if cfg.per else ReplayBuffer
        kw = dict(alpha=cfg.per_alpha) if cfg.per else {}
        self.buffer = Buffer(obs_dim, 1, cfg.buffer_size, n_step=cfg.n_step, gamma=cfg.gamma, **kw)
        self.nets = {"q": self.q}
        self.n_updates = 0

    def greedy(self, obs):
        return self.q(obs).argmax(-1)

    def epsilon(self, progress: float) -> float:
        frac = min(1.0, progress / self.cfg.eps_fraction)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def update(self, b: dict) -> float:
        q = self.q(b["obs"]).gather(1, b["act"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double:
                a_next = self.q(b["next_obs"]).argmax(1, keepdim=True)
                q_next = self.q_target(b["next_obs"]).gather(1, a_next).squeeze(1)
            else:
                q_next = self.q_target(b["next_obs"]).max(1).values
            y = b["rew"] + b["discount"] * (1 - b["done"]) * q_next
        td = y - q
        loss = (b["weights"] * F.smooth_l1_loss(q, y, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        clip_grads(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.buffer.update_priorities(b["idx"], td.abs().detach().cpu().numpy())
        self.n_updates += 1
        if self.n_updates % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def learn(self, env, tracker):
        cfg = self.cfg
        obs, _ = env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)  # reproducible warm-up / epsilon actions
        ep_ret, ep_len = 0.0, 0
        while not tracker.done:
            if random.random() < self.epsilon(tracker.progress()):
                action = env.action_space.sample()
            else:
                action = self.act(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            self.buffer.add(obs, action, r, next_obs, term, trunc)
            obs, ep_ret, ep_len = next_obs, ep_ret + r, ep_len + 1
            if term or trunc:
                tracker.add_episode(ep_ret, ep_len)
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
            if tracker.steps >= cfg.learning_starts and tracker.steps % cfg.train_freq == 0:
                if cfg.per:  # anneal the importance-sampling exponent beta towards 1
                    self.buffer.beta = cfg.per_beta0 + tracker.progress() * (1 - cfg.per_beta0)
                for _ in range(cfg.gradient_steps):
                    loss = self.update(self.buffer.sample(cfg.batch_size))
                tracker.log("q_loss", loss)
                tracker.log("epsilon", self.epsilon(tracker.progress()))
            tracker.add_steps(1)
