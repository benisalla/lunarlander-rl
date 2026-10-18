"""The interaction loop shared by the replay-based actor-critics (DDPG, TD3, SAC, SAC-discrete).

    for every env step:
        a = random action          if step < learning_starts   (fills the buffer with diverse data)
            agent.explore(s)       otherwise                   (noisy / stochastic policy)
        store (s, a, r, s', terminated) in the replay buffer
        every train_freq steps: gradient_steps x agent.update(buffer.sample(batch_size))
"""

from __future__ import annotations

from ..buffers import ReplayBuffer
from .base import Agent


class OffPolicyAgent(Agent):
    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        self.buffer = ReplayBuffer(obs_dim, n_actions, cfg.buffer_size, n_step=cfg.n_step, gamma=cfg.gamma,
                                   discrete=not cfg.continuous)
        self.n_updates = 0

    def explore(self, obs):
        raise NotImplementedError

    def update(self, batch) -> dict:
        raise NotImplementedError

    def learn(self, env, tracker):
        cfg = self.cfg
        obs, _ = env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)  # reproducible warm-up / epsilon actions
        ep_ret, ep_len = 0.0, 0
        while not tracker.done:
            if tracker.steps < cfg.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.explore(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            self.buffer.add(obs, action, r, next_obs, term, trunc)
            obs, ep_ret, ep_len = next_obs, ep_ret + r, ep_len + 1
            if term or trunc:
                tracker.add_episode(ep_ret, ep_len)
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
            if tracker.steps >= cfg.learning_starts and tracker.steps % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    stats = self.update(self.buffer.sample(cfg.batch_size))
                    self.n_updates += 1
                for k, v in stats.items():
                    tracker.log(k, v)
            tracker.add_steps(1)
