"""On-policy rollouts for A2C and PPO: collect n_steps x n_envs transitions, then compute GAE.

Generalised advantage estimation, computed backwards over the rollout:

    delta_t = r_t + g * V(s_{t+1}) * (1 - end_t) - V(s_t)
    A_t     = delta_t + g * lambda * (1 - end_t) * A_{t+1}
    R_t     = A_t + V(s_t)                         (the regression target for the critic)

lambda = 0 gives the one-step TD advantage (low variance, biased), lambda = 1 the Monte-Carlo
advantage (unbiased, noisy). ``end_t`` is 1 when the episode ended at step t (crash, landing or
time limit). On a time-limit truncation the episode did not really end, so g * V(final obs) is
added to the last reward: the return is bootstrapped instead of being cut to zero.
"""

from __future__ import annotations

import numpy as np
import torch

from ..networks import CategoricalActor, GaussianActor, ValueNet
from .base import Agent, clip_grads


class OnPolicyAgent(Agent):
    """Actor + critic (separate MLPs) and the rollout / GAE machinery shared by A2C and PPO."""

    vectorized = True

    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        Actor = GaussianActor if cfg.continuous else CategoricalActor
        self.actor = Actor(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.critic = ValueNet(obs_dim, cfg.hidden).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.nets = {"actor": self.actor, "critic": self.critic}
        self._obs = None
        self._ep_ret = np.zeros(cfg.n_envs)
        self._ep_len = np.zeros(cfg.n_envs, np.int64)

    def greedy(self, obs):
        return self.actor.greedy(obs)

    def clip_actor_critic(self) -> None:
        """Clip the actor's and the critic's gradient norms separately.

        Early on the value loss is in the thousands (returns of +-100s) while the policy loss is
        ~0.01. With one shared clip the critic's gradient uses up the whole norm budget and the
        policy step shrinks to almost nothing (PPO then shows clip_frac = 0, approx_kl ~ 1e-4).
        """
        clip_grads(self.actor.parameters(), self.cfg.max_grad_norm)
        clip_grads(self.critic.parameters(), self.cfg.max_grad_norm)

    @torch.no_grad()
    def collect(self, envs, tracker) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        T, N = cfg.n_steps, cfg.n_envs
        if self._obs is None:
            self._obs, _ = envs.reset(seed=cfg.seed)
        obs_buf = np.zeros((T, N, self.obs_dim), np.float32)
        act_buf = np.zeros((T, N, self.n_actions) if cfg.continuous else (T, N), np.float32)
        logp_buf, rew_buf, end_buf, val_buf = (np.zeros((T, N), np.float32) for _ in range(4))
        for t in range(T):
            obs_t = self.t(self._obs)
            dist = self.actor.dist(obs_t)
            a = dist.sample()
            logp_buf[t] = dist.log_prob(a).cpu().numpy()
            val_buf[t] = self.critic(obs_t).cpu().numpy()
            a = a.cpu().numpy()
            env_a = np.clip(a, -1, 1) if cfg.continuous else a.astype(np.int64)
            next_obs, r, term, trunc, info = envs.step(env_a)
            r = r.astype(np.float32)
            self._ep_ret += r  # logged returns use the raw rewards
            self._ep_len += 1
            if trunc.any():  # bootstrap through time-limit truncations
                for i in np.where(trunc & ~term)[0]:
                    r[i] += cfg.gamma * self.critic(self.t(info["final_obs"][i]).unsqueeze(0)).item()
            obs_buf[t], act_buf[t], rew_buf[t], end_buf[t] = self._obs, a, r, term | trunc
            for i in np.where(term | trunc)[0]:
                tracker.add_episode(self._ep_ret[i], self._ep_len[i])
                self._ep_ret[i], self._ep_len[i] = 0.0, 0
            self._obs = next_obs
            tracker.add_steps(N)
        last_val = self.critic(self.t(self._obs)).cpu().numpy()
        adv = gae(rew_buf, val_buf, end_buf, last_val, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), device=self.device)  # noqa: E731
        act = flat(act_buf) if cfg.continuous else flat(act_buf).long()
        return dict(obs=flat(obs_buf), act=act, logp=flat(logp_buf), adv=flat(adv),
                    ret=flat(adv + val_buf), val=flat(val_buf))


def gae(rewards, values, ends, last_value, gamma: float, lam: float) -> np.ndarray:
    """All inputs are (T, N) except ``last_value`` (N,) = V(observation after the rollout)."""
    T = len(rewards)
    adv = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - ends[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv
