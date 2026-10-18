"""A2C: synchronous advantage actor-critic.

Collect a short rollout (``n_steps`` per env, e.g. 16 x 8 envs = 128 transitions), compute
advantages with GAE, then take ONE gradient step on

    loss = - mean[ log pi(a|s) * A ]  + vf_coef * mean[(V(s) - R)^2]  - ent_coef * H[pi]

and throw the data away. Compared with REINFORCE it bootstraps from V (it does not wait for the
end of the episode), so it updates much more often with lower-variance targets. Compared with
PPO it uses every sample once and has no trust region, so it is simpler but less sample
efficient and more sensitive to the learning rate.
"""

from __future__ import annotations

import torch

from .rollout import OnPolicyAgent


class A2CAgent(OnPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)

    def update(self, b) -> dict:
        dist = self.actor.dist(b["obs"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pi_loss = -(dist.log_prob(b["act"]) * adv).mean()
        v_loss = (self.critic(b["obs"]) - b["ret"]).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        self.clip_actor_critic()
        self.opt.step()
        return dict(pi_loss=pi_loss.item(), v_loss=v_loss.item(), entropy=entropy.item())

    def learn(self, envs, tracker):
        while not tracker.done:
            stats = self.update(self.collect(envs, tracker))
            for k, v in stats.items():
                tracker.log(k, v)
