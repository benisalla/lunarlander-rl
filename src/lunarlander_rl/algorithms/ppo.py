"""PPO: proximal policy optimisation with the clipped surrogate objective.

After each rollout (``n_envs`` x ``n_steps`` transitions, GAE advantages), do ``n_epochs``
passes over it in shuffled minibatches of ``batch_size`` and minimise

    ratio   = pi_new(a|s) / pi_old(a|s) = exp(logp_new - logp_old)
    L_clip  = - mean[ min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) ]
    loss    = L_clip + vf_coef * mean[(V(s) - R)^2] - ent_coef * H[pi]

The clip removes the incentive to push the ratio outside [1 - eps, 1 + eps], so the many
gradient steps on the same data cannot move the policy too far from the one that collected it
(a cheap trust region). This is what lets PPO reuse each sample several times where A2C uses it
once. The same code runs the discrete (Categorical) and continuous (Gaussian) action spaces.
"""

from __future__ import annotations

import torch

from .rollout import OnPolicyAgent


class PPOAgent(OnPolicyAgent):
    def __init__(self, cfg, obs_dim, n_actions):
        super().__init__(cfg, obs_dim, n_actions)
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)

    def update(self, b) -> dict:
        cfg = self.cfg
        n = len(b["obs"])
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, clip_frac=0.0, approx_kl=0.0)
        count = 0
        for _ in range(cfg.n_epochs):
            for idx in torch.randperm(n, device=self.device).split(cfg.batch_size):
                adv = b["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                dist = self.actor.dist(b["obs"][idx])
                logp = dist.log_prob(b["act"][idx])
                log_ratio = logp - b["logp"][idx]
                ratio = log_ratio.exp()
                pi_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                v_loss = (self.critic(b["obs"][idx]) - b["ret"][idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                self.clip_actor_critic()
                self.opt.step()
                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                count += 1
        return {k: v / count for k, v in stats.items()}

    def learn(self, envs, tracker):
        while not tracker.done:
            stats = self.update(self.collect(envs, tracker))
            for k, v in stats.items():
                tracker.log(k, v)
