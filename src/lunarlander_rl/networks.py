"""All the neural networks: small MLPs over the 8-float observation.

Shapes (B = batch, O = 8 observation floats, A = 4 discrete actions or 2 continuous ones):

* ``QNet``            O -> A         one Q-value per discrete action (DQN, Double DQN)
* ``DuelingQNet``     O -> V + A - mean(A)
* ``CategoricalActor`` O -> logits   pi(a|s) = softmax(logits)          (REINFORCE, A2C, PPO, SAC-discrete)
* ``GaussianActor``   O -> mean, plus a learned state-independent log std  (continuous PPO / A2C / REINFORCE)
* ``TanhGaussianActor`` O -> mean, log std; a = tanh(u), u ~ N           (SAC)
* ``DeterministicActor`` O -> tanh(.) in [-1, 1]^A                     (DDPG, TD3)
* ``QCritic``         (O, A) -> 1                                       (DDPG, TD3, SAC)
* ``ValueNet``        O -> 1                                            (baselines and critics for PG methods)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def mlp(inp: int, out: int, hidden=(256, 256), act=nn.ReLU, out_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    last = nn.Linear(d, out)
    if out_gain is not None:  # small last layer: start with near-uniform policies / near-zero values
        nn.init.orthogonal_(last.weight, out_gain)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (B, O) -> (B, A)
        return self.net(obs)


class DuelingQNet(nn.Module):
    """Q(s, a) = V(s) + A(s, a) - mean_a A(s, a). The mean makes V and A identifiable."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden[-1], hidden[:-1])
        self.value = nn.Sequential(nn.ReLU(), nn.Linear(hidden[-1], 1))
        self.adv = nn.Sequential(nn.ReLU(), nn.Linear(hidden[-1], n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64), act=nn.Tanh):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden, act, out_gain=0.01)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).argmax(-1)


class GaussianActor(nn.Module):
    """Diagonal Gaussian with a state-independent log std (the standard PPO choice).

    The sampled action is unbounded; the environment clips it to [-1, 1]. ``log_prob`` is the
    density of the unclipped sample, which is what the policy-gradient estimators need.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), act=nn.Tanh):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden, act, out_gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: torch.Tensor):
        return _IndependentNormal(self.net(obs), self.log_std.exp())

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).clamp(-1, 1)


class _IndependentNormal(Normal):
    """Normal whose log_prob / entropy are summed over action dimensions (one number per sample)."""

    def log_prob(self, value):
        return super().log_prob(value).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)


class TanhGaussianActor(nn.Module):
    """SAC actor: u ~ N(mu(s), sigma(s)), a = tanh(u) in [-1, 1].

    log pi(a|s) = log N(u) - sum log(1 - tanh(u)^2)   (change of variables through tanh).
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs: torch.Tensor):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        u = mu + log_std.exp() * torch.randn_like(mu)  # reparameterisation: gradients flow through u
        a = torch.tanh(u)
        logp = Normal(mu, log_std.exp()).log_prob(u).sum(-1)
        # log(1 - tanh(u)^2) written in a numerically stable form
        logp = logp - (2 * (math.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)
        return a, logp

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(obs)[0])


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    greedy = forward


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # -> (B,)
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(64, 64), act=nn.Tanh):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden, act, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # -> (B,)
        return self.net(obs).squeeze(-1)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """theta_target <- (1 - tau) * theta_target + tau * theta  (tau = 1 is a hard copy)."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1 - tau).add_(s, alpha=tau)
