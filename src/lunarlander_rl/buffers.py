"""Replay buffers for the off-policy methods.

``ReplayBuffer`` stores transitions (s, a, R, s', done, discount) in pre-allocated numpy arrays
(a ring buffer). With ``n_step > 1`` it stores n-step transitions:

    R = r_t + g r_{t+1} + ... + g^{k-1} r_{t+k-1},   s' = s_{t+k},   discount = g^k

so every learner can use the same target ``R + discount * (1 - done) * bootstrap(s')``
whatever n is. ``done`` is True only for a real termination (crash / landing). A time-limit
truncation is not a terminal state, so the target still bootstraps from s'.

``PrioritizedReplayBuffer`` samples transition i with probability p_i^alpha / sum p^alpha
(p_i = |TD error| + eps) using a sum tree, and returns importance weights
w_i = (N P(i))^-beta / max w that correct the bias of non-uniform sampling.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, n_step: int = 1, gamma: float = 0.99,
                 discrete: bool = True, device: str = "cpu"):
        self.capacity, self.n_step, self.gamma, self.device = capacity, n_step, gamma, device
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity,) if discrete else (capacity, act_dim), np.int64 if discrete else np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)
        self.pos, self.size = 0, 0
        self._pending: deque = deque()  # last n raw transitions, waiting to become n-step ones

    def __len__(self) -> int:
        return self.size

    def add(self, obs, act, rew: float, next_obs, terminated: bool, truncated: bool) -> None:
        """Add one environment step. Call it for every step, in order, for a single env."""
        self._pending.append((obs, act, float(rew), next_obs, bool(terminated)))
        if len(self._pending) == self.n_step:
            self._commit()
        if terminated or truncated:  # episode over: flush the shorter tails too
            while self._pending:
                self._commit()

    def _commit(self) -> None:
        obs, act = self._pending[0][0], self._pending[0][1]
        ret, disc, next_obs, done = 0.0, 1.0, None, False
        for (_, _, r, nobs, term) in self._pending:
            ret += disc * r
            disc *= self.gamma
            next_obs, done = nobs, term
            if term:
                break
        self._store(obs, act, ret, next_obs, done, disc)
        self._pending.popleft()

    def _store(self, obs, act, ret, next_obs, done, disc) -> int:
        i = self.pos
        self.obs[i], self.act[i], self.rew[i] = obs, act, ret
        self.next_obs[i], self.done[i], self.discount[i] = next_obs, done, disc
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _batch(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        t = lambda x: torch.as_tensor(x[idx], device=self.device)  # noqa: E731
        return dict(obs=t(self.obs), act=t(self.act), rew=t(self.rew), next_obs=t(self.next_obs),
                    done=t(self.done), discount=t(self.discount))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        b = self._batch(idx)
        b["weights"] = torch.ones(batch_size, device=self.device)
        b["idx"] = idx
        return b

    def update_priorities(self, idx, td_abs) -> None:  # uniform replay ignores priorities
        pass


class SumTree:
    """Binary tree whose leaves hold priorities and whose inner nodes hold sums of their children.

    ``find(v)`` walks from the root to the leaf whose cumulative-priority interval contains v,
    in O(log N), for a whole batch of v at once.
    """

    def __init__(self, capacity: int):
        self.n = 1
        while self.n < capacity:
            self.n *= 2
        self.tree = np.zeros(2 * self.n, np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: np.ndarray, prio: np.ndarray) -> None:
        i = np.asarray(idx) + self.n
        self.tree[i] = prio
        i = np.unique(i // 2)
        while i[0] >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            if i[0] == 1:
                break
            i = np.unique(i // 2)

    def find(self, values: np.ndarray) -> np.ndarray:
        i = np.ones(len(values), np.int64)
        v = values.astype(np.float64).copy()
        while i[0] < self.n:
            left = 2 * i
            go_right = v > self.tree[left]
            v -= self.tree[left] * go_right
            i = left + go_right
        return i - self.n


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha: float = 0.6, eps: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.eps = alpha, eps
        self.tree = SumTree(self.capacity)
        self.max_prio = 1.0
        self.beta = 0.4  # annealed to 1 by the learner

    def _store(self, *args) -> int:
        i = super()._store(*args)
        self.tree.update(np.array([i]), np.array([self.max_prio ** self.alpha]))  # new = max priority
        return i

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        # stratified sampling: one uniform draw inside each of batch_size equal slices of the total
        bounds = np.linspace(0, self.tree.total, batch_size + 1)
        values = np.random.uniform(bounds[:-1], bounds[1:])
        idx = np.minimum(self.tree.find(values), self.size - 1)
        probs = self.tree.tree[idx + self.tree.n] / self.tree.total
        weights = (self.size * np.maximum(probs, 1e-12)) ** (-self.beta)
        weights /= weights.max()
        b = self._batch(idx)
        b["weights"] = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        b["idx"] = idx
        return b

    def update_priorities(self, idx, td_abs) -> None:
        prio = np.asarray(td_abs, np.float64) + self.eps
        self.max_prio = max(self.max_prio, float(prio.max()))
        self.tree.update(idx, prio ** self.alpha)
