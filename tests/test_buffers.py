import numpy as np
import pytest

from lunarlander_rl.buffers import PrioritizedReplayBuffer, ReplayBuffer, SumTree


def obs(i):
    return np.full(8, i, np.float32)


def test_one_step_transitions_are_stored_as_is():
    b = ReplayBuffer(8, 1, 10, n_step=1, gamma=0.9)
    b.add(obs(0), 1, 2.0, obs(1), False, False)
    assert len(b) == 1 and b.rew[0] == 2.0 and b.discount[0] == pytest.approx(0.9) and b.done[0] == 0
    assert b.next_obs[0, 0] == 1


def test_n_step_return_discount_and_termination():
    g = 0.5
    b = ReplayBuffer(8, 1, 10, n_step=3, gamma=g)
    for t in range(4):  # rewards 1, 1, 1, 1 ; the 4th step terminates
        b.add(obs(t), 0, 1.0, obs(t + 1), t == 3, False)
    # (s0: 1 + .5 + .25, s3, discount .125), (s1: same, s4 terminal), then the tail flushes: s2 (2 steps), s3 (1 step)
    assert len(b) == 4
    assert b.rew[:4] == pytest.approx([1.75, 1.75, 1.5, 1.0])
    assert b.discount[:4] == pytest.approx([0.125, 0.125, 0.25, 0.5])
    assert list(b.done[:4]) == [0, 1, 1, 1]
    assert list(b.next_obs[:4, 0]) == [3, 4, 4, 4]


def test_truncation_flushes_but_is_not_terminal():
    b = ReplayBuffer(8, 1, 10, n_step=3, gamma=0.5)
    b.add(obs(0), 0, 1.0, obs(1), False, False)
    b.add(obs(1), 0, 1.0, obs(2), False, True)  # time limit
    assert len(b) == 2 and list(b.done[:2]) == [0, 0]  # still bootstraps from s'


def test_ring_buffer_overwrites_oldest():
    b = ReplayBuffer(8, 1, 3)
    for t in range(5):
        b.add(obs(t), 0, float(t), obs(t + 1), False, False)
    assert len(b) == 3 and sorted(b.rew.tolist()) == [2.0, 3.0, 4.0]


def test_sum_tree_finds_proportionally():
    tree = SumTree(4)
    tree.update(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert tree.total == pytest.approx(10.0)
    assert list(tree.find(np.array([0.5, 1.5, 3.5, 9.9]))) == [0, 1, 2, 3]
    counts = np.bincount(tree.find(np.random.default_rng(0).uniform(0, 10, 20000)), minlength=4)
    assert counts / counts.sum() == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=0.02)


def test_prioritized_replay_prefers_high_td_and_weights_are_normalised():
    b = PrioritizedReplayBuffer(8, 1, 64, alpha=1.0)
    for t in range(64):
        b.add(obs(t), 0, 0.0, obs(t), False, False)
    b.update_priorities(np.arange(64), np.where(np.arange(64) == 5, 100.0, 0.01))
    batch = b.sample(256)
    assert np.mean(batch["idx"] == 5) > 0.5
    w = batch["weights"].numpy()
    assert w.max() == pytest.approx(1.0) and w[batch["idx"] == 5].max() < 0.05
