import numpy as np
import pytest

from lunarlander_rl.algorithms.reinforce import discounted_returns
from lunarlander_rl.algorithms.rollout import gae


def test_discounted_returns():
    assert discounted_returns([1.0, 1.0, 1.0], 0.5) == pytest.approx([1.75, 1.5, 1.0])


def test_gae_lambda_one_is_monte_carlo_and_lambda_zero_is_td():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 1)).astype(np.float32)
    v = rng.normal(size=(5, 1)).astype(np.float32)
    ends = np.zeros((5, 1), np.float32)
    ends[2] = 1  # episode ends after step 2
    last = np.array([0.7], np.float32)
    g = 0.9
    a1 = gae(r, v, ends, last, g, 1.0)
    mc = np.zeros(5)
    mc[2] = r[2, 0]
    mc[1] = r[1, 0] + g * mc[2]
    mc[0] = r[0, 0] + g * mc[1]
    mc[4] = r[4, 0] + g * last[0]
    mc[3] = r[3, 0] + g * mc[4]
    assert (a1[:, 0] + v[:, 0]) == pytest.approx(mc, abs=1e-5)
    a0 = gae(r, v, ends, last, g, 0.0)
    nxt = np.append(v[1:, 0], last)
    td = r[:, 0] + g * nxt * (1 - ends[:, 0]) - v[:, 0]
    assert a0[:, 0] == pytest.approx(td, abs=1e-5)
