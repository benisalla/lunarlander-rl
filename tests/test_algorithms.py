"""Every agent: one update on a random batch, save / load round trip, and it learns a bandit."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from lunarlander_rl import METHODS, build_agent, make_config
from lunarlander_rl.envs import set_seed
from lunarlander_rl.metrics import Tracker

from conftest import ContextBandit, bandit_score

SMALL = dict(hidden=(32, 32), batch_size=32, learning_starts=200, buffer_size=5000, eval_every=10**9)


def fill(agent, n=300):
    rng = np.random.default_rng(0)
    for _ in range(n):
        a = rng.uniform(-1, 1, 2).astype(np.float32) if agent.cfg.continuous else int(rng.integers(4))
        agent.buffer.add(rng.normal(size=8), a, float(rng.normal()), rng.normal(size=8), False, False)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_act_save_load_roundtrip(method):
    cfg = make_config(method, **SMALL)
    agent = build_agent(cfg)
    obs = np.random.default_rng(0).normal(size=8).astype(np.float32)
    a = agent.act(obs)
    if cfg.continuous:
        assert a.shape == (2,) and np.all(np.abs(a) <= 1)
    else:
        assert a in range(4)
    other = build_agent(cfg)
    other.load_state_dict(agent.state_dict())
    for _ in range(5):
        o = np.random.normal(size=8).astype(np.float32)
        assert np.allclose(agent.act(o), other.act(o))


@pytest.mark.parametrize("method", [m for m, p in METHODS.items() if p.algo in ("dqn", "ddpg", "td3", "sac", "sac_discrete")])
def test_replay_update_changes_weights(method):
    agent = build_agent(make_config(method, **SMALL))
    fill(agent)
    before = [p.detach().clone() for net in agent.nets.values() for p in net.parameters()]
    for _ in range(3):
        stats = agent.update(agent.buffer.sample(32))
        agent.n_updates += 1
    after = [p for net in agent.nets.values() for p in net.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    loss = stats if isinstance(stats, float) else stats["q_loss"]
    assert np.isfinite(loss)


LEARN = {
    # method: (steps, overrides, threshold); discrete chance = 0.25, continuous random ~ -0.9
    "dqn": (1500, dict(train_freq=1, gradient_steps=1, eps_fraction=0.3, target_update=50, lr=1e-3), 0.7),
    "dueling-dqn": (1500, dict(train_freq=1, gradient_steps=1, eps_fraction=0.3, target_update=50, lr=1e-3), 0.7),
    "reinforce": (4000, dict(episodes_per_update=32, lr=3e-3, ent_coef=0.0), 0.7),
    "a2c": (6000, dict(n_envs=8, n_steps=4, lr=3e-3, ent_coef=0.0), 0.7),
    "ppo": (4000, dict(n_envs=8, n_steps=64, batch_size=64, n_epochs=4, lr=3e-3, ent_coef=0.0), 0.7),
    "sac-discrete": (1200, dict(lr=1e-3, target_entropy_scale=0.1), 0.7),
    "td3": (1500, dict(actor_lr=1e-3, critic_lr=1e-3), -0.15),
    "sac": (900, dict(actor_lr=1e-3, critic_lr=1e-3), -0.15),
    "ppo-continuous": (8000, dict(n_envs=8, n_steps=64, batch_size=64, n_epochs=8, lr=3e-3, ent_coef=0.0), -0.25),
}


@pytest.mark.parametrize("method", sorted(LEARN))
def test_learns_a_contextual_bandit(method):
    steps, over, threshold = LEARN[method]
    set_seed(0)
    cfg = make_config(method, **{**SMALL, "total_steps": steps, "seed": 0, "gamma": 0.0, **over})
    agent = build_agent(cfg)
    if agent.vectorized:
        env = gym.vector.SyncVectorEnv([lambda: ContextBandit(cfg.continuous) for _ in range(cfg.n_envs)],
                                       autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        env = ContextBandit(cfg.continuous)
    tracker = Tracker(cfg, agent, verbose=False, eval_fn=lambda ag: [bandit_score(ag, cfg.continuous, 50)])
    agent.learn(env, tracker)
    assert bandit_score(agent, cfg.continuous) > threshold
