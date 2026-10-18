"""The real LunarLander: train a few hundred steps, save, reload, evaluate, compare."""

import json

import numpy as np

from lunarlander_rl import evaluate, load_agent, make_config, make_env, train
from lunarlander_rl.compare import load_runs, main as compare_main, results_table
from lunarlander_rl.evaluate import BASELINES
from lunarlander_rl.play import record_episode, save_gif


def test_env_shapes():
    for cont, shape in ((False, ()), (True, (2,))):
        env = make_env(cont, seed=0)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8,) and env.action_space.sample().shape == shape


def test_heuristic_baseline_beats_random():
    h = evaluate(None, False, 2, policy=BASELINES["heuristic"])
    r = evaluate(None, False, 2, policy=BASELINES["random"])
    assert h.mean() > r.mean() and h.mean() > 100


def test_train_save_load_compare(tmp_path):
    cfg = make_config("dqn", total_steps=600, learning_starts=200, eval_every=300, eval_episodes=1,
                      final_eval_episodes=2, hidden=(32, 32))
    agent, summary = train(cfg, tmp_path / "runs" / "dqn", verbose=False)
    m = json.loads((tmp_path / "runs" / "dqn" / "metrics.json").read_text())
    assert len(m["evals"]) == 2 and m["summary"]["final_n"] == 2 and summary["steps"] == 600
    again, cfg2 = load_agent(tmp_path / "runs" / "dqn")
    o = np.zeros(8, np.float32)
    assert cfg2 == cfg and again.act(o) == agent.act(o)
    runs = load_runs(tmp_path / "runs")
    assert "| `dqn` | discrete |" in results_table(runs)
    compare_main([str(tmp_path / "runs"), "--out", str(tmp_path / "assets")])
    assert (tmp_path / "assets" / "results.md").exists() and (tmp_path / "assets" / "curves_discrete.png").exists()


def test_gif(tmp_path):
    frames, ret = record_episode(None, False, policy=BASELINES["heuristic"], max_steps=20)
    path = save_gif(frames, tmp_path / "x.gif")
    assert len(frames) == 21 and path.stat().st_size > 0


def test_tracker_evaluates_once_past_the_budget():
    from lunarlander_rl.metrics import Tracker

    cfg = make_config("ppo", total_steps=100, eval_every=40)
    calls = []
    tr = Tracker(cfg, None, verbose=False, eval_fn=lambda ag: calls.append(1) or [0.0])
    for _ in range(30):  # a vector env keeps stepping past the budget until its rollout ends
        tr.add_steps(8)
    assert [e[0] for e in tr.evals] == [40, 80, 104] and len(calls) == 3
