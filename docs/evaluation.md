# Evaluation and comparison

Code: [`evaluate.py`](../src/lunarlander_rl/evaluate.py), [`metrics.py`](../src/lunarlander_rl/metrics.py),
[`compare.py`](../src/lunarlander_rl/compare.py), [`viz.py`](../src/lunarlander_rl/viz.py).

## One protocol for every method

* **Greedy policy.** Exploration is switched off: argmax Q for the DQN family, argmax pi for
  categorical policies, the Gaussian mean for PPO/A2C/REINFORCE on the continuous env, the
  deterministic actor for DDPG/TD3, tanh(mean) for SAC. We measure what the agent learned, not
  how lucky its exploration noise was.
* **Fixed, unseen episodes.** Periodic evaluations use reset seeds `10000 + seed + i`, the final
  score `20000 + seed + i`; training episodes use other seeds.
* **Periodic evaluation** every `eval_every` env steps (10k by default) over `eval_episodes`
  episodes (10). These points are the learning curves.
* **Final score** at the end of training over `final_eval_episodes` (20) fresh episodes:
  mean +- std of the undiscounted return.
* **Solved** = mean return >= 200. *Steps to solve* is the env-step count of the first periodic
  evaluation whose mean reaches 200 (`--stop-when-solved` stops training there).

Training-episode returns are logged too (`metrics.json -> episodes`) and drawn in the per-run
figure, but they include exploration and are not used in the tables.

## What a run directory contains

```
runs/<method>/
  config.json    every hyperparameter (Config dataclass)
  model.pt       network weights ({"method", "state": {net name: state_dict}})
  metrics.json   summary, episodes [(step, return, length)], evals [(step, mean, std, seconds)],
                 losses {name: [(step, value)]}
```

## Commands

```bash
lunarlander-eval runs/ppo --episodes 100       # re-evaluate a saved agent
lunarlander-eval --baseline heuristic          # gymnasium's PID controller (~266 discrete, ~288 continuous)
lunarlander-eval --baseline random             # random actions (~-143)
lunarlander-compare runs --out assets          # results.md + curves_*.png
lunarlander-play runs/ppo --gif assets/ppo.gif # one landing as a GIF
python scripts/make_gifs.py                    # README before/after + solved-grid GIFs
```

## Before / after GIFs

`scripts/make_gifs.py` builds the README's visual comparison from the saved checkpoints:

* `assets/before_after_ppo.gif` and `assets/before_after_ppo-continuous.gif`: the same episode
  seed played by an untrained network of the same method (fresh random weights, left) and by
  the trained checkpoint (right), with each episode's return in the caption.
* `assets/solved_grid.gif`: one panel per method whose run reached a greedy evaluation of 200 or
  more and kept a final 20-episode score of at least 150, all on the same seed.

Short episodes freeze on their last frame until the longest one ends. Every third frame is kept,
downscaled to half size and reduced to 64 colours, so each file stays under 1 MB (`--every`,
`--scale` and `--seed` change that). The helpers are `play.before_after` and
`play.save_panels_gif`. If a checkpoint is missing, the script trains it first with the
method's preset; that takes minutes, so on a shared machine run it under the lock.

## Table columns

| column | meaning |
|---|---|
| steps to solve | first periodic evaluation with mean >= 200, or "not reached" |
| final return | mean +- std over 20 final episodes (bold when >= 200) |
| best eval | best periodic evaluation mean |
| steps run | env steps actually taken (less than the budget if it stopped when solved or hit `max_minutes`) |
| wall time | training time on one CPU thread (evaluations included, final evaluation not) |

## How to read the numbers

These are single-seed, short CPU runs. RL results vary a lot between seeds (a method can solve
at 80k steps with one seed and 200k with another), so the table shows that each method learns
and roughly how fast, not a ranking. For a fair ranking use several seeds
(`--seed 0 1 2 ...` in the notebook) and compare means with confidence intervals.

Wall time and sample efficiency are different axes: PPO needs many more environment steps than
the replay-based methods but each step is cheap, so it can win on wall time while losing on
steps. The compare command draws both (`curves_*.png` vs steps, `curves_*_time.png` vs minutes).
