# Examples

Short scripts that use the `lunarlander_rl` package from Python, in the order one would use it.
Each one is a single command, runs on one CPU thread in a few minutes with small defaults, and
writes to `examples/output/` (gitignored). Run them from the repository root after
`pip install -e ".[dev]"`.

| script | what it shows | time |
|---|---|---|
| [01_explore_the_env.py](01_explore_the_env.py) | observation / action spaces, the reward decomposed into shaping potential + fuel + landing bonus (checked on every step), random vs heuristic baselines, a GIF of the heuristic landing | seconds |
| [02_train_dqn.py](02_train_dqn.py) | `make_config` + `train` for `dqn` / `double-dqn` / `dueling-dqn` (`--method`), then a greedy evaluation and the learning curve | ~2 min |
| [03_train_policy_gradient.py](03_train_policy_gradient.py) | `reinforce` / `a2c` / `ppo` (`--method`) on vectorised envs, and the logged diagnostics (policy loss, value loss, entropy, PPO clip fraction and KL) | ~2-4 min |
| [04_train_continuous.py](04_train_continuous.py) | the continuous env with `sac` / `td3` / `ddpg` / `ppo-continuous` (`--method`), and how SAC's temperature alpha moves | ~2-4 min |
| [05_compare_methods.py](05_compare_methods.py) | several methods with one budget, then `lunarlander_rl.compare`: the markdown table and the curves, like the README (`--all` runs every method) | ~5 min |
| [06_watch_and_record.py](06_watch_and_record.py) | `load_agent` on a saved run, evaluation next to the random / heuristic baselines, a GIF of its landing | seconds |

```bash
python examples/01_explore_the_env.py
python examples/02_train_dqn.py                 # --method double-dqn | dueling-dqn
python examples/03_train_policy_gradient.py     # --method reinforce | a2c | ppo
python examples/04_train_continuous.py          # --method sac | td3 | ddpg | ppo-continuous
python examples/05_compare_methods.py           # --all --steps 5000 to run every method once
python examples/06_watch_and_record.py examples/output/dqn
```

The short budgets show each method *learning* (the return climbing from about -200 towards 0
and beyond); reaching 200 takes the preset budgets of `lunarlander-train` (see the README
results) or the Kaggle/Colab notebook.
