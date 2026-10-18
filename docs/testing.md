# Tests

```bash
OMP_NUM_THREADS=1 pytest -q          # 47 tests, about 1 minute on one CPU
```

| file | what it checks |
|---|---|
| `test_config.py` | every preset builds; overrides, CLI `key=value` casting, JSON round trip |
| `test_buffers.py` | 1-step and n-step returns, discounts and `done` flags by hand; truncation flushes but still bootstraps; ring overwrite; the sum tree samples proportionally to priority; PER prefers high-TD transitions and normalises importance weights |
| `test_networks.py` | output shapes of every network; dueling mean-advantage identity; the tanh-Gaussian log-prob equals torch's `TanhTransform` density; Polyak update |
| `test_returns.py` | reward-to-go; GAE with lambda = 1 equals Monte-Carlo returns and lambda = 0 equals the one-step TD error, across an episode boundary |
| `test_algorithms.py` | for all 11 methods: act in the right action format, save/load gives identical actions; replay methods: an update changes the weights with a finite loss; **learning check**: 9 methods (DQN, dueling DQN + PER + n-step, REINFORCE, A2C, PPO, discrete SAC, TD3, SAC, continuous PPO) learn a contextual bandit with LunarLander's shapes far above chance in a few thousand steps |
| `test_end_to_end.py` | the real LunarLander: env shapes, the heuristic beats random, a 600-step training run saves / reloads / evaluates and `compare` writes the table and figures, a GIF is written |

## Why a contextual bandit for the learning check

LunarLander needs 10^5 steps and minutes per method: too slow for a test. The bandit
(`tests/conftest.py`) keeps the exact interface (8-float observations; 4 discrete actions or 2
floats in [-1, 1]) but every episode is one step and the best action is a simple function of
the observation (argmax of the first 4 floats; or half the first two floats). An agent whose
update rule is wrong (sign error, wrong target, detached gradient, wrong log-prob) stays at
chance; a correct one gets near-perfect in seconds. DDPG is covered by the update test only:
it is TD3 without its three fixes and is too noisy for a tight threshold in 1000 steps.
