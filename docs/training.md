# Training

```bash
lunarlander-train --list                                   # methods, env, default budget, one-line idea
lunarlander-train --method ppo                             # preset budget -> runs/ppo
lunarlander-train --method td3 --steps 50000 --stop-when-solved
lunarlander-train --method dqn --set double=true --set n_step=3 --set lr=1e-3
lunarlander-train --method sac --threads 4 --max-minutes 20
```

Every field of [`Config`](../src/lunarlander_rl/config.py) can be set with `--set key=value`;
the full config is saved to `runs/<method>/config.json`. From Python:

```python
from lunarlander_rl import make_config, train
agent, summary = train(make_config("dueling-dqn", total_steps=100_000), "runs/dueling-dqn")
```

## Presets and why these values

| method | budget | key settings | why |
|---|---|---|---|
| `dqn`, `double-dqn` | 150k | 128x128, lr 6.3e-4, batch 128, buffer 50k, 2 gradient steps every 4 env steps, target copy every 250 updates, epsilon 1 -> 0.05 over 12% | the tuned rl-baselines3-zoo LunarLander setting, with half the gradient steps to fit one CPU thread |
| `dueling-dqn` | 150k | as DQN + double + dueling + PER (alpha 0.6, beta 0.4 -> 1) + 3-step | all the Rainbow-style additions that apply without distributional RL or noisy nets |
| `reinforce` | 1M | 64x64 tanh, lr 1e-3, 4 episodes per update, gamma 0.999 | gamma close to 1: a landing pays off hundreds of steps later |
| `a2c` | 1M | 8 envs x 16 steps, GAE 0.95, lr 7e-4, entropy 0.01 | short rollouts = many small updates |
| `ppo` | 1M | 16 envs x 1024 steps, 4 epochs, minibatch 64, GAE 0.98, gamma 0.999, clip 0.2, entropy 0.01 | zoo setting; PPO needs many env steps but they are cheap |
| `sac-discrete` | 100k | 128x128, lr 3e-4, batch 128, 1 update every 2 steps, target entropy 0.2 log 4 | with 0.5 log 4 alpha stayed at ~0.7 and the greedy policy swung between +177 and -580 from one evaluation to the next; a lower target lets the policy commit |
| `ddpg`, `td3`, `sac` | 120k | 128x128, lr 1e-3 (SAC 7.3e-4, tau 0.01), batch 128, 5k random warm-up steps, 1 update every 2 steps | SAC/TD3 normally use 256x256 and one update per step; halved for one CPU thread |
| `ppo-continuous` | 1M | as `ppo`, Gaussian policy with learned log std | |

`--stop-when-solved` ends training at the first periodic evaluation >= 200 (the README runs use
it to save CPU time); `--max-minutes` caps wall time.

## Where the time goes

On one CPU thread the cost is dominated by gradient steps, not by the simulator (a LunarLander
step is ~0.1 ms):

* replay methods do one gradient step (or more) every few env steps, each on a batch of 128;
  TD3/SAC update an actor and two critics, so they are 2-3x DQN's cost per update;
* PPO does 4 epochs x 256 minibatches per 16k-step rollout = 1 gradient step every 16 env
  steps, on 64x64 networks, so a million steps is cheaper than 150k DQN steps.

`--threads N` lets torch use more cores (useful from 256x256 networks up); the notebook uses a
GPU when one is present (`--set device=cuda`), which mostly helps the replay methods.

## Tuning tips specific to LunarLander

* **Hovering plateau.** Returns stuck around -50 to +50 with 1000-step episodes: the agent
  learned not to crash but not to land. More exploration (higher entropy coefficient, slower
  epsilon decay) or simply more steps gets it out. Bootstrapping through truncation (done here)
  matters for this.
* **gamma.** 0.99 works for the replay methods; on-policy methods do better with 0.999 because
  their advantage estimates do not reuse a learned Q over long horizons.
* **DDPG instability.** If the return collapses after rising, that is Q overestimation; TD3 is
  the fix, not a smaller learning rate.
* **Discrete SAC.** If the policy stays random (entropy near log 4 = 1.39), lower
  `target_entropy_scale`; if it collapses early, raise it.
