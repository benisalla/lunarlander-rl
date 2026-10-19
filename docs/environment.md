# The environment: LunarLander

`gymnasium` `LunarLander-v3` (Box2D physics). Installed with `pip install "gymnasium[box2d]"`;
the `box2d` package ships prebuilt wheels for Linux/macOS/Windows, so no `swig` or compiler is
needed. Code: [`src/lunarlander_rl/envs.py`](../src/lunarlander_rl/envs.py).

## What the agent sees

An 8-float observation, already normalised by the environment:

| index | meaning | typical range |
|---|---|---|
| 0, 1 | x, y position (pad centre is (0, 0)) | x in [-1, 1], y in [0, 1.5] |
| 2, 3 | vx, vy velocity | [-2, 2] |
| 4 | angle (0 = upright) | [-1, 1] rad |
| 5 | angular velocity | [-2, 2] |
| 6, 7 | left / right leg touches the ground | 0 or 1 |

The observation is the full physical state (no images, no history needed), which is why small
MLPs are enough and why every method here can be compared on the same networks.

## What the agent can do

| env | action | meaning |
|---|---|---|
| discrete (`continuous=False`) | `0, 1, 2, 3` | nothing, fire left engine, fire main engine, fire right engine |
| continuous (`continuous=True`) | `[main, lateral]` in [-1, 1]^2 | main fires at 50-100% power when main > 0; a side engine fires when abs(lateral) > 0.5 (sign picks the side) |

The discrete env is for value-based methods (DQN family) and categorical policies; the
continuous one is where DDPG, TD3, SAC and Gaussian PPO live. Note that the continuous env is
not "the discrete one with finer control": the engines have dead zones (main <= 0 is off,
abs(lateral) <= 0.5 is off), which a deterministic policy has to discover.

## How it is rewarded (reward shaping)

The per-step reward is the change of a potential function minus fuel:

```
Phi(s)   = -100 * |(x, y)| - 100 * |(vx, vy)| - 100 * |angle| + 10 * leg_left + 10 * leg_right
reward_t = Phi(s_{t+1}) - Phi(s_t) - 0.30 * main_engine_power - 0.03 * side_engine_power
```

plus `-100` and the episode ends if the lander crashes or leaves the screen, and `+100` when it
comes to rest (the body falls asleep in Box2D). `envs.shaping_potential(obs)` computes Phi, and
`examples/01_explore_the_env.py` checks the decomposition on every step of an episode.

Why it matters in plain words:

* **Potential-based shaping does not change the optimal policy** (Ng et al., 1999): summed over
  an episode, the Phi terms telescope to `Phi(final) - Phi(start)`, so it only tells the agent
  earlier, step by step, what the terminal outcome will reward anyway. That dense signal is
  what makes LunarLander learnable in 10^5 steps instead of 10^7.
* **Fuel cost** pushes towards short, efficient landings. Early on it also creates the classic
  local optimum: *hover forever*. Hovering avoids the -100 crash but costs fuel and gets cut at
  1000 steps (`truncated`), with a return around -50 to +100. Many learning curves plateau there
  before the agent learns to commit to touching down.
* **Leg contact +10 each** rewards touching the pad; losing contact costs it back, which
  discourages bouncing.
* **Solved = mean return >= 200** over evaluation episodes. A good landing scores 200-300; the
  heuristic controller that ships with gymnasium gets about 266 (discrete) and 288 (continuous) (see the README baselines).

## Episode end: terminated vs truncated

* `terminated`: crash, flew off screen, or landed and came to rest. This is a real terminal
  state: the value of what follows is 0.
* `truncated`: the 1000-step time limit. The episode is cut, not finished: the state still has
  a value. All methods here bootstrap through truncation (replay buffers store `done` only for
  `terminated`; the on-policy rollouts add `gamma * V(final_obs)` to the last reward). Treating
  truncation as terminal tells a hovering agent that hovering leads to value 0 at step 1000,
  which is wrong and slows learning.

## Vectorised environments

A2C and PPO step `n_envs` copies in lock-step with `gymnasium.vector.SyncVectorEnv` (one
process; LunarLander steps in ~0.1 ms so subprocesses would cost more than they save). It uses
`SAME_STEP` autoreset: when a copy's episode ends, `step` already returns the first observation
of its next episode and the true last observation is in `info["final_obs"]`, which is exactly
what the truncation bootstrap needs.

## Seeds

`train` seeds Python, NumPy and torch with `cfg.seed`, resets the training env with that seed,
and evaluates on episodes seeded `10000 + seed + i` (periodic) and `20000 + seed + i` (final), so
the evaluation episodes are the same for every method and never the ones it trained on.
