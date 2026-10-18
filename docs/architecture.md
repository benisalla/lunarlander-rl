# Architecture: every method, its update rule, and why

All methods share the same pieces: an environment ([envs.py](../src/lunarlander_rl/envs.py)), small
MLPs ([networks.py](../src/lunarlander_rl/networks.py)), one config ([config.py](../src/lunarlander_rl/config.py)),
the same evaluation ([evaluate.py](../src/lunarlander_rl/evaluate.py)) and the same bookkeeping
([metrics.py](../src/lunarlander_rl/metrics.py)). Each algorithm is one file in
[`algorithms/`](../src/lunarlander_rl/algorithms) with its equations in the docstring, so reading
a method means reading one file of 60-120 lines.

```
train(cfg) ─► build_agent(cfg) ─► agent.learn(env, tracker) ─► evaluate(agent) ─► runs/<method>/
                                       │
         ┌─────────────────────────────┼───────────────────────────────┐
   replay methods                 on-policy rollouts               Monte-Carlo
   dqn.py, offpolicy.py           rollout.py (GAE)                 reinforce.py
   (DQN family, DDPG, TD3,        (A2C, PPO; vector envs)          (whole episodes)
    SAC, SAC-discrete)
```

Notation: s observation (8 floats), a action, r reward, g = gamma the discount, done = 1 on a
real termination (crash / landing; not on the 1000-step time limit).

## Two families, one question

Every method answers "which actions should be made more likely?" in one of two ways:

* **Learn values, act greedily** (DQN family): learn Q(s, a), the expected return of doing a in
  s, by regressing it on a bootstrapped target. The policy is implicit: argmax Q.
* **Learn the policy directly** (REINFORCE, A2C, PPO): raise log pi(a|s) in proportion to how
  much better than average the action turned out (the advantage).
* **Actor-critic with replay** (DDPG, TD3, SAC, SAC-discrete) combines both: a critic Q learned
  like DQN, and an actor trained to pick actions the critic likes. This is what makes
  continuous actions possible: argmax over a continuous action is not computable, but a
  gradient through Q(s, mu(s)) is.

Replay methods reuse every transition many times (sample efficient, but the data are off-policy
so the target must be careful). On-policy methods use fresh data from the current policy
(simple, stable gradients, but every sample is used once or a few times, so they need more
environment steps).

## DQN, Double DQN, Dueling DQN, PER, n-step  ([dqn.py](../src/lunarlander_rl/algorithms/dqn.py))

```
y    = R + discount * (1 - done) * Q_target(s', a*)
a*   = argmax_a Q_target(s', a)                       DQN
a*   = argmax_a Q_online(s', a)                       Double DQN
loss = mean_i w_i * Huber(Q(s_i, a_i) - y_i)
```

* **Replay buffer** ([buffers.py](../src/lunarlander_rl/buffers.py)): consecutive steps are
  strongly correlated; sampling random minibatches from the last 50k transitions decorrelates
  them and reuses each one many times.
* **Target network**: if the target y used the network being trained, every gradient step would
  move the target too (chasing your own tail). A frozen copy, refreshed every 250 gradient
  steps, keeps y still while we regress towards it.
* **Huber loss + gradient clipping**: early TD errors are huge (returns of -100 or +100); Huber
  is quadratic for small errors and linear for big ones, so one crash does not blow up the step.
* **epsilon-greedy**: random action with probability epsilon, decayed linearly from 1 to 0.05
  over the first 12% of the budget. Q-learning only improves actions it tries.
* **Double DQN**: max over noisy Q estimates is biased upwards (the max picks the positive
  noise). Letting the online net choose a* and the target net score it decorrelates the
  choice from the evaluation. One line differs from DQN.
* **Dueling head** ([networks.py](../src/lunarlander_rl/networks.py) `DuelingQNet`):
  Q = V(s) + A(s, a) - mean_a A(s, a). In most LunarLander states the action barely matters
  (high in the sky); V learns that shared part from every sample, A only the differences.
* **Prioritized replay (PER)**: sample transition i with probability proportional to
  (|TD error| + eps)^alpha, so surprising transitions (the rare landings and crashes) are
  replayed more. Non-uniform sampling biases the gradient; importance weights
  w_i = (N P(i))^-beta / max w undo it, with beta annealed 0.4 -> 1. A sum tree makes sampling
  O(log N).
* **n-step returns**: R = r_t + g r_{t+1} + g^2 r_{t+2} with discount g^3: rewards propagate
  back n steps per update instead of one, at the cost of a bias (the n steps were taken by an
  older, more random policy). The buffer computes them, so the learner code is unchanged.

The preset `dueling-dqn` turns on all of: double, dueling, PER, 3-step. Each is also a flag:
`--set double=true`, `--set dueling=true`, `--set per=true`, `--set n_step=3`.

## REINFORCE with baseline  ([reinforce.py](../src/lunarlander_rl/algorithms/reinforce.py))

```
G_t  = r_t + g r_{t+1} + ... (to the end of the episode)
A_t  = G_t - V(s_t)          then normalised over the batch
loss = - mean log pi(a_t|s_t) A_t  + 0.5 mean (V(s_t) - G_t)^2  - c_ent H[pi]
```

The policy-gradient theorem says the gradient of expected return is E[grad log pi(a|s) * G].
Subtracting a baseline V(s) does not change that expectation but removes the part of G that
has nothing to do with the action (was the state good?), which cuts the variance a lot.
REINFORCE waits for whole episodes (up to 1000 steps) and its target G is a single sampled
return, so it is unbiased but noisy and slow. It is the reference point for everything below.

## A2C  ([a2c.py](../src/lunarlander_rl/algorithms/a2c.py), [rollout.py](../src/lunarlander_rl/algorithms/rollout.py))

Same loss as REINFORCE but the advantage is estimated with bootstrapping, from short rollouts
(16 steps x 8 parallel envs), with GAE:

```
delta_t = r_t + g V(s_{t+1}) (1 - end_t) - V(s_t)
A_t     = delta_t + g lambda (1 - end_t) A_{t+1}
```

lambda trades bias and variance: 0 is the one-step TD error, 1 is the Monte-Carlo return.
Bootstrapping means A2C does not wait for the end of the episode, so it updates every 128
transitions instead of every 4 episodes. Parallel envs make the batch less correlated (8
different episodes at once) in place of a replay buffer.

## PPO  ([ppo.py](../src/lunarlander_rl/algorithms/ppo.py))

Collect 16 envs x 1024 steps, compute GAE (lambda 0.98), then do 4 epochs of minibatches of 64:

```
ratio  = exp(log pi_new(a|s) - log pi_old(a|s))
L_clip = - mean min(ratio A, clip(ratio, 0.8, 1.2) A)
loss   = L_clip + 0.5 (V - R)^2 - 0.01 H[pi]
```

Doing many gradient steps on one batch is where the sample efficiency comes from, but the data
were collected by pi_old: after a few steps the policy has changed and the plain
policy-gradient estimate is no longer valid. The ratio corrects for that (importance
sampling), and the clip removes the incentive to move the ratio outside [0.8, 1.2], a cheap
trust region. Watch `clip_frac` and `approx_kl` in `metrics.json`: if they grow large the
learning rate or the number of epochs is too high. The same code runs the continuous env with a
diagonal Gaussian policy whose log std is a learned parameter (`ppo-continuous`).

## DDPG  ([ddpg.py](../src/lunarlander_rl/algorithms/ddpg.py))

```
critic: y = r + g (1 - done) Q'(s', mu'(s'))        loss = (Q(s, a) - y)^2
actor:  loss = - Q(s, mu(s))                         (gradient flows through the action)
targets: theta' <- 0.995 theta' + 0.005 theta
```

DQN for continuous actions: the argmax over a is replaced by an actor mu(s) trained to climb
the critic. The deterministic actor explores with Gaussian noise. DDPG is famously brittle:
the actor exploits any overestimated bump in Q, which inflates Q further.

## TD3  ([td3.py](../src/lunarlander_rl/algorithms/td3.py))

Three fixes to that, each one line in the code:

1. **Twin critics, take the minimum** in the target: a pessimistic estimate cancels the
   upward bias of maximising over a learned function.
2. **Target policy smoothing**: add clipped noise to the target action, so the target is an
   average over a small neighbourhood and the actor cannot exploit a sharp, wrong peak.
3. **Delayed actor**: update the actor and the targets every 2 critic steps, so the actor
   follows a critic that has had time to fit.

## SAC  ([sac.py](../src/lunarlander_rl/algorithms/sac.py))

Maximise reward plus entropy: the agent is paid alpha * H for staying random.

```
critic: y = r + g (1 - done) [min Q'_i(s', a') - alpha log pi(a'|s')],  a' ~ pi(.|s')
actor:  loss = alpha log pi(a~|s) - min Q_i(s, a~),   a~ = tanh(mu + sigma * eps)
alpha:  loss = - log alpha (log pi(a~|s) + target_entropy),   target_entropy = -2
```

The actor is a tanh-squashed Gaussian; the reparameterisation `a = tanh(mu + sigma eps)` lets
the gradient of Q flow into mu and sigma, and the log-prob gets the tanh change-of-variables
correction `- sum log(1 - tanh(u)^2)` (tested against torch's `TanhTransform`). Exploration
is part of the objective instead of hand-tuned noise, and alpha is learned so that the
policy's entropy stays near the target. No target actor is needed: the twin min and the entropy
term already regularise the target.

## Discrete SAC  ([sac_discrete.py](../src/lunarlander_rl/algorithms/sac_discrete.py))

With 4 actions the expectations over pi are sums, computed exactly (no sampling, no
reparameterisation):

```
V'(s') = sum_a' pi(a'|s') [min Q'_i(s', a') - alpha log pi(a'|s')]
actor:  loss = sum_a pi(a|s) [alpha log pi(a|s) - min Q_i(s, a)]
alpha:  loss = log alpha (H[pi(.|s)] - target_entropy),  target_entropy = 0.5 log 4
```

The critics have one output per action, like DQN. The target entropy is the delicate knob: the
paper's 0.98 log |A| keeps the policy almost uniform; a smaller fraction lets it commit.

## Networks

| used by | network | hidden | activation |
|---|---|---|---|
| DQN family, SAC-discrete | `QNet` / `DuelingQNet`, actor logits | 128 x 128 | ReLU |
| DDPG, TD3, SAC | `DeterministicActor` / `TanhGaussianActor`, `QCritic` | 128 x 128 | ReLU |
| REINFORCE, A2C, PPO | `CategoricalActor` / `GaussianActor` + `ValueNet` (separate) | 64 x 64 | tanh |

The on-policy actors start with a near-zero last layer (orthogonal init, gain 0.01) so the first
policy is close to uniform, the usual PPO trick.
