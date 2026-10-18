"""Generate notebooks/train_colab_kaggle.ipynb from the package (run from the repo root).

    python notebooks/build_notebook.py

The notebook installs the package from GitHub, explains the environment and every algorithm
(printing each update rule straight from src/ with inspect.getsource, so it never drifts from
the code), then trains every method with bigger budgets than the local CPU runs and builds the
same comparison table and figures.
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# LunarLander with every RL method: DQN family, REINFORCE, A2C, PPO, SAC, DDPG, TD3

This notebook drives the [`lunarlander-rl`](https://github.com/benisalla/lunarlander-rl) package
on Kaggle or Colab. It explains the environment and each algorithm next to its code, then
trains every method with bigger budgets than the short CPU runs in the README, and compares
them with the same evaluation, table and curves.

Runtime: CPU is fine for PPO / A2C / REINFORCE; a GPU helps the replay methods a little (small
networks). The full comparison below takes about 2-3 hours on a Kaggle CPU session; cut
`BUDGET` to try it faster.
""")

md("""
## 1. Install

The repository is private: on Kaggle add a secret `GITHUB_TOKEN` (Add-ons -> Secrets), on Colab
put it in the Secrets panel; the cell below reads it. With a public fork no token is needed.
`gymnasium[box2d]` installs Box2D from a prebuilt wheel (no swig needed).
""")
code("""
import os, subprocess, sys

token = os.environ.get("GITHUB_TOKEN", "")
try:  # Kaggle secrets
    from kaggle_secrets import UserSecretsClient
    token = token or UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
try:  # Colab secrets
    from google.colab import userdata
    token = token or userdata.get("GITHUB_TOKEN")
except Exception:
    pass
auth = f"{token}@" if token else ""
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"git+https://{auth}github.com/benisalla/lunarlander-rl.git"], check=True)
""")
code("""
import inspect, json, os, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

import lunarlander_rl as ll
from lunarlander_rl import METHODS, evaluate, load_agent, make_config, train
from lunarlander_rl.algorithms import ALGORITHMS
from lunarlander_rl.envs import make_env, shaping_potential, heuristic_action
from lunarlander_rl.evaluate import BASELINES
from lunarlander_rl.play import record_episode, save_gif
from lunarlander_rl.viz import plot_curves, plot_training
from lunarlander_rl.compare import load_runs, results_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THREADS = os.cpu_count() or 1
print("device:", DEVICE, "| cpu threads:", THREADS, "| torch", torch.__version__)

def show_source(obj):
    display(Markdown("```python\\n" + inspect.getsource(obj) + "\\n```"))
""")

md("""
## 2. The environment

Observation: 8 floats `[x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact]`.
Discrete actions: `0` nothing, `1` left engine, `2` main engine, `3` right engine. Continuous
actions: `[main, lateral]` in [-1, 1]; the main engine fires (50-100% power) when main > 0, a side
engine when |lateral| > 0.5. An episode ends on a crash / landing (`terminated`) or after 1000
steps (`truncated`). **Solved = mean return >= 200.**

Two reference policies give the scale: random actions (about -170) and the PID controller that
ships with gymnasium (about +260).
""")
code("""
env = make_env(continuous=False)
print(env.observation_space, env.action_space, make_env(continuous=True).action_space)
for name, pol in BASELINES.items():
    r = evaluate(None, False, 10, policy=pol)
    print(f"{name:9s}: {r.mean():7.1f} +- {r.std():.1f}")
frames, ret = record_episode(None, False, seed=1, policy=BASELINES["heuristic"])
save_gif(frames, "heuristic.gif"); print("heuristic return", round(ret, 1))
display(Image("heuristic.gif"))
""")

md("""
## 3. Reward shaping: why LunarLander is learnable

The per-step reward is the change of a *potential* minus fuel:

    Phi(s)   = -100 |(x, y)| - 100 |(vx, vy)| - 100 |angle| + 10 leg_left + 10 leg_right
    reward_t = Phi(s_{t+1}) - Phi(s_t) - 0.30 main_power - 0.03 side_power

plus -100 for a crash and +100 for coming to rest. Over an episode the Phi terms telescope to
Phi(end) - Phi(start), so shaping does not change which policy is optimal (Ng et al., 1999); it
only gives a dense signal at every step: getting closer, slower and more upright is rewarded
immediately instead of only at touchdown. The fuel cost creates the classic local optimum of
*hovering* (no crash, but no landing either); many curves below plateau there for a while.

The cell checks the decomposition on every step of a heuristic episode.
""")
code("""
show_source(shaping_potential)
obs, _ = env.reset(seed=0); done, t, parts = False, 0, []
while not done:
    a = heuristic_action(env, obs)
    nxt, r, term, trunc, _ = env.step(a)
    dphi = shaping_potential(nxt) - shaping_potential(obs)
    fuel = 0.30 * (a == 2) + 0.03 * (a in (1, 3))
    if t > 0 and not term:
        assert abs(r - (dphi - fuel)) < 1e-3
    parts.append((r, dphi, -fuel)); obs, t, done = nxt, t + 1, term or trunc
parts = np.array(parts)
plt.figure(figsize=(9, 3)); plt.plot(np.cumsum(parts[:, 0]), label="cumulative reward")
plt.plot(np.cumsum(parts[:, 1]), label="cumulative shaping"); plt.plot(np.cumsum(parts[:, 2]), label="cumulative fuel")
plt.legend(); plt.xlabel("step"); plt.title("one heuristic landing"); plt.show()
""")

md("""
## 4. Shared machinery

Every method uses the same `Config`, the same `Tracker` (step counter, periodic greedy
evaluation, episode log) and the same `evaluate` (greedy policy, fixed unseen seeds). Replay
methods share the replay buffer, whose n-step logic means every learner uses the same target
form `R + discount * (1 - done) * bootstrap(s')`. Note `done` is only a real termination: a
1000-step time limit still bootstraps.
""")
code("""
from lunarlander_rl.buffers import ReplayBuffer, PrioritizedReplayBuffer
from lunarlander_rl.metrics import Tracker
show_source(evaluate)
show_source(ReplayBuffer._commit)
show_source(PrioritizedReplayBuffer.sample)
""")

ALGOS = [
    ("dqn", "DQN, Double DQN, Dueling DQN, PER, n-step", """
Learn Q(s, a) by regressing on `y = R + discount (1 - done) Q_target(s', a*)`. The **replay
buffer** decorrelates consecutive steps and reuses each transition many times; the **target
network** (copied every 250 updates) keeps y still while we regress towards it; the **Huber loss**
keeps one -100 crash from blowing up the step. **Double DQN** changes one line: the online net
picks a*, the target net scores it, which removes the upward bias of max over noisy estimates.
**Dueling** splits Q = V + A - mean(A), so the many states where the action barely matters
train V from every sample. **PER** replays surprising transitions more and corrects the bias with
importance weights. **n-step** propagates rewards n steps per update.
""", ["update", "learn"]),
    ("reinforce", "REINFORCE with a baseline", """
Play whole episodes, compute the reward-to-go G_t, and push up log pi(a_t|s_t) by
`A_t = G_t - V(s_t)`. The baseline V does not change the expected gradient, it removes the part of
G that is about the state rather than the action, which cuts the variance. No bootstrapping:
unbiased but noisy, and it waits up to 1000 steps per episode before learning anything.
""", ["update"]),
    ("a2c", "A2C", """
Same loss as REINFORCE, but advantages come from short rollouts (8 envs x 16 steps) with GAE,
bootstrapping from V instead of waiting for the episode end. One gradient step per rollout, then
the data are thrown away. The parallel envs replace the replay buffer as the source of
decorrelated data.
""", ["update"]),
    ("ppo", "PPO (discrete and continuous)", """
Collect 16 x 1024 steps, compute GAE, then take 4 epochs of minibatch steps on the **clipped**
surrogate `min(ratio A, clip(ratio, 0.8, 1.2) A)`. The ratio pi_new / pi_old corrects for the
policy having moved since the data were collected; the clip stops it moving too far, a cheap
trust region. This is what lets PPO reuse each sample ~4 times where A2C uses it once. With a
Gaussian policy and a learned log std, the same code runs the continuous env.
""", ["update"]),
    ("sac_discrete", "Discrete SAC", """
Soft actor-critic with a categorical policy: maximise reward + alpha * entropy. With 4 actions all
expectations over pi are exact sums, so there is no sampling or reparameterisation. Twin
critics (min) fight overestimation; alpha is tuned so the entropy stays near a target (a
fraction of log 4).
""", ["update"]),
    ("ddpg", "DDPG (continuous)", """
DQN for continuous actions: the argmax over a becomes an actor mu(s) trained to climb the
critic, `loss = -Q(s, mu(s))`. Slow Polyak-averaged targets for both nets, Gaussian exploration
noise. Brittle: the actor exploits any overestimated bump in Q.
""", ["update"]),
    ("td3", "TD3 (continuous)", """
DDPG plus three fixes: **twin critics with a min** in the target, **target policy smoothing**
(clipped noise on the target action), and a **delayed actor** (every 2 critic steps).
""", ["update"]),
    ("sac", "SAC (continuous)", """
Stochastic tanh-Gaussian actor trained through the reparameterisation a = tanh(mu + sigma eps),
twin critics, entropy bonus alpha * H, and alpha learned towards a target entropy of -dim(A).
Exploration is part of the objective instead of hand-tuned noise.
""", ["update"]),
]

md("""
## 5. Every algorithm, step by step

For each method: the idea in plain words, then its update rule printed from the package source.
""")
for algo, title, text, methods in ALGOS:
    md(f"### {title}\n\n{text}")
    lines = [f"cls = ALGORITHMS[{algo!r}]", "print(cls.__module__); print(inspect.getmodule(cls).__doc__)"]
    lines += [f"show_source(cls.{m})" for m in methods]
    code("\n".join(lines))
code("""
from lunarlander_rl.algorithms.rollout import gae, OnPolicyAgent
show_source(gae)
show_source(OnPolicyAgent.collect)
""")

md("""
## 6. Train every method

Bigger budgets than the README's CPU runs: 256 x 256 networks and one gradient step per env
step for the replay methods, more steps for everyone. `SEEDS` can hold several seeds for a fair
comparison (RL results vary a lot between seeds). Each run is saved in `runs/<method>_s<seed>/`.
""")
code("""
SEEDS = [0]
BIG = dict(hidden=(256, 256), train_freq=1, gradient_steps=1, batch_size=256)
BUDGET = {
    "dqn": dict(total_steps=300_000, hidden=(256, 256), gradient_steps=4),
    "double-dqn": dict(total_steps=300_000, hidden=(256, 256), gradient_steps=4),
    "dueling-dqn": dict(total_steps=300_000, hidden=(256, 256), gradient_steps=4),
    "reinforce": dict(total_steps=1_000_000),
    "a2c": dict(total_steps=1_000_000),
    "ppo": dict(total_steps=2_000_000),
    "sac-discrete": dict(total_steps=300_000, **BIG),
    "ddpg": dict(total_steps=300_000, learning_starts=10_000, **BIG),
    "td3": dict(total_steps=300_000, learning_starts=10_000, **BIG),
    "sac": dict(total_steps=300_000, learning_starts=10_000, **BIG),
    "ppo-continuous": dict(total_steps=2_000_000),
}
for k, v in BUDGET.items():
    print(f"{k:15s} {METHODS[k].summary}")
""")
code("""
summaries = {}
for method, over in BUDGET.items():
    for seed in SEEDS:
        cfg = make_config(method, seed=seed, device=DEVICE, threads=THREADS, **over)
        print(f"\\n=== {method} seed {seed}: {cfg.total_steps:,} steps ===")
        agent, s = train(cfg, f"runs/{method}_s{seed}")
        summaries[(method, seed)] = s
""")

md("""
## 7. Compare: table, curves, landings

Same columns as the README: steps to the first evaluation >= 200, final return over 20 fresh
episodes, best evaluation, and wall time.
""")
code("""
runs = load_runs("runs")
display(Markdown(results_table(runs)))
disc = {k: v for k, v in runs.items() if not v["summary"]["continuous"]}
cont = {k: v for k, v in runs.items() if v["summary"]["continuous"]}
plot_curves(disc, "curves_discrete.png", "LunarLander (discrete)")
plot_curves(cont, "curves_continuous.png", "LunarLander (continuous)")
plot_curves(disc, "curves_discrete_time.png", "LunarLander (discrete) vs wall time", x_time=True)
plot_curves(cont, "curves_continuous_time.png", "LunarLander (continuous) vs wall time", x_time=True)
for f in ["curves_discrete.png", "curves_continuous.png", "curves_discrete_time.png", "curves_continuous_time.png"]:
    display(Image(f))
""")
code("""
# training curve of one run (exploration included) and a GIF of its greedy landing
name = "ppo_s0"
plot_training(runs[name], f"{name}_train.png"); display(Image(f"{name}_train.png"))
agent, cfg = load_agent(f"runs/{name}")
frames, ret = record_episode(agent, cfg.continuous, seed=3)
save_gif(frames, f"{name}.gif"); print("return", round(ret, 1)); display(Image(f"{name}.gif"))
""")
code("""
# PPO diagnostics: the clip fraction and approximate KL tell whether the trust region is active
m = runs["ppo_s0"]["losses"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for a, k in zip(ax, ["clip_frac", "approx_kl", "entropy"]):
    s = np.array(m[k]); a.plot(s[:, 0], s[:, 1]); a.set_title(k); a.set_xlabel("env steps")
plt.show()
""")

md("""
## 8. Things to try

* Turn the DQN additions on one at a time: `make_config("dqn", per=True)`, `n_step=3`, `dueling=True`.
* PPO without clipping (`clip=1e9`) or with 10 epochs: watch `approx_kl` explode.
* SAC with a fixed temperature (`auto_alpha=False, alpha=0.05`) vs automatic.
* Treat truncation as terminal (edit `ReplayBuffer.add`) and see the hovering plateau get longer.
* Several seeds (`SEEDS = [0, 1, 2]`) and compare the spread, not only the mean.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
out = Path(__file__).with_name("train_colab_kaggle.ipynb")
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
