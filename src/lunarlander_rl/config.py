"""One flat config for every method, plus the per-method presets.

Every algorithm reads the fields it needs from the same ``Config`` dataclass, so a run is fully
described by one JSON file (saved next to the checkpoint). The presets in ``METHODS`` hold the
hyperparameters that make each method learn LunarLander on a CPU; anything can be overridden
from the CLI with ``--set key=value``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- run -------------------------------------------------------------------------------
    method: str = "dqn"            # preset name (see METHODS)
    algo: str = "dqn"              # the algorithm file that implements it
    continuous: bool = False       # LunarLander (4 discrete actions) or LunarLanderContinuous (2 floats)
    total_steps: int = 150_000     # environment steps (summed over all parallel envs)
    seed: int = 0
    threads: int = 1               # torch threads; 1 keeps runs reproducible and polite on a shared CPU
    device: str = "cpu"
    eval_every: int = 10_000       # run a greedy evaluation every this many env steps
    eval_episodes: int = 10        # episodes per periodic evaluation
    final_eval_episodes: int = 20  # episodes for the final score reported in the tables
    stop_when_solved: bool = False # stop early once a periodic evaluation reaches SOLVED_SCORE
    max_minutes: float = 0.0       # wall-clock cap for the training loop (0 = none)
    # ---- shared learning knobs ---------------------------------------------------------------
    hidden: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    max_grad_norm: float = 10.0
    # ---- replay-based methods (DQN family, DDPG, TD3, SAC) -----------------------------------
    buffer_size: int = 100_000
    learning_starts: int = 10_000  # random actions before the first update, fills the buffer
    train_freq: int = 1            # env steps between update rounds
    gradient_steps: int = 1        # gradient steps per update round
    tau: float = 0.005             # Polyak factor for target networks (soft update)
    # ---- DQN family --------------------------------------------------------------------------
    double: bool = False           # Double DQN target: online net picks a', target net scores it
    dueling: bool = False          # Q = V + A - mean(A)
    per: bool = False              # prioritized experience replay
    n_step: int = 1                # n-step returns in the replay buffer
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.12     # fraction of total_steps over which epsilon decays linearly
    target_update: int = 250       # hard target-network copy every this many gradient steps
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    # ---- policy-gradient methods (REINFORCE, A2C, PPO) ---------------------------------------
    n_envs: int = 16               # parallel environments
    n_steps: int = 1024            # rollout length per environment
    gae_lambda: float = 0.98
    n_epochs: int = 4              # PPO passes over each rollout
    clip: float = 0.2              # PPO ratio clipping
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    episodes_per_update: int = 4   # REINFORCE: whole episodes per gradient step
    # ---- actor-critic replay methods (DDPG, TD3, SAC) ----------------------------------------
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    explore_noise: float = 0.1     # DDPG/TD3 Gaussian exploration noise
    policy_delay: int = 2          # TD3: actor + targets updated every this many critic steps
    target_noise: float = 0.2      # TD3 target policy smoothing
    noise_clip: float = 0.5
    alpha: float = 0.2             # SAC entropy temperature (initial value when auto_alpha)
    auto_alpha: bool = True
    target_entropy_scale: float = 0.2  # discrete SAC: target entropy = scale * log(n_actions)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if "hidden" in d:
            d["hidden"] = tuple(d["hidden"])
        return cls(**d)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class Preset:
    algo: str
    continuous: bool
    summary: str
    overrides: dict = field(default_factory=dict)


# The hyperparameters follow the well-known working settings for LunarLander (rl-baselines3-zoo,
# CleanRL), shrunk so that one run fits in about 5-15 minutes on ONE CPU thread: 128x128 nets for
# the replay methods and one gradient step every 2 env steps (train_freq=2) for DDPG/TD3/SAC.
# The notebook raises these (256x256 nets, train_freq=1, bigger budgets) on a Kaggle/Colab machine.
_DQN = dict(
    hidden=(128, 128), lr=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=1_000, train_freq=4,
    gradient_steps=2, target_update=250, eps_end=0.05, eps_fraction=0.12, total_steps=150_000,
)
_PG = dict(hidden=(64, 64), lr=3e-4, max_grad_norm=0.5, gamma=0.999)
_CONT = dict(
    hidden=(128, 128), buffer_size=200_000, learning_starts=5_000, batch_size=128, train_freq=2,
    actor_lr=1e-3, critic_lr=1e-3, total_steps=120_000,
)

METHODS: dict[str, Preset] = {
    "dqn": Preset("dqn", False, "Q-learning with a replay buffer and a target network", _DQN),
    "double-dqn": Preset("dqn", False, "DQN whose target uses argmax of the online net", {**_DQN, "double": True}),
    "dueling-dqn": Preset(
        "dqn", False, "Double DQN with a V + A head, prioritized replay and 3-step returns",
        {**_DQN, "double": True, "dueling": True, "per": True, "n_step": 3},
    ),
    "reinforce": Preset(
        "reinforce", False, "Monte-Carlo policy gradient with a learned value baseline",
        {**_PG, "lr": 1e-3, "episodes_per_update": 4, "total_steps": 1_000_000},
    ),
    "a2c": Preset(
        "a2c", False, "synchronous advantage actor-critic, short rollouts, one step per rollout",
        {**_PG, "lr": 7e-4, "n_envs": 8, "n_steps": 16, "gae_lambda": 0.95, "gamma": 0.99,
         "ent_coef": 0.01, "total_steps": 1_000_000},
    ),
    "ppo": Preset(
        "ppo", False, "clipped surrogate objective, GAE, several epochs per rollout",
        {**_PG, "n_envs": 16, "n_steps": 1024, "batch_size": 64, "n_epochs": 4, "total_steps": 1_000_000},
    ),
    "sac-discrete": Preset(
        "sac_discrete", False, "soft actor-critic with a categorical policy (exact expectation over actions)",
        {"hidden": (128, 128), "lr": 3e-4, "batch_size": 128, "buffer_size": 100_000, "learning_starts": 5_000,
         "train_freq": 2, "tau": 0.005, "target_entropy_scale": 0.2,
         "total_steps": 100_000},
    ),
    "ddpg": Preset("ddpg", True, "deterministic actor + Q critic, Gaussian exploration noise", _CONT),
    "td3": Preset("td3", True, "DDPG + twin critics, delayed actor, target policy smoothing", _CONT),
    "sac": Preset(
        "sac", True, "stochastic tanh-Gaussian actor, twin critics, automatic entropy temperature",
        {**_CONT, "actor_lr": 7.3e-4, "critic_lr": 7.3e-4, "tau": 0.01},
    ),
    "ppo-continuous": Preset(
        "ppo", True, "PPO with a diagonal Gaussian policy on the 2 continuous engines",
        {**_PG, "n_envs": 16, "n_steps": 1024, "batch_size": 64, "n_epochs": 4, "total_steps": 1_000_000},
    ),
}


def make_config(method: str, **overrides) -> Config:
    """Config for a preset, with optional field overrides (``make_config("ppo", total_steps=50_000)``)."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    p = METHODS[method]
    cfg = Config(method=method, algo=p.algo, continuous=p.continuous)
    for k, v in {**p.overrides, **overrides}.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, tuple(v) if k == "hidden" else v)
    return cfg


def parse_override(text: str) -> tuple[str, object]:
    """Parse ``key=value`` from the CLI, casting value to the type of the default field."""
    key, _, raw = text.partition("=")
    key = key.strip().replace("-", "_")
    default = getattr(Config(), key)
    if isinstance(default, bool):
        val: object = raw.lower() in ("1", "true", "yes", "y")
    elif isinstance(default, int):
        val = int(float(raw))
    elif isinstance(default, float):
        val = float(raw)
    elif isinstance(default, tuple):
        val = tuple(int(x) for x in raw.replace("x", ",").split(",") if x)
    else:
        val = raw
    return key, val
