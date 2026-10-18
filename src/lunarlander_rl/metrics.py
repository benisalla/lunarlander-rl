"""Run bookkeeping shared by every algorithm: the step counter, the training-episode returns,
the periodic greedy evaluations and the summary numbers that go into the comparison table.

An algorithm's training loop only has to call three things:

    tracker.add_steps(n)          after stepping the env(s) n times (runs an evaluation when due)
    tracker.add_episode(ret, len) whenever a training episode ends
    while not tracker.done: ...   to know when to stop
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .envs import SOLVED_SCORE


@dataclass
class Tracker:
    cfg: Config
    agent: object = None
    verbose: bool = True
    eval_fn: object = None                         # optional eval_fn(agent) -> returns; default: LunarLander
    steps: int = 0
    episodes: list = field(default_factory=list)   # (step, return, length)
    evals: list = field(default_factory=list)      # (step, mean, std, wall seconds)
    losses: dict = field(default_factory=dict)     # name -> list of (step, value)
    solved_at: int | None = None                   # first eval step with mean >= 200
    solved_time: float | None = None

    def __post_init__(self):
        self.start = time.time()
        self._next_eval = self.cfg.eval_every
        self._evaluated_at_end = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def done(self) -> bool:
        if self.steps >= self.cfg.total_steps:
            return True
        if self.cfg.max_minutes and self.elapsed > 60 * self.cfg.max_minutes:
            return True
        return self.cfg.stop_when_solved and self.solved_at is not None

    def progress(self) -> float:
        """Fraction of the budget used, for linear schedules (epsilon, PER beta)."""
        return min(1.0, self.steps / max(1, self.cfg.total_steps))

    def add_episode(self, ret: float, length: int) -> None:
        self.episodes.append((self.steps, float(ret), int(length)))

    def log(self, name: str, value: float) -> None:
        series = self.losses.setdefault(name, [])
        if not series or self.steps - series[-1][0] >= 1000:  # thin out: one point per 1000 steps
            series.append((self.steps, float(value)))

    def add_steps(self, n: int = 1) -> None:
        self.steps += n
        if self._evaluated_at_end:  # budget used up: the loop is only finishing its rollout
            return
        at_end = self.steps >= self.cfg.total_steps and not self._evaluated_at_end
        if self.steps >= self._next_eval or at_end:
            while self._next_eval <= self.steps:
                self._next_eval += self.cfg.eval_every
            self._evaluated_at_end = self.steps >= self.cfg.total_steps
            self.evaluate_now()

    def evaluate_now(self) -> float:
        from .evaluate import evaluate

        if self.eval_fn is not None:
            rets = self.eval_fn(self.agent)
        else:
            rets = evaluate(self.agent, self.cfg.continuous, self.cfg.eval_episodes, seed=10_000 + self.cfg.seed)
        mean, std = float(np.mean(rets)), float(np.std(rets))
        self.evals.append((self.steps, mean, std, self.elapsed))
        if mean >= SOLVED_SCORE and self.solved_at is None:
            self.solved_at, self.solved_time = self.steps, self.elapsed
        if self.verbose:
            recent = [r for _, r, _ in self.episodes[-20:]]
            train = f"{np.mean(recent):8.1f}" if recent else "     n/a"
            print(f"[{self.cfg.method}] step {self.steps:>8,d}  train(last 20) {train}  "
                  f"eval {mean:8.1f} +- {std:6.1f}  {self.elapsed:6.0f}s", flush=True)
        return mean

    def summary(self, final_returns: np.ndarray | None = None) -> dict:
        best = max((m for _, m, _, _ in self.evals), default=float("nan"))
        out = dict(
            method=self.cfg.method, algo=self.cfg.algo, continuous=self.cfg.continuous,
            total_steps=self.cfg.total_steps, steps=self.steps, seed=self.cfg.seed,
            solved_at=self.solved_at, solved_time=self.solved_time, best_eval=best,
            wall_time=self.elapsed, episodes=len(self.episodes),
        )
        if final_returns is not None:
            out.update(final_mean=float(np.mean(final_returns)), final_std=float(np.std(final_returns)),
                       final_n=len(final_returns), final_returns=[float(r) for r in final_returns])
        return out

    def save(self, path, final_returns=None, wall_time: float | None = None) -> dict:
        summary = self.summary(final_returns)
        if wall_time is not None:
            summary["wall_time"] = wall_time
        data = dict(summary=summary, config=self.cfg.to_dict(),
                    episodes=self.episodes, evals=self.evals, losses=self.losses)
        with open(path, "w") as f:
            json.dump(data, f)
        return data["summary"]
