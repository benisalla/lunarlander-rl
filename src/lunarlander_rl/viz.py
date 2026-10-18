"""Plots: evaluation curves of several runs on one figure, and the training curve of one run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .envs import SOLVED_SCORE  # noqa: E402


def smooth(x, k: int = 20) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def plot_curves(runs: dict[str, dict], path, title: str = "LunarLander: greedy evaluation return",
                x_time: bool = False) -> Path:
    """``runs`` maps a label to a metrics dict (as saved in metrics.json)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, m in runs.items():
        ev = np.array(m["evals"])
        if len(ev) == 0:
            continue
        x = ev[:, 3] / 60 if x_time else ev[:, 0] / 1000
        ax.plot(x, ev[:, 1], marker="o", ms=2.5, lw=1.6, label=label)
        ax.fill_between(x, ev[:, 1] - ev[:, 2], ev[:, 1] + ev[:, 2], alpha=0.12)
    ax.axhline(SOLVED_SCORE, color="k", ls="--", lw=1, label="solved (200)")
    ax.set_xlabel("wall time (minutes)" if x_time else "environment steps (thousands)")
    ax.set_ylabel("mean return over eval episodes")
    ax.set_ylim(bottom=max(-600, ax.get_ylim()[0]))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_training(m: dict, path, title: str | None = None) -> Path:
    """Training-episode returns (raw + smoothed) and periodic eval of one run."""
    ep = np.array(m["episodes"])
    ev = np.array(m["evals"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(ep):
        ax.plot(ep[:, 0] / 1000, ep[:, 1], ".", ms=2, alpha=0.3, label="training episode")
        ax.plot(ep[:, 0] / 1000, smooth(ep[:, 1]), lw=1.5, label="training (20-episode mean)")
    if len(ev):
        ax.plot(ev[:, 0] / 1000, ev[:, 1], "o-", ms=3, label="greedy eval")
    ax.axhline(SOLVED_SCORE, color="k", ls="--", lw=1)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("return")
    ax.set_ylim(bottom=max(-600, ax.get_ylim()[0]))
    ax.set_title(title or m["summary"]["method"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
