"""Compare saved runs: one markdown table and the evaluation-curve figures.

    lunarlander-compare runs/ --out assets/          # every runs/*/metrics.json

Columns: steps to solve (first periodic greedy evaluation with mean >= 200), the final score
(mean +- std over ``final_eval_episodes`` fresh episodes), the best periodic evaluation, the
budget used and the wall time of training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import METHODS
from .viz import plot_curves

ORDER = list(METHODS)


def load_runs(root) -> dict[str, dict]:
    runs = {}
    for f in sorted(Path(root).glob("*/metrics.json")):
        runs[f.parent.name] = json.loads(f.read_text())
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[1]["summary"]["method"])
                       if kv[1]["summary"]["method"] in ORDER else 99))


def results_table(runs: dict[str, dict]) -> str:
    rows = ["| method | env | steps to solve | final return (20 ep) | best eval | steps run | wall time |",
            "|---|---|---|---|---|---|---|"]
    for name, m in runs.items():
        s = m["summary"]
        env = "continuous" if s["continuous"] else "discrete"
        solve = f"**{s['solved_at']:,}**" if s.get("solved_at") else "not reached"
        final = f"{s['final_mean']:.1f} +- {s['final_std']:.1f}"
        if s["final_mean"] >= 200:
            final = f"**{final}**"
        rows.append(f"| `{name}` | {env} | {solve} | {final} | {s['best_eval']:.1f} | {s['steps']:,} | "
                    f"{s['wall_time'] / 60:.1f} min |")
    return "\n".join(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Table + curves for saved LunarLander runs.")
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--out", default="assets", help="where to write results.md and the figures")
    a = p.parse_args(argv)

    runs = load_runs(a.runs_dir)
    if not runs:
        raise SystemExit(f"no */metrics.json under {a.runs_dir}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = results_table(runs)
    (out / "results.md").write_text(table + "\n")
    print(table)
    disc = {k: v for k, v in runs.items() if not v["summary"]["continuous"]}
    cont = {k: v for k, v in runs.items() if v["summary"]["continuous"]}
    if disc:
        plot_curves(disc, out / "curves_discrete.png", "LunarLander-v3 (discrete): greedy evaluation return")
        plot_curves(disc, out / "curves_discrete_time.png", "LunarLander-v3 (discrete) vs wall time", x_time=True)
    if cont:
        plot_curves(cont, out / "curves_continuous.png", "LunarLander-v3 continuous: greedy evaluation return")
        plot_curves(cont, out / "curves_continuous_time.png", "LunarLander-v3 continuous vs wall time", x_time=True)
    print(f"wrote {out}/results.md and the curve figures")
