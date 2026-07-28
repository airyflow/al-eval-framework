#!/usr/bin/env python
"""Figure 2: Recall@k(t) curves for all representations on one target,
showing the sequential discovery process AUDC compresses into a single
number. Complements decoupling.py's rank-based Figure 1.

Usage
-----
python -m src.analysis.recall_curves
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.dockstring import load_oracle
from src.metrics.al_metrics import recall_at_k_curve

ROOT = Path(__file__).resolve().parent.parent.parent


def gather_curve(target: str, representation: str, seeds: List[int], k_frac: float) -> Optional[Dict]:
    """Average Recall@k(t) across seeds. Some (representation, seed) runs
    stop a round early if MolPAL's own moving-average convergence
    criterion triggers (Explorer.completed's `delta` check) before
    max_iters is reached -- seeds for the same representation can then
    have different numbers of rounds. Truncate to the shortest common
    length before averaging rather than assuming equal-length curves.
    """
    oracle = load_oracle(target)
    curves = []
    for seed in seeds:
        run_dir = ROOT / "results" / target / representation / f"seed{seed}"
        if not (run_dir / "data").exists():
            continue
        t, recall = recall_at_k_curve(run_dir, oracle, k_frac=k_frac, minimize=True)
        curves.append((t, recall))

    if not curves:
        return None

    min_len = min(len(t) for t, _ in curves)
    if any(len(t) != min_len for t, _ in curves):
        print(f"[note] {representation}: unequal round counts across seeds, truncating to {min_len}")

    t_ref = curves[0][0][:min_len]
    recalls = np.stack([recall[:min_len] for _, recall in curves])

    return {
        "t": t_ref,
        "recall_mean": recalls.mean(axis=0),
        "recall_std": recalls.std(axis=0),
        "n_seeds": len(curves),
    }


def plot_recall_curves(
    target: str, representations: List[str], seeds: List[int], k_frac: float,
    pool_size: int, out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for rep in representations:
        data = gather_curve(target, rep, seeds, k_frac)
        if data is None:
            print(f"[skip] no AL run data for {rep}")
            continue
        t_frac = data["t"] / pool_size * 100
        line, = ax.plot(t_frac, data["recall_mean"], marker="o", markersize=3, label=rep)
        ax.fill_between(
            t_frac,
            data["recall_mean"] - data["recall_std"],
            data["recall_mean"] + data["recall_std"],
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_xlabel("% of pool queried")
    ax.set_ylabel(f"Recall@{k_frac:.0%}(t)")
    ax.set_title(f"Recall@{k_frac:.0%}(t), {target} ({len(seeds)}-seed mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


def main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "recall_curves.png"))
    args = parser.parse_args()

    with open(args.experiment_config) as f:
        exp_cfg = yaml.safe_load(f)

    target = exp_cfg["target"]
    representations = exp_cfg["representations"]
    seeds = exp_cfg["seeds"]
    k_frac = exp_cfg["metrics"]["recall_k_frac"]

    oracle = load_oracle(target)
    pool_size = len(oracle)

    plot_recall_curves(target, representations, seeds, k_frac, pool_size, Path(args.out))


if __name__ == "__main__":
    main()
