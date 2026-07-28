#!/usr/bin/env python
"""Decoupling analysis (Figure 1, the paper's central claim): pairs each
representation's static RMSE with its AUDC, ranks both axes, and reports the
Spearman rho between the rankings. A low/negative rho means static accuracy
does not reliably predict AL sample efficiency. Nothing in either source
repo computes this -- it's built fresh (DESIGN.md Sec 4.6).

Usage
-----
python -m src.analysis.decoupling
    (reads configs/experiment.yaml's target/representations/seeds, requires
    src.metrics.run_static_and_criteria and src.al.run_al to have already
    been run for each representation)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from src.data.dockstring import load_oracle
from src.metrics.al_metrics import audc, recall_at_k_curve

ROOT = Path(__file__).resolve().parent.parent.parent


def gather_row(
    target: str, representation: str, seeds: List[int], k_frac: float, t_max_frac: float,
    tag: str = "",
) -> Dict:
    """Average static RMSE and AUDC for one representation across seeds.

    `tag` selects an alternate acquisition-function run (e.g. "greedy",
    "ei") stored at results/{target}/{representation}/{tag}/seed{seed}/,
    for the representation-acquisition compatibility check -- static RMSE
    is acquisition-independent and always read from the untagged baseline
    path, only the AL run (AUDC) is read from the tagged path.
    """
    oracle = load_oracle(target)
    base_dir = ROOT / "results" / target / representation
    al_dir = base_dir / tag if tag else base_dir

    rmses, audcs = [], []
    for seed in seeds:
        static_path = base_dir / f"seed{seed}" / "static_criteria.json"
        if static_path.exists():
            rmses.append(json.loads(static_path.read_text())["static_rmse"])

        run_dir = al_dir / f"seed{seed}"
        if (run_dir / "data").exists():
            t, recall = recall_at_k_curve(run_dir, oracle, k_frac=k_frac, minimize=True)
            audcs.append(audc(t, recall, t_max_frac=t_max_frac, pool_size=len(oracle)))

    return {
        "representation": representation,
        "static_rmse": float(np.mean(rmses)) if rmses else None,
        "audc": float(np.mean(audcs)) if audcs else None,
        "n_seeds_rmse": len(rmses),
        "n_seeds_audc": len(audcs),
    }


def compute_decoupling(rows: List[Dict]) -> Dict:
    """Rank representations by static RMSE (ascending -- lower is better) and
    by AUDC (descending -- higher is better), then Spearman-correlate the two
    rankings.
    """
    reps = [r["representation"] for r in rows]
    rmses = np.array([r["static_rmse"] for r in rows], dtype=float)
    audcs = np.array([r["audc"] for r in rows], dtype=float)

    rmse_rank = rmses.argsort().argsort() + 1  # 1 = lowest RMSE = best
    audc_rank = (-audcs).argsort().argsort() + 1  # 1 = highest AUDC = best

    rho, pval = spearmanr(rmse_rank, audc_rank)

    return {
        "representations": reps,
        "static_rmse": rmses.tolist(),
        "audc": audcs.tolist(),
        "rmse_rank": rmse_rank.tolist(),
        "audc_rank": audc_rank.tolist(),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
    }


def plot_decoupling(result: Dict, out_path: Path) -> None:
    """Side-by-side rank panels connected by lines (DESIGN.md Sec 4.6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = result["representations"]
    rmse_rank = result["rmse_rank"]
    audc_rank = result["audc_rank"]
    n = len(reps)

    fig, ax = plt.subplots(figsize=(5, 0.8 * n + 2))
    for i, rep in enumerate(reps):
        ax.plot([0, 1], [rmse_rank[i], audc_rank[i]], marker="o")
        ax.annotate(rep, (0, rmse_rank[i]), xytext=(-10, 0), textcoords="offset points", ha="right", va="center")
        ax.annotate(rep, (1, audc_rank[i]), xytext=(10, 0), textcoords="offset points", ha="left", va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Static RMSE rank\n(1 = best)", "AUDC rank\n(1 = best)"])
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_yticks(range(1, n + 1))
    ax.set_title(f"Decoupling: Spearman rho = {result['spearman_rho']:.2f}")
    ax.set_xlim(-0.4, 1.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


def main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "decoupling.png"))
    args = parser.parse_args()

    with open(args.experiment_config) as f:
        exp_cfg = yaml.safe_load(f)

    target = exp_cfg["target"]
    representations = exp_cfg["representations"]
    seeds = exp_cfg["seeds"]
    k_frac = exp_cfg["metrics"]["recall_k_frac"]
    t_max_frac = exp_cfg["metrics"]["audc_t_frac"]

    rows = [gather_row(target, rep, seeds, k_frac, t_max_frac) for rep in representations]
    missing = [r["representation"] for r in rows if r["static_rmse"] is None or r["audc"] is None]
    if missing:
        raise RuntimeError(
            f"Missing static_criteria.json or AL run data for: {missing}. "
            f"Run src.metrics.run_static_and_criteria and src.al.run_al for these first."
        )

    result = compute_decoupling(rows)
    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    plot_decoupling(result, out_path)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
