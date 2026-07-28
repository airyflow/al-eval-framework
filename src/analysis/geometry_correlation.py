#!/usr/bin/env python
"""Correlates label-free representation geometry (src/analysis/geometry.py)
against AL sample efficiency (AUDC), across every (target, representation)
pair with both a geometry_stats.json and a completed AL run on disk.

This is the test of the geometry-aware extension's core question: does the
chemical-space geometry an encoder induces predict how well it does in
active learning, beyond (or instead of) static RMSE?

Statistical honesty note: with only ~6 representations, the multivariate
regression (ALC ~ geometry features) has at most ~6 distinct geometry
tuples repeated across targets/seeds -- not enough independent variation
to fit a multi-parameter model credibly. It is reported here as
EXPLORATORY/illustrative only; the defensible statistic is the univariate
Spearman correlation per metric, computed across the (target,
representation) grid (n ~ 20-24).

Usage
-----
python -m src.analysis.geometry_correlation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from scipy.stats import spearmanr

from src.analysis.decoupling import gather_row

ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS = ["PARP1", "F2", "ESR2", "EGFR"]

# Okabe-Ito colorblind-safe qualitative set, one fixed color per
# representation, held constant across every panel/figure.
REP_COLORS = {
    "morgan": "#E69F00",
    "molformer": "#56B4E9",
    "unimol": "#009E73",
    "grover": "#0072B2",
    "fusional_lite": "#D55E00",
    "fusional_full": "#CC79A7",
}

GEOMETRY_METRICS = [
    "intrinsic_dim_twonn",
    "participation_ratio",
    "pca_dim_90pct",
    "neighborhood_stability",
    "density_variance",
    "hubness_skewness",
    "smoothness_rho",  # rho_phi / C1, pulled in from static_criteria.json for the same table
]


def gather_geometry_audc_table(
    representations: List[str], seeds: List[int], k_frac: float, t_max_frac: float,
    targets: List[str] = TARGETS,
) -> List[Dict]:
    rows = []
    for target in targets:
        for rep in representations:
            geom_path = ROOT / "results" / target / rep / "geometry_stats.json"
            if not geom_path.exists():
                continue
            geom = json.loads(geom_path.read_text())

            al_row = gather_row(target, rep, seeds, k_frac, t_max_frac)
            if al_row["audc"] is None:
                print(f"[skip] {target}/{rep}: no completed AL run (AUDC missing)")
                continue

            row = {"target": target, "representation": rep, "audc": al_row["audc"]}
            for m in GEOMETRY_METRICS:
                val = geom.get(m)
                if m == "pca_dim_90pct" and geom.get("pca_dim_90pct_truncated"):
                    val = None  # truncated estimate isn't a real dim -- exclude, don't misreport
                row[m] = val
            rows.append(row)
    return rows


def compute_correlations(rows: List[Dict]) -> Dict:
    audc = np.array([r["audc"] for r in rows], dtype=float)
    result = {"n_pairs": len(rows), "metrics": {}}

    for m in GEOMETRY_METRICS:
        vals = np.array([r[m] for r in rows], dtype=float)
        mask = ~np.isnan(vals)
        n = int(mask.sum())
        if n < 4:
            result["metrics"][m] = {"n": n, "spearman_rho": None, "spearman_pvalue": None, "note": "too few pairs"}
            continue
        rho, pval = spearmanr(vals[mask], audc[mask])
        result["metrics"][m] = {"n": n, "spearman_rho": float(rho), "spearman_pvalue": float(pval)}

    return result


def exploratory_regression(rows: List[Dict]) -> Dict:
    """Multivariate OLS of standardized geometry features -> AUDC, over
    rows with every feature present. EXPLORATORY ONLY -- see module
    docstring; do not present this as a confirmed predictive result."""
    feature_names = [m for m in GEOMETRY_METRICS if m != "pca_dim_90pct"]  # drop the metric most prone to None/truncation
    complete_rows = [r for r in rows if all(r[m] is not None for m in feature_names)]

    n_reps = len(set(r["representation"] for r in complete_rows))
    if len(complete_rows) < len(feature_names) + 2:
        return {
            "n": len(complete_rows), "n_distinct_representations": n_reps,
            "note": "insufficient complete rows to fit; not attempted",
        }

    X = np.array([[r[m] for m in feature_names] for r in complete_rows], dtype=float)
    y = np.array([r["audc"] for r in complete_rows], dtype=float)

    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    X_design = np.column_stack([np.ones(len(X_std)), X_std])

    coefs, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ coefs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n": len(complete_rows),
        "n_distinct_representations": n_reps,
        "features": feature_names,
        "standardized_coefficients": dict(zip(["intercept"] + feature_names, coefs.tolist())),
        "r_squared": float(r_squared),
        "caveat": (
            f"EXPLORATORY ONLY: fit over {len(complete_rows)} (target, representation) rows "
            f"spanning only {n_reps} distinct representations for {len(feature_names)} features. "
            f"Geometry is representation-level, not per-row-independent, so this is functionally "
            f"closer to fitting representation identity than fitting geometry; treat r_squared as "
            f"illustrative, not confirmatory. The defensible statistic is the univariate Spearman "
            f"correlation per metric (see compute_correlations)."
        ),
    }


def plot_geometry_scatter(rows: List[Dict], correlations: Dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [m for m in GEOMETRY_METRICS if correlations["metrics"].get(m, {}).get("n", 0) >= 4]
    n = len(metrics_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // ncols][idx % ncols]
        for rep, color in REP_COLORS.items():
            xs = [r[metric] for r in rows if r["representation"] == rep and r[metric] is not None]
            ys = [r["audc"] for r in rows if r["representation"] == rep and r[metric] is not None]
            if xs:
                ax.scatter(xs, ys, color=color, label=rep, s=36, edgecolors="white", linewidths=0.5, zorder=3)

        stat = correlations["metrics"][metric]
        ax.set_title(f"{metric}\nρ={stat['spearman_rho']:.2f} (n={stat['n']})", fontsize=9)
        ax.set_xlabel(metric, fontsize=8)
        ax.set_ylabel("AUDC", fontsize=8)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(REP_COLORS), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Representation geometry vs. AL efficiency (AUDC)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--targets", default=",".join(TARGETS), help="Comma-separated target list (default: all with data on disk)")
    arg_parser.add_argument("--out-prefix", default="geometry_correlation", help="Output basename under results/ (json/png)")
    args = arg_parser.parse_args()
    targets = args.targets.split(",")

    parser_cfg = ROOT / "configs" / "experiment.yaml"
    with open(parser_cfg) as f:
        exp_cfg = yaml.safe_load(f)

    representations = exp_cfg["representations"]
    seeds = exp_cfg["seeds"]
    k_frac = exp_cfg["metrics"]["recall_k_frac"]
    t_max_frac = exp_cfg["metrics"]["audc_t_frac"]

    rows = gather_geometry_audc_table(representations, seeds, k_frac, t_max_frac, targets=targets)
    print(f"\n[gathered] {len(rows)} (target, representation) pairs with both geometry stats and AUDC, targets={targets}\n")

    correlations = compute_correlations(rows)
    regression = exploratory_regression(rows)

    print("=== Univariate Spearman correlations (geometry metric vs. AUDC) ===")
    for m, stat in correlations["metrics"].items():
        if stat["spearman_rho"] is not None:
            print(f"  {m:<24} rho={stat['spearman_rho']:+.3f}  p={stat['spearman_pvalue']:.3f}  (n={stat['n']})")
        else:
            print(f"  {m:<24} skipped ({stat.get('note', 'n/a')})")

    print("\n=== Exploratory multivariate regression (see caveat) ===")
    print(json.dumps(regression, indent=2))

    result = {"rows": rows, "correlations": correlations, "exploratory_regression": regression}

    out_json = ROOT / "results" / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {out_json}")

    plot_geometry_scatter(rows, correlations, ROOT / "results" / f"{args.out_prefix}.png")


if __name__ == "__main__":
    main()
