#!/usr/bin/env python
"""Computes static RMSE/ECE and diagnostic criteria C1-C4 for one
representation against a DOCKSTRING target (DESIGN.md Sec 4.5/4.6). ECE *is*
C2 (Predictive Calibration) -- static.py's held-out calibration error and the
paper's C2 are the same number.

C1 (smoothness) and C4 (knn_precision) are computed over the full pool +
oracle scores, independent of any AL run. C3 (latent_diversity,
tanimoto_diversity) characterizes the *discovered* set, so it requires
src.al.run_al to have already produced all_explored_final.csv for this
(representation, seed) -- if that file doesn't exist yet, C3 is skipped with
a note rather than computed on the wrong set.

Usage
-----
python -m src.metrics.run_static_and_criteria --representation morgan --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from molpal.analysis.criteria import (
    expected_calibration_error,
    knn_target_precision,
    latent_diversity,
    local_smoothness,
    tanimoto_diversity,
)
from src.data.dockstring import load_oracle
from src.metrics.static import static_eval
from src.representations.feature_source import build_feature_source

ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_explored_smiles(run_dir: Path) -> list:
    path = run_dir / "all_explored_final.csv"
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        return [row[0] for row in reader]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--n-pairs", type=int, default=20_000)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()

    exp_cfg = load_yaml(Path(args.experiment_config))
    target = exp_cfg["target"]
    rep_cfg = load_yaml(ROOT / "configs" / "representation" / f"{args.representation}.yaml")
    surr_cfg = exp_cfg["surrogate"]
    criteria_wanted = set(exp_cfg["metrics"].get("criteria", ["smoothness", "calibration"]))

    oracle = load_oracle(target)
    feature_source = build_feature_source(rep_cfg, target)

    rmse, ece = static_eval(
        feature_source,
        oracle,
        train_frac=exp_cfg["init_frac"],
        seed=args.seed,
        hidden_dims=tuple(surr_cfg["hidden_dims"]),
        dropout=surr_cfg["dropout"],
        mc_samples=surr_cfg["mc_samples"],
        lr=surr_cfg["lr"],
        epochs=surr_cfg["epochs"],
        batch_size=surr_cfg["batch_size"],
    )

    smis = list(oracle.keys())
    scores = np.array([oracle[s] for s in smis])
    embeddings = feature_source(smis)

    row = {
        "target": target,
        "representation": args.representation,
        "seed": args.seed,
        "static_rmse": rmse,
        "ece": ece,  # C2 Predictive Calibration
    }

    if "smoothness" in criteria_wanted:
        row["smoothness_rho"] = local_smoothness(embeddings, scores, n_pairs=args.n_pairs, seed=args.seed)

    if "knn_precision" in criteria_wanted:
        row["knn_precision"] = knn_target_precision(embeddings, scores, k=args.knn_k, seed=args.seed)

    run_dir = ROOT / "results" / target / args.representation / f"seed{args.seed}"
    explored_smis = _load_explored_smiles(run_dir)

    if "latent_diversity" in criteria_wanted:
        if explored_smis is not None:
            explored_embeddings = feature_source(explored_smis)
            row["latent_diversity"] = latent_diversity(explored_embeddings, n_pairs=args.n_pairs, seed=args.seed)
        else:
            print("[skip] latent_diversity requires src.al.run_al to have run first (no all_explored_final.csv)")

    if "tanimoto_diversity" in criteria_wanted:
        if explored_smis is not None:
            row["tanimoto_diversity"] = tanimoto_diversity(explored_smis)
        else:
            print("[skip] tanimoto_diversity requires src.al.run_al to have run first (no all_explored_final.csv)")

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "static_criteria.json"
    out_path.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
