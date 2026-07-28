#!/usr/bin/env python
"""Computes label-free geometry diagnostics for one representation against
one DOCKSTRING target, and pulls in the already-computed distance-property
correlation (smoothness_rho from static_criteria.json -- algebraically the
same statistic the geometry proposal calls rho_phi, see geometry.py's
module docstring).

Requires src.metrics.run_static_and_criteria to have already run for this
(representation, target) (for smoothness_rho); everything else here only
needs the embeddings, so it can run standalone.

Usage
-----
python -m src.analysis.run_geometry --target PARP1 --representation morgan --seed 0
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import json
from pathlib import Path

import yaml

from src.analysis.geometry import compute_geometry_row
from src.data.dockstring import load_oracle
from src.representations.feature_source import MorganFeatureSource, build_feature_source

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=20_000, help="Subsample cap for PCA/TwoNN/density (cost control at scale)")
    parser.add_argument("--max-anchors", type=int, default=5_000, help="Subsample cap for neighborhood_stability anchors")
    args = parser.parse_args()

    rep_cfg_path = ROOT / "configs" / "representation" / f"{args.representation}.yaml"
    with open(rep_cfg_path) as f:
        rep_cfg = yaml.safe_load(f)

    oracle = load_oracle(args.target)
    smis = list(oracle.keys())

    feature_source = build_feature_source(rep_cfg, args.target)
    embeddings = feature_source(smis)

    # Fixed reference space for neighborhood_stability: Morgan fingerprints,
    # a chemistry-native similarity notion independent of any learned
    # encoder. Skip entirely when the representation being evaluated *is*
    # Morgan -- comparing it against itself is a tautology (NS_k == 1).
    if args.representation == "morgan":
        ns_k = None
    else:
        morgan = MorganFeatureSource()
        ref_embeddings = morgan(smis)

    row = {"target": args.target, "representation": args.representation, "seed": args.seed}

    if args.representation == "morgan":
        from src.analysis.geometry import density_stats, intrinsic_dim_twonn, pca_dimensionality

        row.update({
            "intrinsic_dim_twonn": intrinsic_dim_twonn(embeddings, max_samples=args.max_samples, seed=args.seed),
            **pca_dimensionality(embeddings, max_samples=args.max_samples, seed=args.seed),
            "neighborhood_stability": None,
            **density_stats(embeddings, k=args.k, max_samples=args.max_samples, seed=args.seed),
        })
    else:
        row.update(compute_geometry_row(
            embeddings, ref_embeddings, k=args.k,
            max_samples=args.max_samples, max_anchors=args.max_anchors, seed=args.seed,
        ))

    static_path = ROOT / "results" / args.target / args.representation / f"seed{args.seed}" / "static_criteria.json"
    if static_path.exists():
        row["smoothness_rho"] = json.loads(static_path.read_text()).get("smoothness_rho")
    else:
        print(f"[note] {static_path} not found -- smoothness_rho (rho_phi) omitted; "
              f"run src.metrics.run_static_and_criteria first to include it")

    out_dir = ROOT / "results" / args.target / args.representation
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "geometry_stats.json"
    out_path.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
