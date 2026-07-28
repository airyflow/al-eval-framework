#!/usr/bin/env python
"""Drives the real molpal.explorer.Explorer for one (representation, seed)
pair against a DOCKSTRING target -- the Phase 0 AL loop (DESIGN.md Sec 2.1,
4.4). Wraps the vendored Explorer/Acquirer/LookupObjective unchanged; the
only new piece is McDropoutModel (src/surrogate/model_adapter.py), wired in
through molpal/models/__init__.py's "mc_dropout_mlp" branch.

Usage
-----
python -m src.al.run_al --representation morgan --seed 0
"""
from __future__ import annotations

import os

# Must be set before numpy/torch import. On this machine's 48-core node,
# unconstrained BLAS/OpenMP thread pools have caused a real, reproducible
# near-hang (49 threads, <1% CPU, stalled 35+ min on a plain morgan run) --
# constrain them rather than rely on every invocation setting the env var.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
from pathlib import Path

import yaml

from molpal.explorer import Explorer
from src.al.objective_dockstring import write_objective_config
from src.data.dockstring import cache_path
from src.representations.feature_source import build_feature_source

ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--experiment-config", default=str(ROOT / "configs" / "experiment.yaml")
    )
    # smoke-test overrides, for validating the plumbing without waiting on
    # the full Phase 0 grid (DESIGN.md milestone: "run one full AL loop
    # end-to-end (Morgan FP, 1 seed) ... verify checkpointing works")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    # overrides for sweeping axes beyond the fixed Phase 0/1 protocol
    # (e.g. the acquisition-function grid) without editing/duplicating
    # configs/experiment.yaml. --tag keeps these runs from overwriting the
    # UCB baseline results every earlier analysis in this repo depends on.
    parser.add_argument("--target", default=None)
    parser.add_argument("--acquisition", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    exp_cfg = load_yaml(Path(args.experiment_config))
    target = args.target or exp_cfg["target"]
    acquisition = args.acquisition or exp_cfg["acquisition"]
    target_cfg = load_yaml(ROOT / "configs" / "target" / f"{target}.yaml")
    rep_cfg = load_yaml(ROOT / "configs" / "representation" / f"{args.representation}.yaml")
    surr_cfg = dict(exp_cfg["surrogate"])
    if args.epochs is not None:
        surr_cfg["epochs"] = args.epochs

    objective_config = write_objective_config(target)
    pool_csv = cache_path(target)
    feature_source = build_feature_source(rep_cfg, target)

    run_dir = ROOT / "results" / target / args.representation
    if args.tag:
        run_dir = run_dir / args.tag
    run_dir = run_dir / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    explorer = Explorer(
        path=str(run_dir),
        libraries=[str(pool_csv)],
        title_line=True,
        delimiter=",",
        smiles_col=0,
        fingerprint="morgan",
        radius=2,
        length=2048,
        pool="eager",
        model="mc_dropout_mlp",
        feature_source=feature_source,
        hidden_dims=tuple(surr_cfg["hidden_dims"]),
        dropout=surr_cfg["dropout"],
        mc_samples=surr_cfg["mc_samples"],
        lr=surr_cfg["lr"],
        epochs=surr_cfg["epochs"],
        batch_size=surr_cfg["batch_size"],
        objective="lookup",
        objective_config=objective_config,
        minimize=target_cfg["minimize"],
        metric=acquisition,
        beta=exp_cfg["ucb_beta"],
        init_size=exp_cfg["init_frac"],
        batch_sizes=[exp_cfg["batch_frac"]],
        seed=args.seed,
        max_iters=args.max_iters or exp_cfg["n_rounds"],
        budget=1.0,
        k=exp_cfg["metrics"]["recall_k_frac"],
        write_intermediate=True,
        chkpt_freq=1,
        verbose=1,
    )
    explorer.run()
    return explorer


if __name__ == "__main__":
    main()
