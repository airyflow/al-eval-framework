"""AL-loop metrics: Recall@k(t) and AUDC (area under the Recall@k(t) curve).

Neither exists in either source repo -- FusionAL's run_al.py only tracks
top-1% recall and best score per round, with no cumulative curve and no
integral. AUDC is the paper's headline AL metric (DESIGN.md Sec 4.5) and has
to be built fresh.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _read_scores_csv(path: Path) -> Dict[str, float]:
    scores = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                scores[row[0]] = float(row[1])
            except (ValueError, IndexError):
                continue
    return scores


def recall_at_k_curve(
    run_dir: Path, oracle: Dict[str, float], k_frac: float, minimize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Recall@k(t): fraction of the oracle's true top-k found by round t.

    Reads the sequence of `top_<n>_explored_iter_<i>.csv` snapshots an
    Explorer writes when `write_intermediate=True` -- each is the full set
    of molecules queried up to and including iteration i.
    """
    data_dir = Path(run_dir) / "data"
    snapshot_paths = sorted(
        data_dir.glob("top_*_explored_iter_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not snapshot_paths:
        raise FileNotFoundError(f"No intermediate score snapshots found in {data_dir}")

    k = max(1, int(k_frac * len(oracle)))
    true_topk = set(sorted(oracle, key=lambda s: oracle[s], reverse=not minimize)[:k])

    t_vals, recall_vals = [], []
    for path in snapshot_paths:
        explored = _read_scores_csv(path)
        n_hit = len(true_topk & explored.keys())
        t_vals.append(len(explored))
        recall_vals.append(n_hit / k)

    return np.array(t_vals), np.array(recall_vals)


def audc(t: np.ndarray, recall: np.ndarray, t_max_frac: float, pool_size: int) -> float:
    """Area under the Recall@k(t) curve, integrated over t in
    [0, t_max_frac * pool_size] and normalized by that window so AUDC in
    [0, 1] (1 = every true top-k hit was found immediately).
    """
    t_max = t_max_frac * pool_size
    t_ext = np.concatenate([[0.0], t, [t_max]])
    r_ext = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])

    mask = t_ext <= t_max
    t_ext, r_ext = t_ext[mask], r_ext[mask]
    if t_ext[-1] < t_max:
        t_ext = np.append(t_ext, t_max)
        r_ext = np.append(r_ext, r_ext[-1])

    return float(np.trapz(r_ext, t_ext) / t_max)
