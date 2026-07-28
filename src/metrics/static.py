"""Static (non-AL) representation evaluation: train the surrogate once on a
small random split, report held-out RMSE and calibration (ECE). This is the
"how good is this representation normally" number DESIGN.md Sec 4.5/4.6
pairs against AUDC in the decoupling analysis -- it does not exist in either
source repo and has to be built fresh.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from molpal.analysis.criteria import expected_calibration_error
from src.representations.feature_source import FeatureSource
from src.surrogate.mc_dropout_mlp import McDropoutMLP


def static_eval(
    feature_source: FeatureSource,
    oracle: Dict[str, float],
    train_frac: float = 0.005,
    seed: int = 0,
    **surrogate_kwargs,
) -> Tuple[float, float]:
    """Train the surrogate once on `train_frac` of the pool, report RMSE and
    ECE on the remaining held-out molecules.
    """
    smis = list(oracle.keys())
    rng = np.random.RandomState(seed)
    idxs = rng.permutation(len(smis))
    n_train = max(1, int(train_frac * len(smis)))
    train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]

    train_smis = [smis[i] for i in train_idxs]
    test_smis = [smis[i] for i in test_idxs]
    y_train = np.array([oracle[s] for s in train_smis])
    y_test = np.array([oracle[s] for s in test_smis])

    surrogate = McDropoutMLP(in_dim=feature_source.dim, **surrogate_kwargs)
    surrogate.fit(feature_source(train_smis), y_train)
    mu, sigma = surrogate.predict(feature_source(test_smis))

    rmse = float(np.sqrt(np.mean((mu - y_test) ** 2)))
    ece = expected_calibration_error(y_test, mu, sigma)
    return rmse, ece
