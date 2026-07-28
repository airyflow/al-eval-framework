"""molpal Model ABC bridge for McDropoutMLP, backed by a FeatureSource
(Morgan on-the-fly or precomputed embeddings). Mirrors the shape of
FusionAL's EmbeddingMVEModel (molpal/models/mvemodels.py) but is
representation-agnostic via FeatureSource rather than hardcoding an
emb_dict lookup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from molpal.models.base import Model
from src.representations.feature_source import FeatureSource
from src.surrogate.mc_dropout_mlp import DEVICE, McDropoutMLP


class McDropoutModel(Model):
    @property
    def provides(self) -> Set[str]:
        return {"means", "vars"}

    @property
    def type_(self) -> str:
        return "mc_dropout_mlp"

    def __init__(
        self,
        feature_source: FeatureSource,
        hidden_dims=(512, 128),
        dropout: float = 0.2,
        mc_samples: int = 50,
        lr: float = 3e-4,
        epochs: int = 50,
        batch_size: int = 256,
        test_batch_size: int = 4096,
        **kwargs,
    ):
        super().__init__(test_batch_size=test_batch_size, **kwargs)
        self.feature_source = feature_source
        self.surrogate = McDropoutMLP(
            in_dim=feature_source.dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            mc_samples=mc_samples,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )

    def train(
        self,
        xs: Iterable[str],
        ys: np.ndarray,
        *,
        featurizer: Optional[Callable] = None,
        retrain: bool = False,
        **kwargs,
    ) -> bool:
        X = self.feature_source(list(xs))
        self.surrogate.fit(X, np.asarray(ys))
        return True

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        X = self.feature_source(list(xs))
        mu, _ = self.surrogate.predict(X)
        return mu

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self.feature_source(list(xs))
        mu, sigma = self.surrogate.predict(X)
        return mu, sigma**2  # Model ABC expects variance, not std

    def apply(self, x_ids, x_feats, batched_size=None, size=None, mean_only=True):
        xs = list(x_ids)
        if mean_only:
            return self.get_means(xs), np.array([])
        return self.get_means_and_vars(xs)

    def save(self, path) -> str:
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.surrogate._net.state_dict(),
                "ym": self.surrogate._ym,
                "ys": self.surrogate._ys,
            },
            path,
        )
        return str(path)

    def load(self, path):
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        ckpt = torch.load(path, map_location=DEVICE)
        self.surrogate._net.load_state_dict(ckpt["state_dict"])
        self.surrogate._ym = ckpt["ym"]
        self.surrogate._ys = ckpt["ys"]
