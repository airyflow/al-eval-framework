"""Feature sources bridge SMILES -> feature matrix for the surrogate model.

Two kinds, matching configs/representation/*.yaml:
  - MorganFeatureSource      computed on the fly (molpal.featurizer.Featurizer)
  - EmbeddingFeatureSource   looks up rows in one or more precomputed .npz
                             embedding files (molformer/unimol/fusional_lite)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np


class FeatureSource:
    def __call__(self, smis: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class MorganFeatureSource(FeatureSource):
    def __init__(self, radius: int = 2, length: int = 2048):
        from molpal.featurizer import Featurizer

        self._f = Featurizer(fingerprint="morgan", radius=radius, length=length)

    def __call__(self, smis: Sequence[str]) -> np.ndarray:
        feats = [self._f(s) for s in smis]
        return np.stack(
            [f if f is not None else np.zeros(len(self._f)) for f in feats]
        ).astype(np.float32)

    @property
    def dim(self) -> int:
        return len(self._f)


class EmbeddingFeatureSource(FeatureSource):
    """Wraps one or more pre-extracted .npz embedding files (each holding
    'embeddings' (N, D) and 'smiles' (N,)). With more than one backbone,
    l2-normalizes each view then concatenates -- the FusionAL construction
    (Eq. concat, paper draft Sec 5.1).
    """

    def __init__(self, npz_paths: Dict[str, str], l2_normalize: bool = False):
        self._backbones = list(npz_paths.keys())
        self._l2 = l2_normalize
        self._arrays: Dict[str, np.ndarray] = {}

        smi2idx_ref = None
        for bb, path in npz_paths.items():
            data = np.load(path, allow_pickle=False)
            emb, smiles = data["embeddings"], data["smiles"]
            smi2idx = {s: i for i, s in enumerate(smiles)}
            if smi2idx_ref is None:
                smi2idx_ref = smi2idx
                self._order = list(smiles)
            elif set(smi2idx) != set(smi2idx_ref):
                raise ValueError(
                    f"backbone '{bb}' SMILES set does not match backbone "
                    f"'{self._backbones[0]}' -- embeddings were extracted "
                    f"over different pools."
                )
            self._arrays[bb] = emb.astype(np.float32)
            self._smi2idx_per_bb = getattr(self, "_smi2idx_per_bb", {})
            self._smi2idx_per_bb[bb] = smi2idx

    def __call__(self, smis: Sequence[str]) -> np.ndarray:
        parts = []
        for bb in self._backbones:
            idxs = [self._smi2idx_per_bb[bb][s] for s in smis]
            x = self._arrays[bb][idxs]
            if self._l2:
                norm = np.linalg.norm(x, axis=1, keepdims=True)
                x = x / np.clip(norm, 1e-12, None)
            parts.append(x)
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    @property
    def dim(self) -> int:
        return sum(a.shape[1] for a in self._arrays.values())


def build_feature_source(rep_config: dict, target: str) -> FeatureSource:
    """Construct a FeatureSource from a configs/representation/*.yaml dict."""
    kind = rep_config["kind"]

    if kind == "precomputed_no_extraction":
        return MorganFeatureSource(
            radius=rep_config.get("radius", 2), length=rep_config.get("length", 2048)
        )

    if kind == "frozen_embedding":
        (backbone,) = rep_config["backbones"]
        path = rep_config["embed_path"].format(target=target)
        return EmbeddingFeatureSource({backbone: path}, l2_normalize=False)

    if kind == "frozen_embedding_fusion":
        paths = {
            bb: p.format(target=target) for bb, p in rep_config["embed_paths"].items()
        }
        return EmbeddingFeatureSource(
            paths, l2_normalize=rep_config.get("l2_normalize_per_view", True)
        )

    raise ValueError(f"Unrecognized representation kind: {kind}")
