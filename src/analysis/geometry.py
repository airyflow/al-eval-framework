"""Representation-only (label-free) geometry diagnostics.

These characterize the chemical space induced by an encoder phi: X -> R^d,
independent of any oracle score or AL run -- they can be computed the
moment embeddings exist. They extend criteria.py's C1/C4 (which need
oracle scores) with metrics answering "what kind of manifold did this
encoder create?" rather than "does it track this task?".

d_int (intrinsic dimension) -> intrinsic_dim_twonn, participation_ratio, pca_dim_at_variance
NS_k (neighborhood stability) -> neighborhood_stability
density / hubness -> density_stats

Note: the proposal's "distance-property correlation" rho_phi is
algebraically identical to criteria.py's local_smoothness (both are
Spearman(||phi_i - phi_j||, |y_i - y_j|)) -- reuse that function rather
than duplicating it; see src/analysis/run_geometry.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors


def _subsample(embeddings: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    n = len(embeddings)
    if max_samples is None or n <= max_samples:
        return embeddings
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=max_samples, replace=False)
    return embeddings[idxs]


# ── Intrinsic dimensionality ─────────────────────────────────────────────────

def intrinsic_dim_twonn(
    embeddings: np.ndarray, discard_fraction: float = 0.1,
    max_samples: Optional[int] = 20_000, seed: int = 0,
) -> float:
    """TwoNN estimator (Facco et al. 2017, "Estimating the intrinsic
    dimension of datasets by a minimal neighborhood information").

    For each point, mu_i = r2_i / r1_i (ratio of its 2nd- to 1st-nearest-
    neighbor distance). Under a locally-uniform-density assumption, mu
    follows a Pareto distribution with shape parameter d (the intrinsic
    dimension), giving the closed-form MLE d_hat = N / sum(log(mu_i)).

    The top `discard_fraction` of mu values are dropped before fitting --
    the largest ratios are the most sensitive to density non-uniformity
    and are the standard robustness trick in the original paper.
    """
    X = _subsample(embeddings, max_samples, seed)
    n = len(X)
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]

    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1]  # mu < 1 can only occur from floating-point ties at r1==r2

    mu_sorted = np.sort(mu)
    if discard_fraction > 0:
        cutoff = max(1, int(len(mu_sorted) * (1 - discard_fraction)))
        mu_sorted = mu_sorted[:cutoff]

    return float(len(mu_sorted) / np.sum(np.log(mu_sorted)))


def _pca_eigen(embeddings: np.ndarray, max_samples: Optional[int], seed: int, max_components: int = 300):
    """PCA fit, shared by participation_ratio/pca_dim_at_variance/
    pca_dimensionality so multi-metric callers (compute_geometry_row) don't
    pay for the same SVD twice.

    Exact full-spectrum PCA (svd_solver='full', the sklearn default when
    n_components=None) is O(min(n,d)^2 * max(n,d)) -- on 20K x 2048 Morgan
    fingerprints that never finished in a reasonable time (observed
    directly: killed after 14+ CPU-minutes with no result). Truncated
    randomized SVD is the standard fix: capped to `max_components`, cost
    scales with the requested rank instead of the full spectrum. This
    slightly *underestimates* participation_ratio and pca_dim_90pct when
    the true effective dimension exceeds max_components, which is an
    accepted, documented approximation rather than a silent one.
    """
    from sklearn.decomposition import PCA

    X = _subsample(embeddings, max_samples, seed)
    n_components = min(max_components, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(X)
    return p.explained_variance_, p.explained_variance_ratio_


def participation_ratio(
    embeddings: np.ndarray, max_samples: Optional[int] = 20_000, seed: int = 0,
) -> float:
    """Effective number of dimensions from the PCA eigenvalue spectrum:

        PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

    PR = d (the ambient dimension) if variance is spread evenly across all
    axes; PR -> 1 if variance concentrates on a single axis. Complements
    intrinsic_dim_twonn (a local, neighbor-based estimate) with a global,
    linear-subspace estimate.
    """
    lam, _ = _pca_eigen(embeddings, max_samples, seed)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def _pca_dim_from_ratio(ratio: np.ndarray, threshold: float) -> tuple:
    """Returns (dim, truncated). `ratio` may itself be truncated to
    max_components (see _pca_eigen) -- if the cumulative sum never reaches
    `threshold` within it, that's not "dim = max_components + 1", it's
    "unknown, but at least max_components"; report that honestly via the
    truncated flag rather than a number that looks precise but isn't."""
    cum = np.cumsum(ratio)
    idx = np.searchsorted(cum, threshold)
    if idx >= len(ratio):
        return len(ratio), True
    return int(idx + 1), False


def pca_dim_at_variance(
    embeddings: np.ndarray, threshold: float = 0.9,
    max_samples: Optional[int] = 20_000, seed: int = 0,
) -> int:
    """Number of principal components needed to explain `threshold` of the
    total variance -- a coarser, easier-to-interpret companion to
    participation_ratio. Returns -1 if that isn't reached within the
    (possibly truncated, see _pca_eigen) computed spectrum."""
    _, ratio = _pca_eigen(embeddings, max_samples, seed)
    dim, truncated = _pca_dim_from_ratio(ratio, threshold)
    return -1 if truncated else dim


def pca_dimensionality(
    embeddings: np.ndarray, threshold: float = 0.9,
    max_samples: Optional[int] = 20_000, seed: int = 0,
) -> dict:
    """participation_ratio + pca_dim_at_variance from a single PCA fit --
    use this instead of calling both separately when you need both."""
    lam, ratio = _pca_eigen(embeddings, max_samples, seed)
    dim, truncated = _pca_dim_from_ratio(ratio, threshold)
    return {
        "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "pca_dim_90pct": dim,
        "pca_dim_90pct_truncated": truncated,
    }


# ── Neighborhood stability ───────────────────────────────────────────────────

def neighborhood_stability(
    embeddings: np.ndarray, ref_embeddings: np.ndarray, k: int = 10,
    max_anchors: Optional[int] = 5_000, seed: int = 0,
) -> float:
    """NS_k: mean fraction of a molecule's k nearest neighbors in phi-space
    that are also its k nearest neighbors in a reference space (Morgan
    fingerprints by default -- a fixed, chemistry-native notion of
    similarity). Label-free counterpart to criteria.py's
    knn_target_precision (which compares against oracle score neighbors
    instead of a reference embedding).

    `embeddings` and `ref_embeddings` must be row-aligned (same molecule
    order).
    """
    n = len(embeddings)
    assert len(ref_embeddings) == n, "embeddings and ref_embeddings must be row-aligned"
    k = min(k, n - 1)

    if max_anchors is not None and n > max_anchors:
        rng = np.random.default_rng(seed)
        anchors = rng.choice(n, size=max_anchors, replace=False)
    else:
        anchors = np.arange(n)

    nn_phi = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, idx_phi = nn_phi.kneighbors(embeddings[anchors])

    nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(ref_embeddings)
    _, idx_ref = nn_ref.kneighbors(ref_embeddings[anchors])

    overlaps = np.empty(len(anchors))
    for row, a in enumerate(anchors):
        phi_neighbors = set(idx_phi[row].tolist()) - {a}
        ref_neighbors = set(idx_ref[row].tolist()) - {a}
        overlaps[row] = len(phi_neighbors & ref_neighbors) / k

    return float(overlaps.mean())


# ── Density / hubness ────────────────────────────────────────────────────────

def density_stats(
    embeddings: np.ndarray, k: int = 10,
    max_samples: Optional[int] = 20_000, seed: int = 0,
) -> dict:
    """rho_i = mean distance to i's k nearest neighbors; reports Var(rho_i)
    (density variance -- large values indicate collapsed/over-concentrated
    regions coexisting with sparse ones) and the skewness of the
    k-occurrence distribution (hubness, Radovanovic et al. 2010 -- how
    unevenly points are chosen as others' nearest neighbors; high positive
    skew means a few "hub" points dominate neighbor lists, which distorts
    density- or neighbor-based AL acquisition)."""
    X = _subsample(embeddings, max_samples, seed)
    n = len(X)
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    rho = dist[:, 1:].mean(axis=1)  # exclude self (column 0)

    k_occurrence = np.bincount(ind[:, 1:].ravel(), minlength=n)

    return {
        "mean_local_density": float(rho.mean()),
        "density_variance": float(np.var(rho)),
        "hubness_skewness": float(skew(k_occurrence)),
    }


# ── Convenience aggregator (one row of the geometry table) ──────────────────

def compute_geometry_row(
    embeddings: np.ndarray, ref_embeddings: np.ndarray, k: int = 10,
    pca_variance_threshold: float = 0.9, max_samples: int = 20_000,
    max_anchors: int = 5_000, seed: int = 0,
) -> dict:
    return {
        "intrinsic_dim_twonn": intrinsic_dim_twonn(embeddings, max_samples=max_samples, seed=seed),
        **pca_dimensionality(embeddings, pca_variance_threshold, max_samples=max_samples, seed=seed),
        "neighborhood_stability": neighborhood_stability(embeddings, ref_embeddings, k=k, max_anchors=max_anchors, seed=seed),
        **density_stats(embeddings, k=k, max_samples=max_samples, seed=seed),
    }
