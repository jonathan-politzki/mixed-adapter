"""Drift-aware clustering of embedding spaces.

Clusters are formed using *both* the position of each sample in the
original embedding space and the residual drift vector left after
applying the global Procrustes alignment.  This lets the downstream
mixture-of-experts system assign a specialised local adapter to each
region of the space that experiences a qualitatively different kind of
distributional shift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.cluster import KMeans


def _compute_procrustes(
    X_old: np.ndarray,
    X_new: np.ndarray,
) -> np.ndarray:
    """Compute the orthogonal Procrustes solution mapping *X_old* to *X_new*.

    Returns the rotation matrix Q that minimises ||X_new - X_old @ Q||_F.

    Parameters
    ----------
    X_old, X_new : np.ndarray
        Aligned embedding matrices of shape ``(n, d)``.

    Returns
    -------
    np.ndarray
        Orthogonal matrix Q of shape ``(d, d)``.
    """
    M = X_old.T @ X_new
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def _normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalise, handling zero-norm rows gracefully."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


class DriftAwareClustering:
    """Partition an embedding space by combining positional and drift features.

    For every sample *i* the drift residual is defined as

        d_i = x_new_i - Q @ x_old_i

    where Q is the global Procrustes rotation.  The clustering feature
    vector is the concatenation of the L2-normalised old-space position
    and the (weighted, normalised) drift residual.  A standard K-means
    is then run on this combined space.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    drift_weight : float
        Relative importance of drift features versus positional features
        when building the combined feature matrix.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        drift_weight: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.drift_weight = drift_weight
        self.random_state = random_state

        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        self._global_transform: Optional[np.ndarray] = None
        self._drift_vectors: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._position_dim: Optional[int] = None
        self._old_dim: Optional[int] = None
        self._new_dim: Optional[int] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        global_transform: Optional[np.ndarray] = None,
    ) -> DriftAwareClustering:
        """Fit clusters on the combined position + drift feature space.

        When ``X_old`` and ``X_new`` have different embedding dimensions
        (e.g. 384 vs 768), the smaller matrix is zero-padded internally
        so that Procrustes alignment and drift computation can proceed in
        a common dimensionality.  The original dimensions are stored so
        that :meth:`predict` can apply the same padding transparently.

        Parameters
        ----------
        X_old : np.ndarray
            Old-space embeddings, shape ``(n, d_old)``.
        X_new : np.ndarray
            New-space embeddings, shape ``(n, d_new)``.
        global_transform : np.ndarray, optional
            Orthogonal matrix Q of shape ``(d_max, d_max)`` where
            ``d_max = max(d_old, d_new)``.  If ``None``, a Procrustes
            solution is computed from the (possibly padded) data.

        Returns
        -------
        DriftAwareClustering
            The fitted instance.
        """
        if X_old.shape[0] != X_new.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X_old has {X_old.shape[0]} rows "
                f"vs X_new has {X_new.shape[0]} rows"
            )

        self._old_dim = X_old.shape[1]
        self._new_dim = X_new.shape[1]

        # Zero-pad to a common dimension when shapes differ
        X_old_aligned, X_new_aligned = _pad_to_common_dim(X_old, X_new)

        if global_transform is None:
            global_transform = _compute_procrustes(X_old_aligned, X_new_aligned)
        self._global_transform = global_transform

        drift_vectors = X_new_aligned - X_old_aligned @ global_transform
        self._drift_vectors = drift_vectors
        self._position_dim = X_old_aligned.shape[1]

        features = self._build_features(X_old_aligned, drift_vectors)
        self._kmeans.fit(features)
        self._labels = self._kmeans.labels_
        self._is_fitted = True
        return self

    def predict(
        self,
        X_old: np.ndarray,
        X_new: Optional[np.ndarray] = None,
        global_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assign cluster labels to new samples.

        At *training* time both ``X_old`` and ``X_new`` are available so
        the full drift features can be used.  At *inference* time only
        ``X_old`` may be available; in that case assignment is performed
        using nearest-centroid in the positional subspace.

        When the model was fitted with different old/new dimensions, the
        same zero-padding is applied automatically to match the internal
        representation.

        Parameters
        ----------
        X_old : np.ndarray
            Old-space embeddings, shape ``(n, d_old)``.
        X_new : np.ndarray, optional
            New-space embeddings.  When provided, drift features are used.
        global_transform : np.ndarray, optional
            Override the stored Procrustes rotation.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape ``(n,)``.
        """
        self._check_fitted()

        if X_new is not None:
            X_old_aligned, X_new_aligned = _pad_to_common_dim(X_old, X_new)
            Q = global_transform if global_transform is not None else self._global_transform
            drift_vectors = X_new_aligned - X_old_aligned @ Q
            features = self._build_features(X_old_aligned, drift_vectors)
            return self._kmeans.predict(features)

        # Inference: pad X_old to the common dimension used during fit
        X_old_padded = _pad_array(X_old, self._position_dim)
        position_features = _normalize(X_old_padded)
        centroids_position = self._kmeans.cluster_centers_[:, : self._position_dim]
        distances = _pairwise_l2(position_features, centroids_position)
        return np.argmin(distances, axis=1)

    def get_drift_statistics(self) -> dict[int, dict[str, float]]:
        """Per-cluster drift diagnostics.

        Returns
        -------
        dict[int, dict[str, float]]
            For each cluster: ``mean_magnitude`` (average L2 drift norm)
            and ``direction_variance`` (variance of cosine similarities
            among drift vectors within the cluster).
        """
        self._check_fitted()
        stats: dict[int, dict[str, float]] = {}

        for k in range(self.n_clusters):
            mask = self._labels == k
            drifts = self._drift_vectors[mask]

            magnitudes = np.linalg.norm(drifts, axis=1)
            mean_mag = float(np.mean(magnitudes)) if len(magnitudes) > 0 else 0.0

            if len(drifts) > 1:
                normed = _normalize(drifts)
                cosines = normed @ normed.T
                upper = cosines[np.triu_indices_from(cosines, k=1)]
                dir_var = float(np.var(upper)) if len(upper) > 0 else 0.0
            else:
                dir_var = 0.0

            stats[k] = {
                "mean_magnitude": mean_mag,
                "direction_variance": dir_var,
            }

        return stats

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> DriftAwareClustering:
        """Restore a saved ``DriftAwareClustering`` instance.

        Parameters
        ----------
        path : str or Path
            Path to the joblib file.

        Returns
        -------
        DriftAwareClustering
            The restored instance.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(
        self,
        X_old: np.ndarray,
        drift_vectors: np.ndarray,
    ) -> np.ndarray:
        """Concatenate normalised position and weighted drift features."""
        pos = _normalize(X_old)
        drift = _normalize(drift_vectors) * self.drift_weight
        return np.concatenate([pos, drift], axis=1)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "DriftAwareClustering has not been fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"DriftAwareClustering(n_clusters={self.n_clusters}, "
            f"drift_weight={self.drift_weight}, {status})"
        )


def _pad_array(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad *X* along the feature axis to reach *target_dim* columns.

    If ``X`` already has ``target_dim`` columns the input is returned
    unchanged.  An error is raised if ``X`` has *more* columns than
    ``target_dim``.
    """
    d = X.shape[1]
    if d == target_dim:
        return X
    if d > target_dim:
        raise ValueError(
            f"Cannot pad: X has {d} columns but target_dim is {target_dim}"
        )
    pad = np.zeros((X.shape[0], target_dim - d), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def _pad_to_common_dim(
    X_old: np.ndarray,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the smaller array so both share the same feature dimension."""
    d_old = X_old.shape[1]
    d_new = X_new.shape[1]
    if d_old == d_new:
        return X_old, X_new
    d_max = max(d_old, d_new)
    return _pad_array(X_old, d_max), _pad_array(X_new, d_max)


def _pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distances between rows of A and rows of B.

    Parameters
    ----------
    A : np.ndarray
        Shape ``(n, d)``.
    B : np.ndarray
        Shape ``(m, d)``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n, m)``.
    """
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)
    return A_sq + B_sq.T - 2.0 * A @ B.T
