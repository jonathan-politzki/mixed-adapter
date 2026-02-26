from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def _pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad *X* along the feature axis to *target_dim* columns."""
    if X.shape[1] >= target_dim:
        return X
    pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


class OrthogonalProcrustes:
    """Orthogonal Procrustes alignment with optional diagonal scaling.

    Finds the optimal orthogonal matrix Q that minimizes
    ||X_new - X_old @ Q||_F via the closed-form SVD solution:

        X_new^T @ X_old = U S V^T  =>  Q = V @ U^T

    When diagonal scaling is enabled, a per-dimension scaling matrix D is
    fit after Procrustes so the full transform becomes x @ Q @ D.
    """

    def __init__(self, fit_scaling: bool = False) -> None:
        self.fit_scaling = fit_scaling
        self.Q: Optional[NDArray[np.floating]] = None
        self.D: Optional[NDArray[np.floating]] = None

    def fit(
        self,
        X_old: NDArray[np.floating],
        X_new: NDArray[np.floating],
    ) -> OrthogonalProcrustes:
        """Compute the optimal orthogonal alignment from *X_old* to *X_new*.

        When ``X_old`` and ``X_new`` have different embedding dimensions
        (e.g. 384 vs 768), the smaller matrix is zero-padded so that
        Procrustes can compute a rotation in the common dimensionality.

        Parameters
        ----------
        X_old:
            Source embeddings of shape ``(n, d_old)``.
        X_new:
            Target embeddings of shape ``(n, d_new)``.

        Returns
        -------
        self
        """
        if X_old.shape[0] != X_new.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X_old has {X_old.shape[0]} rows "
                f"vs X_new has {X_new.shape[0]} rows"
            )

        self._input_dim = X_old.shape[1]
        self._output_dim = X_new.shape[1]

        # Zero-pad to common dimension if needed
        if X_old.shape[1] != X_new.shape[1]:
            d_max = max(X_old.shape[1], X_new.shape[1])
            X_old = _pad_to_dim(X_old, d_max)
            X_new = _pad_to_dim(X_new, d_max)

        M = X_new.T @ X_old
        U, _, Vt = np.linalg.svd(M)
        self.Q = Vt.T @ U.T

        if self.fit_scaling:
            X_rotated = X_old @ self.Q
            # Least-squares diagonal: d_j = (X_rotated[:, j]^T X_new[:, j]) / ||X_rotated[:, j]||^2
            numerator = np.sum(X_rotated * X_new, axis=0)
            denominator = np.sum(X_rotated * X_rotated, axis=0) + 1e-12
            self.D = np.diag(numerator / denominator)
        else:
            self.D = None

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the learned alignment to embeddings *X*.

        When the model was fitted with cross-dimensional inputs, *X* is
        zero-padded to the rotation dimension and the output is truncated
        to the target dimension.

        Parameters
        ----------
        X:
            Embeddings of shape ``(n, d_old)`` in the old model's space.

        Returns
        -------
        Aligned embeddings of shape ``(n, d_new)``.
        """
        if self.Q is None:
            raise RuntimeError("Call fit() before transform().")

        # Pad input if rotation matrix is larger than input dim
        d_rot = self.Q.shape[0]
        if X.shape[1] < d_rot:
            X = _pad_to_dim(X, d_rot)

        out = X @ self.Q
        if self.D is not None:
            out = out @ self.D

        # Truncate to target dimension if cross-dimensional
        if hasattr(self, "_output_dim") and self._output_dim < out.shape[1]:
            out = out[:, : self._output_dim]

        return out

    def save(self, path: Union[str, Path]) -> None:
        """Persist Q (and D if present) to a ``.npz`` file."""
        if self.Q is None:
            raise RuntimeError("Nothing to save -- call fit() first.")

        arrays = {"Q": self.Q}
        if self.D is not None:
            arrays["D"] = self.D
        np.savez(path, **arrays)

    def load(self, path: Union[str, Path]) -> OrthogonalProcrustes:
        """Load Q (and D if present) from a ``.npz`` file.

        Returns
        -------
        self
        """
        data = np.load(path)
        self.Q = data["Q"]
        self.D = data["D"] if "D" in data else None
        self.fit_scaling = self.D is not None
        return self
