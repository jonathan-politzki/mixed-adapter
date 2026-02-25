from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


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

        Parameters
        ----------
        X_old:
            Source embeddings of shape ``(n, d)``.
        X_new:
            Target embeddings of shape ``(n, d)``.

        Returns
        -------
        self
        """
        if X_old.shape != X_new.shape:
            raise ValueError(
                f"Shape mismatch: X_old {X_old.shape} vs X_new {X_new.shape}"
            )

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

        Parameters
        ----------
        X:
            Embeddings of shape ``(n, d)`` in the old model's space.

        Returns
        -------
        Aligned embeddings of shape ``(n, d)``.
        """
        if self.Q is None:
            raise RuntimeError("Call fit() before transform().")

        out = X @ self.Q
        if self.D is not None:
            out = out @ self.D
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
