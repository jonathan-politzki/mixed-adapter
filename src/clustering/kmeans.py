"""Standard K-means clustering on embedding spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.cluster import KMeans


class KMeansClustering:
    """K-means partitioning of an embedding space.

    Wraps scikit-learn's K-means with convenience methods for
    serialisation and cluster diagnostics used downstream by the
    local-adapter training pipeline.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_clusters: int, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray) -> KMeansClustering:
        """Fit k-means to an (n_samples, dim) embedding matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Matrix of shape ``(n_samples, dim)``.

        Returns
        -------
        KMeansClustering
            The fitted instance (for method chaining).
        """
        self._model.fit(embeddings)
        self._is_fitted = True
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return cluster assignments for each row in *embeddings*.

        Parameters
        ----------
        embeddings : np.ndarray
            Matrix of shape ``(n_samples, dim)``.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape ``(n_samples,)``.
        """
        self._check_fitted()
        return self._model.predict(embeddings)

    def get_centroids(self) -> np.ndarray:
        """Return the cluster centroids.

        Returns
        -------
        np.ndarray
            Centroid matrix of shape ``(n_clusters, dim)``.
        """
        self._check_fitted()
        return self._model.cluster_centers_

    def get_cluster_sizes(self) -> dict[int, int]:
        """Return the number of training samples assigned to each cluster.

        Returns
        -------
        dict[int, int]
            Mapping from cluster index to sample count.
        """
        self._check_fitted()
        labels, counts = np.unique(self._model.labels_, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to disk with joblib.

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
    def load(cls, path: Union[str, Path]) -> KMeansClustering:
        """Load a previously saved ``KMeansClustering`` from disk.

        Parameters
        ----------
        path : str or Path
            Path to the joblib file.

        Returns
        -------
        KMeansClustering
            The restored instance.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(obj).__name__}"
            )
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "KMeansClustering has not been fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"KMeansClustering(n_clusters={self.n_clusters}, "
            f"random_state={self.random_state}, {status})"
        )
