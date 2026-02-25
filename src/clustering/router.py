"""Query routing for the mixture-of-experts adapter system.

Given a set of cluster centroids (one per local adapter), the router
computes per-query weight vectors that determine how much each adapter
contributes to the final prediction.  Hard routing selects a single
adapter; soft routing blends them via temperature-scaled softmax over
cosine similarities.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class MoERouter:
    """Mixture-of-experts router based on centroid similarity.

    Parameters
    ----------
    centroids : np.ndarray
        Cluster centroids of shape ``(n_clusters, dim)``.
    temperature : float
        Softmax temperature for soft routing.  Lower values sharpen the
        distribution towards the nearest centroid.
    routing : {'hard', 'soft'}
        Routing strategy.  ``'hard'`` returns one-hot vectors;
        ``'soft'`` returns softmax-weighted distributions.
    """

    def __init__(
        self,
        centroids: np.ndarray,
        temperature: float = 1.0,
        routing: Literal["hard", "soft"] = "soft",
    ) -> None:
        if routing not in ("hard", "soft"):
            raise ValueError(f"routing must be 'hard' or 'soft', got '{routing}'")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self._centroids = centroids.astype(np.float64)
        self._centroid_norms = self._row_normalize(self._centroids)
        self.temperature = temperature
        self.routing = routing

    @property
    def n_clusters(self) -> int:
        """Number of expert clusters."""
        return self._centroids.shape[0]

    @property
    def centroids(self) -> np.ndarray:
        """A copy of the centroid matrix."""
        return self._centroids.copy()

    def route(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Compute routing weights for a batch of query embeddings.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Matrix of shape ``(n_queries, dim)``.

        Returns
        -------
        np.ndarray
            Weight matrix of shape ``(n_queries, n_clusters)``.
            Each row sums to 1.
        """
        similarities = self._cosine_similarity(query_embeddings)

        if self.routing == "hard":
            return self._hard_route(similarities)
        return self._soft_route(similarities)

    def set_temperature(self, tau: float) -> None:
        """Update the softmax temperature.

        Parameters
        ----------
        tau : float
            New temperature value (must be positive).
        """
        if tau <= 0:
            raise ValueError(f"temperature must be positive, got {tau}")
        self.temperature = tau

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, queries: np.ndarray) -> np.ndarray:
        """Cosine similarity between queries and centroids.

        Parameters
        ----------
        queries : np.ndarray
            Shape ``(n_queries, dim)``.

        Returns
        -------
        np.ndarray
            Similarity matrix of shape ``(n_queries, n_clusters)``.
        """
        query_normed = self._row_normalize(queries.astype(np.float64))
        return query_normed @ self._centroid_norms.T

    @staticmethod
    def _hard_route(similarities: np.ndarray) -> np.ndarray:
        """Return one-hot routing from argmax of similarities."""
        n_queries, n_clusters = similarities.shape
        indices = np.argmax(similarities, axis=1)
        one_hot = np.zeros((n_queries, n_clusters), dtype=np.float64)
        one_hot[np.arange(n_queries), indices] = 1.0
        return one_hot

    def _soft_route(self, similarities: np.ndarray) -> np.ndarray:
        """Return softmax-weighted routing."""
        logits = similarities / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _row_normalize(X: np.ndarray) -> np.ndarray:
        """L2-normalise each row, mapping zero vectors to zero."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    def __repr__(self) -> str:
        return (
            f"MoERouter(n_clusters={self.n_clusters}, "
            f"temperature={self.temperature}, routing='{self.routing}')"
        )
