"""Analysis utilities for understanding embedding drift patterns.

These functions help diagnose *how* embeddings change across model versions,
informing adapter design decisions (e.g. whether drift is uniform or
cluster-dependent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_drift_vectors(
    X_old: np.ndarray,
    X_new: np.ndarray,
    transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Compute per-sample drift vectors between old and new embeddings.

    Parameters
    ----------
    X_old:
        Old-model embeddings of shape ``(n, d_old)``.
    X_new:
        New-model embeddings of shape ``(n, d_new)``.
    transform_fn:
        Optional adapter function mapping old embeddings to the new space.
        When provided the drift is ``X_new - transform_fn(X_old)`` (i.e.
        *residual* drift after adaptation).  When ``None``, raw drift
        ``X_new - X_old`` is returned (requires ``d_old == d_new``).

    Returns
    -------
    Drift vectors of shape ``(n, d_new)``.
    """
    if transform_fn is not None:
        return X_new - transform_fn(X_old)

    if X_old.shape[1] != X_new.shape[1]:
        raise ValueError(
            "Dimensions differ (d_old=%d, d_new=%d). Provide a transform_fn "
            "to project X_old into the new embedding space." % (X_old.shape[1], X_new.shape[1])
        )
    return X_new - X_old


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def drift_statistics(drift_vectors: np.ndarray) -> dict:
    """Aggregate statistics for a set of drift vectors.

    Parameters
    ----------
    drift_vectors:
        Array of shape ``(n, d)``.

    Returns
    -------
    Dictionary with the following keys:

    - ``mean_magnitude``: mean L2 norm of drift vectors.
    - ``std_magnitude``: standard deviation of L2 norms.
    - ``min_magnitude``: minimum L2 norm.
    - ``max_magnitude``: maximum L2 norm.
    - ``mean_direction``: unit vector of the mean drift (shape ``(d,)``).
    - ``per_dim_mean``: per-dimension mean drift (shape ``(d,)``).
    - ``per_dim_std``: per-dimension std of drift (shape ``(d,)``).
    """
    magnitudes = np.linalg.norm(drift_vectors, axis=1)
    mean_drift = drift_vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_drift)
    mean_direction = mean_drift / (mean_norm + 1e-12)

    return {
        "mean_magnitude": float(magnitudes.mean()),
        "std_magnitude": float(magnitudes.std()),
        "min_magnitude": float(magnitudes.min()),
        "max_magnitude": float(magnitudes.max()),
        "mean_direction": mean_direction,
        "per_dim_mean": drift_vectors.mean(axis=0),
        "per_dim_std": drift_vectors.std(axis=0),
    }


def per_cluster_drift_stats(
    drift_vectors: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict:
    """Drift statistics broken down by cluster assignment.

    Parameters
    ----------
    drift_vectors:
        Array of shape ``(n, d)``.
    cluster_labels:
        Integer cluster labels of shape ``(n,)``.

    Returns
    -------
    Dictionary keyed by cluster label, each containing:

    - ``mean_magnitude``: mean L2 norm within the cluster.
    - ``std_magnitude``: std of L2 norms within the cluster.
    - ``n_samples``: number of samples in the cluster.
    - ``direction_consistency``: mean pairwise cosine similarity of drift
      vectors within the cluster (higher = more coherent drift).
    """
    unique_labels = np.unique(cluster_labels)
    stats: dict[int, dict] = {}

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_drift = drift_vectors[mask]
        magnitudes = np.linalg.norm(cluster_drift, axis=1)

        # Direction consistency: cosine similarity of each vector with the
        # cluster mean drift direction.
        mean_vec = cluster_drift.mean(axis=0)
        mean_norm = np.linalg.norm(mean_vec)
        if mean_norm < 1e-12:
            consistency = 0.0
        else:
            unit_mean = mean_vec / mean_norm
            norms = magnitudes.copy()
            norms[norms < 1e-12] = 1e-12
            cosines = (cluster_drift @ unit_mean) / norms
            consistency = float(cosines.mean())

        stats[int(label)] = {
            "mean_magnitude": float(magnitudes.mean()),
            "std_magnitude": float(magnitudes.std()),
            "n_samples": int(mask.sum()),
            "direction_consistency": consistency,
        }

    return stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_drift_magnitude_distribution(
    drift_vectors: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Histogram of per-sample drift magnitudes.

    Parameters
    ----------
    drift_vectors:
        Array of shape ``(n, d)``.
    save_path:
        If provided, save the figure to this path instead of showing it.
    """
    magnitudes = np.linalg.norm(drift_vectors, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(magnitudes, bins=50, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Drift magnitude (L2 norm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of embedding drift magnitudes")
    fig.tight_layout()

    _save_or_show(fig, save_path)


def plot_cluster_drift_comparison(
    drift_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Box plot of drift magnitudes grouped by cluster.

    Parameters
    ----------
    drift_vectors:
        Array of shape ``(n, d)``.
    cluster_labels:
        Integer cluster labels of shape ``(n,)``.
    save_path:
        If provided, save the figure to this path instead of showing it.
    """
    magnitudes = np.linalg.norm(drift_vectors, axis=1)
    unique_labels = np.sort(np.unique(cluster_labels))

    data = [magnitudes[cluster_labels == label] for label in unique_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(unique_labels) * 0.8), 5))
    ax.boxplot(data, labels=[str(label) for label in unique_labels], patch_artist=True)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Drift magnitude (L2 norm)")
    ax.set_title("Drift magnitude by cluster")
    fig.tight_layout()

    _save_or_show(fig, save_path)


def plot_tsne_with_clusters(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    n_samples: int = 5000,
) -> None:
    """t-SNE visualisation of embeddings coloured by cluster label.

    Parameters
    ----------
    embeddings:
        Embedding matrix of shape ``(n, d)``.
    cluster_labels:
        Integer cluster labels of shape ``(n,)``.
    save_path:
        If provided, save the figure to this path instead of showing it.
    n_samples:
        Maximum number of points to plot.  If the dataset is larger,
        a random subsample is drawn for readability and performance.
    """
    n = embeddings.shape[0]
    if n > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_samples, replace=False)
        embeddings = embeddings[idx]
        cluster_labels = cluster_labels[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=5,
        alpha=0.6,
    )
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of embeddings coloured by cluster")
    fig.tight_layout()

    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    """Save a matplotlib figure or display it interactively."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
