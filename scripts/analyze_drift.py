#!/usr/bin/env python
"""Analyze and visualize embedding drift patterns between model versions.

This script characterizes the structure of embedding drift to motivate the
local-adapter hypothesis: if drift varies meaningfully across regions of
the embedding space, per-cluster adapters should outperform a single
global adapter.

Usage
-----
Basic drift analysis::

    python scripts/analyze_drift.py \\
        --pairs data/pairs/minilm-6-to-12_msmarco.npz \\
        --output experiments/figures/

With clustering analysis (requires a cluster count)::

    python scripts/analyze_drift.py \\
        --pairs data/pairs/minilm-6-to-12_msmarco.npz \\
        --output experiments/figures/ \\
        --n-clusters 8

With drift-aware clustering instead of k-means::

    python scripts/analyze_drift.py \\
        --pairs data/pairs/minilm-6-to-12_msmarco.npz \\
        --output experiments/figures/ \\
        --n-clusters 8 \\
        --cluster-method drift_aware

Outputs
-------
- ``drift_magnitude_distribution.png`` -- Histogram of drift magnitudes.
- ``tsne_drift_magnitude.png`` -- t-SNE colored by drift magnitude.
- ``tsne_clusters.png`` -- t-SNE colored by cluster assignment (if
  clustering is enabled).
- ``cluster_drift_comparison.png`` -- Box plot of per-cluster drift
  magnitudes (if clustering is enabled).
- ``drift_statistics.json`` -- Numeric drift statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Ensure the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.adapters import OrthogonalProcrustes
from src.clustering import DriftAwareClustering, KMeansClustering
from src.data.pair_generator import EmbeddingPairGenerator
from src.evaluation.analysis import (
    compute_drift_vectors,
    drift_statistics,
    per_cluster_drift_stats,
    plot_cluster_drift_comparison,
    plot_drift_magnitude_distribution,
    plot_tsne_with_clusters,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Additional visualisation: t-SNE colored by drift magnitude
# -----------------------------------------------------------------------

def _plot_tsne_by_drift_magnitude(
    embeddings: np.ndarray,
    drift_magnitudes: np.ndarray,
    save_path: Path,
    n_samples: int = 5000,
) -> None:
    """t-SNE of old embeddings colored by drift magnitude.

    This reveals whether high-drift regions are spatially localised in
    the embedding space (supporting the local-adapter hypothesis).
    """
    n = embeddings.shape[0]
    if n > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_samples, replace=False)
        embeddings = embeddings[idx]
        drift_magnitudes = drift_magnitudes[idx]

    logger.info("Computing t-SNE for %d points ...", embeddings.shape[0])
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings) - 1),
    )
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=drift_magnitudes,
        cmap="viridis",
        s=5,
        alpha=0.6,
    )
    fig.colorbar(scatter, ax=ax, label="Drift magnitude (L2)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Embedding space colored by drift magnitude")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved t-SNE drift-magnitude plot to %s", save_path)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize embedding drift patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pre-generated embedding pairs (.npz file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/figures/",
        help="Output directory for plots and statistics (default: experiments/figures/).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Number of clusters for per-cluster drift analysis.  If not "
            "provided, only global drift statistics and plots are generated."
        ),
    )
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "drift_aware"],
        default="kmeans",
        help="Clustering method (default: kmeans).",
    )
    parser.add_argument(
        "--drift-weight",
        type=float,
        default=0.5,
        help="Drift weight for drift-aware clustering (default: 0.5).",
    )
    parser.add_argument(
        "--max-tsne-samples",
        type=int,
        default=5000,
        help="Maximum points for t-SNE visualisations (default: 5000).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load pairs ----
    logger.info("Loading embedding pairs from %s ...", args.pairs)
    X_old, X_new = EmbeddingPairGenerator.load_pairs(args.pairs)
    logger.info(
        "Loaded %d pairs: X_old %s, X_new %s",
        X_old.shape[0], X_old.shape, X_new.shape,
    )

    # Handle dimension mismatch: pad to equal dims for Procrustes
    if X_old.shape[1] != X_new.shape[1]:
        logger.info(
            "Dimension mismatch (old=%d, new=%d). Padding for Procrustes analysis.",
            X_old.shape[1], X_new.shape[1],
        )
        X_old_padded, X_new_padded = EmbeddingPairGenerator.pad_to_match(X_old, X_new)
    else:
        X_old_padded, X_new_padded = X_old, X_new

    # ---- Fit global Procrustes ----
    logger.info("Fitting global Procrustes alignment ...")
    procrustes = OrthogonalProcrustes(fit_scaling=False)
    procrustes.fit(X_old_padded, X_new_padded)

    # ---- Compute drift vectors (residuals after Procrustes) ----
    drift_vectors = compute_drift_vectors(
        X_old_padded, X_new_padded, transform_fn=procrustes.transform,
    )
    drift_magnitudes = np.linalg.norm(drift_vectors, axis=1)

    # ---- Global drift statistics ----
    stats = drift_statistics(drift_vectors)

    # Convert non-serialisable values for JSON output
    stats_serialisable = {
        "mean_magnitude": stats["mean_magnitude"],
        "std_magnitude": stats["std_magnitude"],
        "min_magnitude": stats["min_magnitude"],
        "max_magnitude": stats["max_magnitude"],
        "n_samples": int(X_old.shape[0]),
        "old_dim": int(X_old.shape[1]),
        "new_dim": int(X_new.shape[1]),
    }

    print("\n" + "=" * 60)
    print("GLOBAL DRIFT STATISTICS (after Procrustes alignment)")
    print("=" * 60)
    print(f"  Number of samples:     {X_old.shape[0]:,}")
    print(f"  Embedding dimensions:  old={X_old.shape[1]}, new={X_new.shape[1]}")
    print(f"  Mean drift magnitude:  {stats['mean_magnitude']:.6f}")
    print(f"  Std drift magnitude:   {stats['std_magnitude']:.6f}")
    print(f"  Min drift magnitude:   {stats['min_magnitude']:.6f}")
    print(f"  Max drift magnitude:   {stats['max_magnitude']:.6f}")
    print(f"  Drift range ratio:     {stats['max_magnitude'] / max(stats['min_magnitude'], 1e-12):.2f}x")
    print()

    # ---- Plot: drift magnitude distribution ----
    plot_drift_magnitude_distribution(
        drift_vectors,
        save_path=output_dir / "drift_magnitude_distribution.png",
    )
    logger.info("Saved drift magnitude histogram.")

    # ---- Plot: t-SNE colored by drift magnitude ----
    _plot_tsne_by_drift_magnitude(
        X_old_padded,
        drift_magnitudes,
        save_path=output_dir / "tsne_drift_magnitude.png",
        n_samples=args.max_tsne_samples,
    )

    # ---- Optional clustering analysis ----
    if args.n_clusters is not None:
        k = args.n_clusters
        logger.info(
            "Running %s clustering with k=%d ...", args.cluster_method, k
        )

        if args.cluster_method == "kmeans":
            clustering = KMeansClustering(n_clusters=k)
            clustering.fit(X_old_padded)
            cluster_labels = clustering.predict(X_old_padded)
        else:
            clustering = DriftAwareClustering(
                n_clusters=k, drift_weight=args.drift_weight,
            )
            clustering.fit(X_old_padded, X_new_padded)
            cluster_labels = clustering.predict(X_old_padded, X_new_padded)

        # Per-cluster statistics
        cluster_stats = per_cluster_drift_stats(drift_vectors, cluster_labels)
        stats_serialisable["per_cluster"] = {
            str(label): vals for label, vals in cluster_stats.items()
        }
        stats_serialisable["clustering"] = {
            "method": args.cluster_method,
            "n_clusters": k,
        }

        # Print per-cluster table
        print("PER-CLUSTER DRIFT STATISTICS")
        print("-" * 60)
        header = f"{'Cluster':>8}  {'N':>7}  {'Mean Mag':>10}  {'Std Mag':>10}  {'Dir Consist':>12}"
        print(header)
        print("-" * 60)
        for label in sorted(cluster_stats.keys()):
            cs = cluster_stats[label]
            print(
                f"{label:>8}  {cs['n_samples']:>7}  "
                f"{cs['mean_magnitude']:>10.6f}  "
                f"{cs['std_magnitude']:>10.6f}  "
                f"{cs['direction_consistency']:>12.4f}"
            )
        print()

        # Plot: t-SNE with cluster coloring
        plot_tsne_with_clusters(
            X_old_padded,
            cluster_labels,
            save_path=output_dir / "tsne_clusters.png",
            n_samples=args.max_tsne_samples,
        )
        logger.info("Saved t-SNE cluster plot.")

        # Plot: per-cluster drift comparison
        plot_cluster_drift_comparison(
            drift_vectors,
            cluster_labels,
            save_path=output_dir / "cluster_drift_comparison.png",
        )
        logger.info("Saved cluster drift comparison plot.")

    # ---- Save statistics JSON ----
    stats_path = output_dir / "drift_statistics.json"
    with open(stats_path, "w") as fh:
        json.dump(stats_serialisable, fh, indent=2)
    logger.info("Saved drift statistics to %s", stats_path)

    # ---- Summary of generated files ----
    print("OUTPUT FILES")
    print("-" * 60)
    for p in sorted(output_dir.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<45} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
