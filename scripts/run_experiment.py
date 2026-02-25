#!/usr/bin/env python
"""Train and evaluate a drift adapter under a single experimental configuration.

This is the core experiment runner for the local-drift-adapter paper.  It
loads pre-generated embedding pairs, trains one or more adapters (depending
on the clustering configuration), evaluates retrieval quality, and saves a
structured results JSON.

Usage
-----
Run a global Procrustes baseline::

    python scripts/run_experiment.py \\
        --config configs/global_procrustes.yaml \\
        --pairs data/pairs/minilm-6-to-12_msmarco.npz

Run local drift-aware adapters with 16 clusters::

    python scripts/run_experiment.py \\
        --config configs/local_drift_aware.yaml \\
        --pairs data/pairs/minilm-6-to-12_msmarco.npz \\
        --override clustering.n_clusters=16

The output is a JSON file containing the full configuration, training
history, and evaluation metrics for the oracle, no-adapter, and adapted
settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# Ensure the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.adapters import LowRankAffine, OrthogonalProcrustes, ResidualMLP
from src.clustering import DriftAwareClustering, KMeansClustering, MoERouter
from src.config import load_config, parse_overrides
from src.data.pair_generator import EmbeddingPairGenerator
from src.evaluation import evaluate_adapter
from src.training import AdapterTrainer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------

def _split_data(
    X_old: np.ndarray,
    X_new: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split paired embeddings into train / val / test sets.

    The test set is further divided in half to create a *query* partition
    and a *corpus* partition.  The ground truth for retrieval evaluation is
    the identity mapping: query *i* should retrieve corpus *i* (they were
    embedded from the same source document).

    Returns
    -------
    dict
        Keys: ``X_old_train``, ``X_new_train``, ``X_old_val``,
        ``X_new_val``, ``X_old_test_q``, ``X_new_test_q``,
        ``X_old_test_c``, ``X_new_test_c``.
    """
    rng = np.random.default_rng(seed)
    n = X_old.shape[0]
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Split test into queries (first half) and corpus (second half)
    n_test_half = len(test_idx) // 2
    query_idx = test_idx[:n_test_half]
    corpus_idx = test_idx[n_test_half : 2 * n_test_half]

    return {
        "X_old_train": X_old[train_idx],
        "X_new_train": X_new[train_idx],
        "X_old_val": X_old[val_idx],
        "X_new_val": X_new[val_idx],
        "X_old_test_q": X_old[query_idx],
        "X_new_test_q": X_new[query_idx],
        "X_old_test_c": X_old[corpus_idx],
        "X_new_test_c": X_new[corpus_idx],
    }


# -----------------------------------------------------------------------
# Adapter construction
# -----------------------------------------------------------------------

def _build_adapter(cfg: dict, input_dim: int, output_dim: int) -> Any:
    """Instantiate the adapter module specified by *cfg['adapter']['type']*.

    Returns a Procrustes object (NumPy) or a PyTorch ``nn.Module``.
    """
    adapter_type = cfg["adapter"]["type"]

    if adapter_type == "procrustes":
        return OrthogonalProcrustes(fit_scaling=cfg["adapter"].get("fit_scaling", False))

    if adapter_type == "affine":
        rank = cfg["adapter"].get("rank", 32)
        return LowRankAffine(input_dim=input_dim, rank=rank, output_dim=output_dim)

    if adapter_type == "residual_mlp":
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dim=cfg["adapter"].get("hidden_dim", 256),
            num_layers=cfg["adapter"].get("num_layers", 2),
            dropout=cfg["adapter"].get("dropout", 0.1),
            output_dim=output_dim,
        )

    raise ValueError(f"Unknown adapter type: {adapter_type!r}")


def _train_adapter(
    cfg: dict,
    adapter: Any,
    X_old_train: np.ndarray,
    X_new_train: np.ndarray,
) -> dict:
    """Train *adapter* on the given data and return the training history.

    Procrustes adapters use a closed-form solution; neural adapters use
    the ``AdapterTrainer`` loop.
    """
    if isinstance(adapter, OrthogonalProcrustes):
        adapter.fit(X_old_train, X_new_train)
        return {"method": "closed_form"}

    # Neural adapter -- use AdapterTrainer
    trainer = AdapterTrainer(
        adapter=adapter,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        loss_fn=cfg["training"]["loss_fn"],
        device=cfg.get("device", "cpu"),
    )
    history = trainer.train(
        X_old_train,
        X_new_train,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        patience=cfg["training"]["patience"],
        verbose=True,
    )
    return history


def _make_adapter_fn(adapter: Any, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an adapter in a callable ``np.ndarray -> np.ndarray``."""
    if isinstance(adapter, OrthogonalProcrustes):
        return adapter.transform

    # PyTorch module
    adapter.eval()
    dev = torch.device(device)
    adapter.to(dev)

    def _fn(X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = adapter(X_t)
        return out.cpu().numpy()

    return _fn


# -----------------------------------------------------------------------
# Global (single-adapter) experiment
# -----------------------------------------------------------------------

def run_global_experiment(
    cfg: dict,
    splits: dict[str, np.ndarray],
) -> dict:
    """Train a single global adapter and evaluate it.

    Returns
    -------
    dict
        ``{"training": ..., "evaluation": ..., "timing": ...}``
    """
    input_dim = splits["X_old_train"].shape[1]
    output_dim = splits["X_new_train"].shape[1]

    adapter = _build_adapter(cfg, input_dim, output_dim)
    logger.info(
        "Training global %s adapter (input_dim=%d, output_dim=%d) ...",
        cfg["adapter"]["type"], input_dim, output_dim,
    )

    t0 = time.time()
    history = _train_adapter(cfg, adapter, splits["X_old_train"], splits["X_new_train"])
    train_time = time.time() - t0
    logger.info("Training completed in %.1f seconds.", train_time)

    adapter_fn = _make_adapter_fn(adapter, cfg.get("device", "cpu"))

    t0 = time.time()
    eval_results = evaluate_adapter(
        adapter_fn=adapter_fn,
        X_old_queries=splits["X_old_test_q"],
        X_new_queries=splits["X_new_test_q"],
        X_old_corpus=splits["X_old_test_c"],
        X_new_corpus=splits["X_new_test_c"],
        k_values=cfg["evaluation"]["k_values"],
    )
    eval_time = time.time() - t0

    return {
        "training": history,
        "evaluation": eval_results,
        "timing": {
            "train_seconds": round(train_time, 2),
            "eval_seconds": round(eval_time, 2),
        },
    }


# -----------------------------------------------------------------------
# Local (clustered MoE) experiment
# -----------------------------------------------------------------------

def run_local_experiment(
    cfg: dict,
    splits: dict[str, np.ndarray],
) -> dict:
    """Train per-cluster adapters with MoE routing and evaluate.

    Steps
    -----
    1. Cluster the training old-space embeddings.
    2. Train a separate adapter for each cluster's subset.
    3. At evaluation time, route each query through MoE and combine
       the per-adapter outputs as a weighted sum.

    Returns
    -------
    dict
        ``{"training": ..., "evaluation": ..., "clustering": ..., "timing": ...}``
    """
    n_clusters = cfg["clustering"]["n_clusters"]
    cluster_method = cfg["clustering"]["method"]
    input_dim = splits["X_old_train"].shape[1]
    output_dim = splits["X_new_train"].shape[1]

    # ------------------------------------------------------------------
    # 1. Clustering
    # ------------------------------------------------------------------
    logger.info(
        "Clustering training embeddings: method=%s, n_clusters=%d",
        cluster_method, n_clusters,
    )
    t0 = time.time()

    if cluster_method == "kmeans":
        clustering = KMeansClustering(n_clusters=n_clusters)
        clustering.fit(splits["X_old_train"])
        train_labels = clustering.predict(splits["X_old_train"])
        centroids = clustering.get_centroids()
    elif cluster_method == "drift_aware":
        clustering = DriftAwareClustering(
            n_clusters=n_clusters,
            drift_weight=cfg["clustering"].get("drift_weight", 0.5),
        )
        clustering.fit(splits["X_old_train"], splits["X_new_train"])
        train_labels = clustering.predict(
            splits["X_old_train"], splits["X_new_train"]
        )
        # For MoE routing we use centroids from the old embedding space.
        # Compute them from the training assignments.
        centroids = np.array([
            splits["X_old_train"][train_labels == k].mean(axis=0)
            for k in range(n_clusters)
        ])
    else:
        raise ValueError(f"Unknown clustering method: {cluster_method!r}")

    cluster_time = time.time() - t0
    logger.info("Clustering completed in %.1f seconds.", cluster_time)

    # Cluster size diagnostics
    cluster_sizes = {
        int(k): int((train_labels == k).sum()) for k in range(n_clusters)
    }
    logger.info("Cluster sizes: %s", cluster_sizes)

    # ------------------------------------------------------------------
    # 2. Per-cluster adapter training
    # ------------------------------------------------------------------
    adapters: list[Any] = []
    per_cluster_history: dict[int, dict] = {}

    t0 = time.time()
    for k in range(n_clusters):
        mask = train_labels == k
        n_k = int(mask.sum())
        if n_k == 0:
            logger.warning("Cluster %d is empty -- using identity adapter.", k)
            adapters.append(None)
            per_cluster_history[k] = {"method": "identity", "n_samples": 0}
            continue

        logger.info(
            "Training adapter for cluster %d/%d (%d samples) ...",
            k + 1, n_clusters, n_k,
        )
        adapter_k = _build_adapter(cfg, input_dim, output_dim)
        history_k = _train_adapter(
            cfg, adapter_k,
            splits["X_old_train"][mask],
            splits["X_new_train"][mask],
        )
        history_k["n_samples"] = n_k
        per_cluster_history[k] = history_k
        adapters.append(adapter_k)

    train_time = time.time() - t0
    logger.info("All per-cluster adapters trained in %.1f seconds.", train_time)

    # ------------------------------------------------------------------
    # 3. Build MoE routing adapter function
    # ------------------------------------------------------------------
    router = MoERouter(
        centroids=centroids,
        temperature=cfg["routing"].get("temperature", 1.0),
        routing=cfg["routing"].get("method", "soft"),
    )

    adapter_fns = [
        _make_adapter_fn(a, cfg.get("device", "cpu")) if a is not None else None
        for a in adapters
    ]

    def moe_adapter_fn(X_old: np.ndarray) -> np.ndarray:
        """Apply the mixture-of-experts adapter to a batch of old embeddings.

        For each input vector, the router produces per-cluster weights
        (soft or hard), each cluster's adapter transforms the input, and
        the final output is the weighted combination:

            adapted_i = sum_k w_{i,k} * f_k(x_i)
        """
        weights = router.route(X_old)  # (n, n_clusters)
        n = X_old.shape[0]

        # Pre-compute all adapter outputs
        cluster_outputs = np.zeros((n_clusters, n, output_dim), dtype=np.float32)
        for k in range(n_clusters):
            if adapter_fns[k] is not None:
                cluster_outputs[k] = adapter_fns[k](X_old)
            else:
                # Identity fallback: if dims match, pass through; else zero
                if input_dim == output_dim:
                    cluster_outputs[k] = X_old
                else:
                    cluster_outputs[k] = np.zeros((n, output_dim), dtype=np.float32)

        # Weighted combination: adapted[i, :] = sum_k weights[i, k] * cluster_outputs[k, i, :]
        adapted = np.einsum("nk,knd->nd", weights, cluster_outputs)
        return adapted

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    t0 = time.time()
    eval_results = evaluate_adapter(
        adapter_fn=moe_adapter_fn,
        X_old_queries=splits["X_old_test_q"],
        X_new_queries=splits["X_new_test_q"],
        X_old_corpus=splits["X_old_test_c"],
        X_new_corpus=splits["X_new_test_c"],
        k_values=cfg["evaluation"]["k_values"],
    )
    eval_time = time.time() - t0

    return {
        "training": per_cluster_history,
        "evaluation": eval_results,
        "clustering": {
            "method": cluster_method,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
        },
        "timing": {
            "cluster_seconds": round(cluster_time, 2),
            "train_seconds": round(train_time, 2),
            "eval_seconds": round(eval_time, 2),
        },
    }


# -----------------------------------------------------------------------
# Results formatting
# -----------------------------------------------------------------------

def _print_results_table(results: dict) -> None:
    """Print a concise results table to stdout."""
    eval_data = results["evaluation"]
    settings = ["oracle", "no_adapter", "adapted"]
    all_keys = list(eval_data["oracle"].keys())

    # Determine column widths
    header = ["Setting"] + all_keys
    col_widths = [max(len(h), 12) for h in header]
    col_widths[0] = max(col_widths[0], 14)

    # Header
    header_str = "  ".join(h.rjust(w) for h, w in zip(header, col_widths))
    print("\n" + header_str)
    print("  ".join("-" * w for w in col_widths))

    # Rows
    for setting in settings:
        row = [setting]
        for key in all_keys:
            val = eval_data[setting].get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        print("  ".join(v.rjust(w) for v, w in zip(row, col_widths)))

    print()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a drift adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to experiment YAML config, merged on top of base.yaml. "
            "Can also be a bare name like 'global_procrustes'."
        ),
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pre-generated embedding pairs (.npz file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results JSON (default: from config 'output.dir').",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides as dotted key=value pairs (e.g. clustering.n_clusters=16).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Public entry point (for programmatic invocation from run_all_experiments)
# -----------------------------------------------------------------------

def run_experiment(
    config_path: str | None,
    pairs_path: str,
    output_dir: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Run a single experiment and return the results dict.

    This function is the main entry point used both by the CLI and by
    ``run_all_experiments.py`` for programmatic invocation.

    Parameters
    ----------
    config_path : str or None
        Path to the experiment YAML config.
    pairs_path : str
        Path to the pre-generated .npz embedding pairs.
    output_dir : str or None
        Directory for results JSON.  If None, uses the config default.
    overrides : list of str or None
        CLI-style overrides (e.g. ``["clustering.n_clusters=16"]``).

    Returns
    -------
    dict
        Complete results including config, training history, and
        evaluation metrics.
    """
    # ----- Load config -----
    # Normalise overrides: ensure each has the '--' prefix that parse_overrides expects.
    # CLI --override args arrive bare (e.g. "clustering.n_clusters=16"), while
    # programmatic callers may include the prefix.
    normalised = []
    for o in (overrides or []):
        if not o.startswith("--"):
            o = "--" + o
        normalised.append(o)
    override_dict = parse_overrides(normalised)
    cfg = load_config(experiment_config=config_path, overrides=override_dict)

    # ----- Set random seeds -----
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Load embedding pairs -----
    logger.info("Loading embedding pairs from %s ...", pairs_path)
    X_old, X_new = EmbeddingPairGenerator.load_pairs(pairs_path)
    logger.info(
        "Loaded %d pairs: X_old %s, X_new %s",
        X_old.shape[0], X_old.shape, X_new.shape,
    )

    # Handle dimension mismatches for Procrustes
    if cfg["adapter"]["type"] == "procrustes" and X_old.shape[1] != X_new.shape[1]:
        logger.info(
            "Padding embeddings to match dimensions for Procrustes "
            "(old=%d, new=%d).", X_old.shape[1], X_new.shape[1],
        )
        X_old, X_new = EmbeddingPairGenerator.pad_to_match(X_old, X_new)

    # ----- Train/val/test split -----
    splits = _split_data(
        X_old, X_new,
        train_frac=cfg["data"]["train_split"],
        val_frac=cfg["data"]["val_split"],
        seed=seed,
    )
    logger.info(
        "Split sizes: train=%d, val=%d, test_queries=%d, test_corpus=%d",
        splits["X_old_train"].shape[0],
        splits["X_old_val"].shape[0],
        splits["X_old_test_q"].shape[0],
        splits["X_old_test_c"].shape[0],
    )

    # ----- Run experiment -----
    cluster_method = cfg["clustering"]["method"]
    if cluster_method == "none":
        logger.info("Running GLOBAL experiment (adapter=%s).", cfg["adapter"]["type"])
        results = run_global_experiment(cfg, splits)
    else:
        logger.info(
            "Running LOCAL experiment (adapter=%s, clustering=%s, k=%d).",
            cfg["adapter"]["type"], cluster_method, cfg["clustering"]["n_clusters"],
        )
        results = run_local_experiment(cfg, splits)

    # ----- Attach metadata -----
    results["config"] = cfg
    results["pairs_path"] = str(pairs_path)
    results["n_samples"] = int(X_old.shape[0])

    # ----- Save results -----
    out_dir = Path(output_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename
    adapter_name = cfg["adapter"]["type"]
    if cluster_method != "none":
        exp_name = f"local_{cluster_method}_{adapter_name}_k{cfg['clustering']['n_clusters']}"
    else:
        exp_name = f"global_{adapter_name}"

    # Include the model pair and dataset from the pairs filename
    pairs_stem = Path(pairs_path).stem
    result_name = f"{pairs_stem}_{exp_name}"
    result_path = out_dir / f"{result_name}.json"

    # Serialise -- convert numpy types to native Python for JSON
    with open(result_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    logger.info("Results saved to %s", result_path)

    # ----- Print summary table -----
    _print_results_table(results)

    return results


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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

    run_experiment(
        config_path=args.config,
        pairs_path=args.pairs,
        output_dir=args.output_dir,
        overrides=args.override,
    )


if __name__ == "__main__":
    main()
