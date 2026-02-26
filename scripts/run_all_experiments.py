#!/usr/bin/env python
"""Orchestrate all experiments for the local-drift-adapter paper.

This is the one-command reproducer.  It generates embedding pairs (if not
already cached), runs every experimental configuration, and produces a
consolidated summary table.

Usage
-----
Run everything with defaults::

    python scripts/run_all_experiments.py

Specify custom directories::

    python scripts/run_all_experiments.py \\
        --data-dir data/pairs/ \\
        --output-dir experiments/results/

Skip pair generation (pairs already exist)::

    python scripts/run_all_experiments.py --skip-generate

The final summary is saved as ``summary.json`` and printed as a formatted
table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import MODEL_PAIRS

# Import the runner directly for cleaner invocation.
from scripts.run_experiment import run_experiment
from scripts.generate_pairs import main as generate_pairs_main

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------

# Global baselines -- one per adapter type
GLOBAL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "global_procrustes",
        "config": "configs/global_procrustes.yaml",
        "overrides": [],
    },
    {
        "name": "global_affine",
        "config": "configs/global_affine.yaml",
        "overrides": [],
    },
    {
        "name": "global_mlp",
        "config": "configs/global_mlp.yaml",
        "overrides": [],
    },
]

# Local experiments -- different clustering strategies at k=8
# Affine adapters (main comparison — can capture per-region drift)
LOCAL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "local_kmeans_affine_k8",
        "config": "configs/local_kmeans.yaml",
        "overrides": ["--clustering.n_clusters=8"],
    },
    {
        "name": "local_drift_aware_affine_k8",
        "config": "configs/local_drift_aware.yaml",
        "overrides": ["--clustering.n_clusters=8"],
    },
    # Procrustes local for ablation (shows clustering alone isn't enough)
    {
        "name": "local_kmeans_procrustes_k8",
        "config": "configs/local_kmeans.yaml",
        "overrides": ["--clustering.n_clusters=8", "--adapter.type=procrustes"],
    },
    {
        "name": "local_drift_aware_procrustes_k8",
        "config": "configs/local_drift_aware.yaml",
        "overrides": ["--clustering.n_clusters=8", "--adapter.type=procrustes"],
    },
]

# Cluster sweep -- drift-aware with varying k
SWEEP_K_VALUES = [2, 4, 8, 16, 32, 64]


# -----------------------------------------------------------------------
# Pair generation helper
# -----------------------------------------------------------------------

def _ensure_pairs_exist(
    model_pair_name: str,
    dataset: str,
    data_dir: Path,
    max_samples: int = 100_000,
    device: str = "cpu",
) -> Path:
    """Check if pairs exist; generate them if not.

    Returns the path to the ``.npz`` file.
    """
    stem = f"{model_pair_name}_{dataset}"
    npz_path = data_dir / f"{stem}.npz"

    if npz_path.exists():
        logger.info("Pairs already exist: %s", npz_path)
        return npz_path

    logger.info(
        "Generating pairs for %s on %s (%d samples) ...",
        model_pair_name, dataset, max_samples,
    )

    # Build the argv list that generate_pairs.py expects
    argv = [
        "--model-pair", model_pair_name,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--output", str(data_dir),
        "--device", device,
    ]

    # Temporarily replace sys.argv and call the generator
    original_argv = sys.argv
    sys.argv = ["generate_pairs.py"] + argv
    try:
        generate_pairs_main()
    finally:
        sys.argv = original_argv

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Pair generation did not produce expected file: {npz_path}"
        )
    return npz_path


# -----------------------------------------------------------------------
# Summary formatting
# -----------------------------------------------------------------------

def _format_summary_table(
    summary: list[dict[str, Any]],
    metric_keys: list[str] | None = None,
) -> str:
    """Format the summary list as a human-readable ASCII table.

    Parameters
    ----------
    summary : list of dict
        Each entry has ``"experiment"``, ``"model_pair"``, and nested
        ``"adapted"`` metrics.
    metric_keys : list of str, optional
        Which metric keys to include.  Defaults to a sensible selection.

    Returns
    -------
    str
        Formatted table string.
    """
    if not summary:
        return "(no results)"

    if metric_keys is None:
        # Auto-detect from first result
        sample_metrics = summary[0].get("adapted", {})
        metric_keys = sorted(sample_metrics.keys())

    headers = ["Experiment", "Model Pair"] + metric_keys
    col_widths = [max(len(h), 12) for h in headers]
    col_widths[0] = max(col_widths[0], 30)
    col_widths[1] = max(col_widths[1], 18)

    lines: list[str] = []

    # Header
    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("  ".join("-" * w for w in col_widths))

    # Rows
    for entry in summary:
        row = [entry["experiment"], entry["model_pair"]]
        adapted = entry.get("adapted", {})
        for key in metric_keys:
            val = adapted.get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append("  ".join(v.rjust(w) for v, w in zip(row, col_widths)))

    return "\n".join(lines)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiments for the local-drift-adapter paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pairs/",
        help="Directory for embedding pair files (default: data/pairs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/",
        help="Directory for results JSON files (default: experiments/results/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco",
        help="Dataset to use for pair generation (default: msmarco).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum samples for pair generation (default: 100000).",
    )
    parser.add_argument(
        "--model-pairs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Model pairs to run (default: all in MODEL_PAIRS). "
            "Choices: " + ", ".join(MODEL_PAIRS.keys())
        ),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip pair generation (assume pairs already exist).",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the cluster-count sweep experiments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
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

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which model pairs to run
    pair_names = args.model_pairs if args.model_pairs else list(MODEL_PAIRS.keys())
    for name in pair_names:
        if name not in MODEL_PAIRS:
            logger.error("Unknown model pair: %s", name)
            sys.exit(1)

    total_t0 = time.time()
    all_summaries: list[dict[str, Any]] = []
    failed_experiments: list[dict[str, str]] = []

    for pair_name in pair_names:
        logger.info("=" * 72)
        logger.info("MODEL PAIR: %s", pair_name)
        logger.info("=" * 72)

        # ---- Ensure embedding pairs exist ----
        if not args.skip_generate:
            try:
                pairs_path = _ensure_pairs_exist(
                    pair_name, args.dataset, data_dir,
                    max_samples=args.max_samples,
                    device=args.device,
                )
            except Exception:
                logger.exception(
                    "Failed to generate pairs for %s. Skipping.", pair_name
                )
                continue
        else:
            pairs_path = data_dir / f"{pair_name}_{args.dataset}.npz"
            if not pairs_path.exists():
                logger.error(
                    "Pairs file not found: %s (use without --skip-generate "
                    "to auto-create)", pairs_path,
                )
                continue

        # ---- Global baselines ----
        for exp in GLOBAL_CONFIGS:
            exp_name = f"{pair_name}/{exp['name']}"
            logger.info("-" * 60)
            logger.info("EXPERIMENT: %s", exp_name)
            logger.info("-" * 60)

            try:
                results = run_experiment(
                    config_path=exp["config"],
                    pairs_path=str(pairs_path),
                    output_dir=str(output_dir),
                    overrides=exp["overrides"] + [f"--device={args.device}"],
                )
                all_summaries.append({
                    "experiment": exp["name"],
                    "model_pair": pair_name,
                    "adapted": results["evaluation"]["adapted"],
                    "oracle": results["evaluation"]["oracle"],
                    "no_adapter": results["evaluation"]["no_adapter"],
                    "timing": results.get("timing", {}),
                })
            except Exception:
                logger.exception("FAILED: %s", exp_name)
                failed_experiments.append({"experiment": exp_name, "error": "see logs"})

        # ---- Local experiments (k=8) ----
        for exp in LOCAL_CONFIGS:
            exp_name = f"{pair_name}/{exp['name']}"
            logger.info("-" * 60)
            logger.info("EXPERIMENT: %s", exp_name)
            logger.info("-" * 60)

            # Run with each adapter type (use affine as default for local)
            try:
                results = run_experiment(
                    config_path=exp["config"],
                    pairs_path=str(pairs_path),
                    output_dir=str(output_dir),
                    overrides=exp["overrides"] + [f"--device={args.device}"],
                )
                all_summaries.append({
                    "experiment": exp["name"],
                    "model_pair": pair_name,
                    "adapted": results["evaluation"]["adapted"],
                    "oracle": results["evaluation"]["oracle"],
                    "no_adapter": results["evaluation"]["no_adapter"],
                    "clustering": results.get("clustering", {}),
                    "timing": results.get("timing", {}),
                })
            except Exception:
                logger.exception("FAILED: %s", exp_name)
                failed_experiments.append({"experiment": exp_name, "error": "see logs"})

        # ---- Cluster sweep ----
        if not args.skip_sweep:
            for k in SWEEP_K_VALUES:
                exp_name = f"{pair_name}/sweep_drift_aware_k{k}"
                logger.info("-" * 60)
                logger.info("EXPERIMENT: %s", exp_name)
                logger.info("-" * 60)

                try:
                    results = run_experiment(
                        config_path="configs/local_drift_aware.yaml",
                        pairs_path=str(pairs_path),
                        output_dir=str(output_dir),
                        overrides=[
                            f"--clustering.n_clusters={k}",
                            f"--device={args.device}",
                        ],
                    )
                    all_summaries.append({
                        "experiment": f"sweep_drift_aware_k{k}",
                        "model_pair": pair_name,
                        "adapted": results["evaluation"]["adapted"],
                        "oracle": results["evaluation"]["oracle"],
                        "no_adapter": results["evaluation"]["no_adapter"],
                        "clustering": results.get("clustering", {}),
                        "timing": results.get("timing", {}),
                    })
                except Exception:
                    logger.exception("FAILED: %s", exp_name)
                    failed_experiments.append({"experiment": exp_name, "error": "see logs"})

    total_time = time.time() - total_t0

    # ------------------------------------------------------------------
    # Save and print summary
    # ------------------------------------------------------------------
    summary_output = {
        "experiments": all_summaries,
        "failed": failed_experiments,
        "total_time_seconds": round(total_time, 2),
        "n_experiments": len(all_summaries),
        "n_failed": len(failed_experiments),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary_output, fh, indent=2, default=_json_default)
    logger.info("Summary saved to %s", summary_path)

    # Print formatted table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(_format_summary_table(all_summaries))
    print()

    if failed_experiments:
        print(f"WARNING: {len(failed_experiments)} experiment(s) failed:")
        for f in failed_experiments:
            print(f"  - {f['experiment']}")
        print()

    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print()


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
