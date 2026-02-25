#!/usr/bin/env python
"""Generate paired (old-model, new-model) embeddings for adapter training.

This script embeds a text corpus with two sentence-transformer models and
saves the aligned embedding matrices as a compressed ``.npz`` file.  The
resulting pairs are consumed by ``run_experiment.py`` for adapter training
and evaluation.

Usage
-----
Generate pairs for the MiniLM-6 -> MiniLM-12 upgrade on MS MARCO::

    python scripts/generate_pairs.py \\
        --model-pair minilm-6-to-12 \\
        --dataset msmarco \\
        --max-samples 100000 \\
        --output data/pairs/

Generate pairs for a custom model pair::

    python scripts/generate_pairs.py \\
        --old-model sentence-transformers/all-MiniLM-L6-v2 \\
        --new-model sentence-transformers/all-MiniLM-L12-v2 \\
        --dataset nq \\
        --output data/pairs/

The output filename is derived from the model-pair name and dataset::

    data/pairs/<model-pair>_<dataset>.npz
    data/pairs/<model-pair>_<dataset>_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so ``src.*`` imports work when
# the script is invoked directly (e.g. ``python scripts/generate_pairs.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import (
    MODEL_PAIRS,
    EmbeddingPairGenerator,
    load_beir_dataset,
    load_heterogeneous_corpus,
    load_msmarco,
    load_nq,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Dataset loading dispatch
# -----------------------------------------------------------------------

def _load_texts(dataset: str, max_samples: int, beir_name: str | None = None) -> list[str]:
    """Load a list of text strings from the requested dataset.

    Parameters
    ----------
    dataset : str
        One of ``"msmarco"``, ``"nq"``, ``"beir"``, or ``"heterogeneous"``.
    max_samples : int
        Maximum number of texts to load.
    beir_name : str or None
        Required when *dataset* is ``"beir"`` -- the BEIR dataset name
        (e.g. ``"scifact"``).

    Returns
    -------
    list[str]
        Loaded text strings.
    """
    if dataset == "msmarco":
        return load_msmarco(max_samples=max_samples)
    if dataset == "nq":
        return load_nq(max_samples=max_samples)
    if dataset == "beir":
        if beir_name is None:
            raise ValueError("--beir-name is required when --dataset=beir")
        data = load_beir_dataset(beir_name, max_samples=max_samples)
        # Combine corpus and queries for pair generation
        return data["corpus"] + data["queries"]
    if dataset == "heterogeneous":
        texts, _labels = load_heterogeneous_corpus(max_per_domain=max_samples // 3)
        return texts
    raise ValueError(f"Unknown dataset: {dataset!r}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired embeddings for adapter training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection -- either a named pair or explicit model names
    model_group = parser.add_argument_group("Model selection")
    model_group.add_argument(
        "--model-pair",
        choices=list(MODEL_PAIRS.keys()),
        default=None,
        help="Named model-upgrade pair (see src.data.pair_generator.MODEL_PAIRS).",
    )
    model_group.add_argument(
        "--old-model",
        default=None,
        help="HuggingFace identifier of the old (source) model.",
    )
    model_group.add_argument(
        "--new-model",
        default=None,
        help="HuggingFace identifier of the new (target) model.",
    )

    # Dataset selection
    data_group = parser.add_argument_group("Dataset selection")
    data_group.add_argument(
        "--dataset",
        choices=["msmarco", "nq", "beir", "heterogeneous"],
        default="msmarco",
        help="Text corpus to embed (default: msmarco).",
    )
    data_group.add_argument(
        "--beir-name",
        default=None,
        help="BEIR dataset name (only used when --dataset=beir).",
    )
    data_group.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum number of texts to embed (default: 100000).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="data/pairs/",
        help="Output directory for the .npz and metadata files.",
    )

    # Runtime
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for encoding (default: cpu).",
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

    # ----- Resolve model names -----
    if args.model_pair is not None:
        old_model_name, new_model_name = MODEL_PAIRS[args.model_pair]
        pair_name = args.model_pair
    elif args.old_model is not None and args.new_model is not None:
        old_model_name = args.old_model
        new_model_name = args.new_model
        # Derive a short name from the model identifiers
        pair_name = (
            old_model_name.split("/")[-1] + "_to_" + new_model_name.split("/")[-1]
        )
    else:
        logger.error(
            "Specify either --model-pair or both --old-model and --new-model."
        )
        sys.exit(1)

    # ----- Load texts -----
    logger.info(
        "Loading dataset=%s (max_samples=%d) ...", args.dataset, args.max_samples
    )
    texts = _load_texts(args.dataset, args.max_samples, args.beir_name)
    logger.info("Loaded %d texts.", len(texts))

    # ----- Generate embedding pairs -----
    generator = EmbeddingPairGenerator(
        old_model_name=old_model_name,
        new_model_name=new_model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    t0 = time.time()
    X_old, X_new = generator.generate_pairs(texts, show_progress=True)
    elapsed = time.time() - t0
    logger.info("Encoding completed in %.1f seconds.", elapsed)

    # ----- Save .npz -----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{pair_name}_{args.dataset}"
    npz_path = output_dir / f"{stem}.npz"
    generator.save_pairs(X_old, X_new, npz_path)
    logger.info("Saved embedding pairs to %s", npz_path)

    # ----- Save metadata JSON -----
    metadata = {
        "model_pair_name": pair_name,
        "old_model": old_model_name,
        "new_model": new_model_name,
        "old_dim": int(X_old.shape[1]),
        "new_dim": int(X_new.shape[1]),
        "n_samples": int(X_old.shape[0]),
        "dataset": args.dataset,
        "beir_name": args.beir_name,
        "device": args.device,
        "encoding_time_s": round(elapsed, 2),
    }

    meta_path = output_dir / f"{stem}_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # ----- Summary -----
    logger.info(
        "Done. X_old: %s (%d-d), X_new: %s (%d-d)",
        X_old.shape,
        X_old.shape[1],
        X_new.shape,
        X_new.shape[1],
    )


if __name__ == "__main__":
    main()
