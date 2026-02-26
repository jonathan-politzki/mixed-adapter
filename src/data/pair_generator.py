"""Generate paired (old-model, new-model) embeddings for adapter training.

The central abstraction is :class:`EmbeddingPairGenerator` which wraps
two ``sentence-transformers`` models and produces aligned numpy arrays
``(X_old, X_new)`` for the same input texts.  These pairs are the
training signal for all drift-adapter variants (Procrustes, affine,
residual MLP, and their mixture-of-experts counterparts).

A registry of common model upgrade pairs is provided in :data:`MODEL_PAIRS`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model upgrade pairs used throughout the paper experiments
# ---------------------------------------------------------------------------

MODEL_PAIRS: dict[str, tuple[str, str]] = {
    "bge-small-to-base": (
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ),
    "minilm-to-bge": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ),
    "minilm-6-to-12": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ),
    "minilm-to-e5-large": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-large-v2",
    ),
}
"""Mapping from friendly pair name to ``(old_model_name, new_model_name)``.

* **bge-small-to-base** -- same family, dimension increase (384 -> 768).
* **minilm-to-bge** -- cross-family, same dimension (384 -> 384).
* **minilm-6-to-12** -- same family, same dimension (384 -> 384).
* **minilm-to-e5-large** -- cross-family, large dimension gap (384 -> 1024).
"""


# ---------------------------------------------------------------------------
# Helper: discover the embedding dimension of a model
# ---------------------------------------------------------------------------

def get_model_dims(model_name: str) -> int:
    """Return the embedding dimension of a sentence-transformer model.

    The model is loaded once, a dummy sentence is encoded, and the
    vector length is returned.  This is useful for programmatically
    choosing adapter architectures (e.g. ``input_dim`` /
    ``output_dim``) without hard-coding dimensions.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``.

    Returns
    -------
    int
        Embedding dimension.
    """
    model = SentenceTransformer(model_name)
    dim: int = model.get_sentence_embedding_dimension()  # type: ignore[assignment]
    return dim


# ---------------------------------------------------------------------------
# Core pair generator
# ---------------------------------------------------------------------------

class EmbeddingPairGenerator:
    """Encode texts with two sentence-transformer models and return aligned pairs.

    Parameters
    ----------
    old_model_name : str
        HuggingFace identifier for the *old* (source) embedding model.
    new_model_name : str
        HuggingFace identifier for the *new* (target) embedding model.
    batch_size : int
        Number of texts to encode per forward pass.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``, ...).

    Examples
    --------
    >>> gen = EmbeddingPairGenerator(
    ...     "sentence-transformers/all-MiniLM-L6-v2",
    ...     "sentence-transformers/all-MiniLM-L12-v2",
    ... )
    >>> X_old, X_new = gen.generate_pairs(["hello world", "foo bar"])
    >>> X_old.shape[0] == X_new.shape[0] == 2
    True
    """

    def __init__(
        self,
        old_model_name: str,
        new_model_name: str,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.old_model_name = old_model_name
        self.new_model_name = new_model_name
        self.batch_size = batch_size
        self.device = device

        logger.info("Loading old model: %s", old_model_name)
        self.old_model = SentenceTransformer(old_model_name, device=device)

        logger.info("Loading new model: %s", new_model_name)
        self.new_model = SentenceTransformer(new_model_name, device=device)

        self.old_dim: int = self.old_model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        self.new_dim: int = self.new_model.get_sentence_embedding_dimension()  # type: ignore[assignment]

        if self.old_dim != self.new_dim:
            logger.warning(
                "Dimension mismatch: old model = %d, new model = %d. "
                "All adapters support different dimensions: neural adapters "
                "(LowRankAffine, ResidualMLP) accept separate input_dim / "
                "output_dim constructor args; OrthogonalProcrustes requires "
                "equal dimensions and can use pad_to_match().",
                self.old_dim,
                self.new_dim,
            )

    # --------------------------------------------------------------------- #
    # Encoding
    # --------------------------------------------------------------------- #

    def generate_pairs(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Embed *texts* with both models and return aligned arrays.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        show_progress : bool
            Whether to show a ``tqdm`` progress bar.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X_old, X_new)`` where ``X_old`` has shape
            ``(len(texts), old_dim)`` and ``X_new`` has shape
            ``(len(texts), new_dim)``.
        """
        n = len(texts)
        X_old = np.empty((n, self.old_dim), dtype=np.float32)
        X_new = np.empty((n, self.new_dim), dtype=np.float32)

        batch_starts = list(range(0, n, self.batch_size))
        iterator = tqdm(batch_starts, desc="Encoding pairs", disable=not show_progress)

        for start in iterator:
            end = min(start + self.batch_size, n)
            batch = texts[start:end]

            X_old[start:end] = self.old_model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            X_new[start:end] = self.new_model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        logger.info(
            "Generated %d embedding pairs: old (%d-d) | new (%d-d).",
            n,
            self.old_dim,
            self.new_dim,
        )
        return X_old, X_new

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #

    def save_pairs(
        self,
        X_old: NDArray[np.floating],
        X_new: NDArray[np.floating],
        path: Union[str, Path],
    ) -> None:
        """Save an embedding pair to a compressed ``.npz`` archive.

        The file stores three entries:

        * ``X_old`` -- old-model embeddings.
        * ``X_new`` -- new-model embeddings.
        * ``metadata`` -- a small array encoding
          ``[old_dim, new_dim, n_samples]`` for quick inspection.

        Parameters
        ----------
        X_old : np.ndarray
            Old-model embeddings of shape ``(n, d_old)``.
        X_new : np.ndarray
            New-model embeddings of shape ``(n, d_new)``.
        path : str or Path
            Destination file path (the ``.npz`` suffix is added
            automatically if absent).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = np.array([self.old_dim, self.new_dim, len(X_old)], dtype=np.int64)
        np.savez_compressed(path, X_old=X_old, X_new=X_new, metadata=metadata)
        logger.info("Saved embedding pairs to %s", path)

    @staticmethod
    def load_pairs(path: Union[str, Path]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Load an embedding pair from a ``.npz`` archive.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` file previously created by
            :meth:`save_pairs`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X_old, X_new)`` numpy arrays.
        """
        data = np.load(path)
        X_old: NDArray[np.floating] = data["X_old"]
        X_new: NDArray[np.floating] = data["X_new"]
        logger.info(
            "Loaded embedding pairs from %s: X_old %s, X_new %s.",
            path,
            X_old.shape,
            X_new.shape,
        )
        return X_old, X_new

    # --------------------------------------------------------------------- #
    # Dimension-mismatch utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def pad_to_match(
        X_old: NDArray[np.floating],
        X_new: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Zero-pad the smaller array so both have the same embedding dimension.

        All adapters now support different input/output dimensions:

        * **Neural adapters**
          (:class:`~src.adapters.affine.LowRankAffine`,
          :class:`~src.adapters.residual_mlp.ResidualMLP`) accept
          separate ``input_dim`` and ``output_dim`` constructor
          arguments and handle dimension changes internally.  They
          do **not** need this utility.

        * **OrthogonalProcrustes** requires equal dimensions, so
          ``pad_to_match`` is primarily useful for that adapter.

        Use this method when you need to feed identically-shaped
        matrices to algorithms that cannot handle a dimension
        mismatch (e.g. Procrustes, or drift-aware clustering).

        Parameters
        ----------
        X_old : np.ndarray
            Old-model embeddings of shape ``(n, d_old)``.
        X_new : np.ndarray
            New-model embeddings of shape ``(n, d_new)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X_old_padded, X_new_padded)`` both of shape
            ``(n, max(d_old, d_new))``.
        """
        d_old = X_old.shape[1]
        d_new = X_new.shape[1]

        if d_old == d_new:
            return X_old, X_new

        d_max = max(d_old, d_new)
        n = X_old.shape[0]

        if d_old < d_max:
            pad = np.zeros((n, d_max - d_old), dtype=X_old.dtype)
            X_old = np.concatenate([X_old, pad], axis=1)
            logger.info("Zero-padded X_old from %d to %d dimensions.", d_old, d_max)

        if d_new < d_max:
            pad = np.zeros((n, d_max - d_new), dtype=X_new.dtype)
            X_new = np.concatenate([X_new, pad], axis=1)
            logger.info("Zero-padded X_new from %d to %d dimensions.", d_new, d_max)

        return X_old, X_new

    # --------------------------------------------------------------------- #
    # Repr
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"EmbeddingPairGenerator("
            f"old='{self.old_model_name}' ({self.old_dim}-d), "
            f"new='{self.new_model_name}' ({self.new_dim}-d), "
            f"batch_size={self.batch_size}, device='{self.device}')"
        )
