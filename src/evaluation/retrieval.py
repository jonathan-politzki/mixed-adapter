"""Retrieval evaluation metrics for embedding adapter quality.

All functions expect dense numpy arrays and use FAISS for nearest-neighbour
search so that evaluation scales to million-point corpora.
"""

from __future__ import annotations

from typing import Callable, Sequence

import faiss
import numpy as np
from sklearn.metrics import ndcg_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index over L2-normalised embeddings.

    Using inner product on L2-normalised vectors is equivalent to cosine
    similarity, which is standard for dense retrieval evaluation.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _search(
    index: faiss.IndexFlatIP,
    queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search *index* for the *k* nearest neighbours of each query.

    Returns ``(distances, indices)`` arrays of shape ``(n_queries, k)``.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    faiss.normalize_L2(queries)
    return index.search(queries, k)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    ground_truth_indices: np.ndarray,
    k_values: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Recall@k -- fraction of queries whose ground-truth doc is in top-k.

    Parameters
    ----------
    query_embeddings:
        Query vectors of shape ``(n_queries, d)``.
    corpus_embeddings:
        Corpus vectors of shape ``(n_corpus, d)``.
    ground_truth_indices:
        Integer array of shape ``(n_queries,)`` giving the index into
        *corpus_embeddings* of the single relevant document per query.
    k_values:
        Values of *k* at which to evaluate recall.

    Returns
    -------
    Dictionary mapping ``"recall@{k}"`` to the recall value.
    """
    max_k = max(k_values)
    index = _build_flat_index(corpus_embeddings.copy())
    _, nn_indices = _search(index, query_embeddings.copy(), max_k)

    gt = np.asarray(ground_truth_indices).reshape(-1, 1)
    results: dict[str, float] = {}
    for k in sorted(k_values):
        hits = np.any(nn_indices[:, :k] == gt, axis=1)
        results[f"recall@{k}"] = float(hits.mean())
    return results


def compute_ndcg_at_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    relevance_scores: np.ndarray,
    k: int = 10,
) -> float:
    """NDCG@k using sklearn's ``ndcg_score``.

    Parameters
    ----------
    query_embeddings:
        Query vectors of shape ``(n_queries, d)``.
    corpus_embeddings:
        Corpus vectors of shape ``(n_corpus, d)``.
    relevance_scores:
        Dense relevance matrix of shape ``(n_queries, n_corpus)`` with
        non-negative relevance grades.
    k:
        Truncation depth.

    Returns
    -------
    Mean NDCG@k across all queries.
    """
    index = _build_flat_index(corpus_embeddings.copy())
    distances, _ = _search(index, query_embeddings.copy(), corpus_embeddings.shape[0])
    # distances are cosine similarities (higher = more similar)
    return float(ndcg_score(relevance_scores, distances, k=k))


def compute_cosine_similarity(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """Mean pairwise cosine similarity between predicted and target embeddings.

    Parameters
    ----------
    predicted:
        Predicted embeddings of shape ``(n, d)``.
    target:
        Target embeddings of shape ``(n, d)``.

    Returns
    -------
    Mean cosine similarity (scalar in ``[-1, 1]``).
    """
    pred_norm = predicted / (np.linalg.norm(predicted, axis=1, keepdims=True) + 1e-12)
    tgt_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(pred_norm * tgt_norm, axis=1)))


def compute_mrr(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    ground_truth_indices: np.ndarray,
) -> float:
    """Mean Reciprocal Rank.

    Parameters
    ----------
    query_embeddings:
        Query vectors of shape ``(n_queries, d)``.
    corpus_embeddings:
        Corpus vectors of shape ``(n_corpus, d)``.
    ground_truth_indices:
        Integer array of shape ``(n_queries,)`` giving the relevant
        corpus index for each query.

    Returns
    -------
    MRR (scalar in ``(0, 1]``).
    """
    n_corpus = corpus_embeddings.shape[0]
    index = _build_flat_index(corpus_embeddings.copy())
    _, nn_indices = _search(index, query_embeddings.copy(), n_corpus)

    gt = np.asarray(ground_truth_indices)
    reciprocal_ranks: list[float] = []
    for i, gt_idx in enumerate(gt):
        positions = np.where(nn_indices[i] == gt_idx)[0]
        if len(positions) > 0:
            reciprocal_ranks.append(1.0 / (positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


# ---------------------------------------------------------------------------
# End-to-end adapter evaluation
# ---------------------------------------------------------------------------

def _pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad *X* along the feature axis to reach *target_dim* columns.

    If *X* already has *target_dim* columns the input is returned unchanged.
    """
    d = X.shape[1]
    if d >= target_dim:
        return X
    pad = np.zeros((X.shape[0], target_dim - d), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def evaluate_adapter(
    adapter_fn: Callable[[np.ndarray], np.ndarray],
    X_old_queries: np.ndarray,
    X_new_queries: np.ndarray,
    X_old_corpus: np.ndarray,
    X_new_corpus: np.ndarray,
    k_values: Sequence[int] = (1, 5, 10, 100),
) -> dict:
    """Evaluate an adapter's retrieval quality against oracle and no-adapter baselines.

    The evaluation protocol:

    1. **Oracle** -- ``X_new_queries`` vs ``X_new_corpus``.  Upper bound on
       retrieval quality; what you get if you re-embed everything.
    2. **No-adapter** -- ``X_old_queries`` vs ``X_new_corpus``.  Lower bound;
       the degradation you suffer if old queries hit a new corpus.
    3. **Adapted** -- ``adapter_fn(X_old_queries)`` vs ``X_new_corpus``.  The
       adapter's correction of old queries into the new space.

    For recall computation, the ground-truth relevant document for query *i*
    is corpus document *i* (identity mapping), which is standard when queries
    and corpus rows are aligned paired samples.

    When old and new embeddings have different dimensions (e.g. 384 vs 768),
    the no-adapter baseline zero-pads old embeddings to match the new
    dimension so that FAISS indexing can proceed.  This naturally yields
    poor retrieval quality, reflecting the true cost of a dimension mismatch
    without adaptation.

    Parameters
    ----------
    adapter_fn:
        Callable that maps ``(n, d_old)`` old embeddings to ``(n, d_new)``.
    X_old_queries:
        Old-model query embeddings, shape ``(n_queries, d_old)``.
    X_new_queries:
        New-model query embeddings, shape ``(n_queries, d_new)``.
    X_old_corpus:
        Old-model corpus embeddings, shape ``(n_corpus, d_old)``.
    X_new_corpus:
        New-model corpus embeddings, shape ``(n_corpus, d_new)``.
    k_values:
        Values of *k* for recall@k.

    Returns
    -------
    Dictionary with keys ``"oracle"``, ``"no_adapter"``, and ``"adapted"``,
    each mapping to a sub-dict of metrics (recall@k, MRR, mean cosine
    similarity).
    """
    n_queries = X_new_queries.shape[0]
    ground_truth = np.arange(n_queries)

    # Adapted queries
    X_adapted_queries = adapter_fn(X_old_queries)

    def _eval_setting(queries: np.ndarray, corpus: np.ndarray) -> dict:
        metrics: dict[str, float] = {}
        metrics.update(compute_recall_at_k(queries, corpus, ground_truth, k_values))
        metrics["mrr"] = compute_mrr(queries, corpus, ground_truth)
        return metrics

    # No-adapter baseline: when dimensions differ, zero-pad old embeddings
    # to match the new dimension so FAISS can build an index.
    d_new = X_new_corpus.shape[1]
    X_old_queries_padded = _pad_to_dim(X_old_queries, d_new)

    results = {
        "oracle": _eval_setting(X_new_queries, X_new_corpus),
        "no_adapter": _eval_setting(X_old_queries_padded, X_new_corpus),
        "adapted": _eval_setting(X_adapted_queries, X_new_corpus),
    }

    # Mean cosine similarity between adapted old queries and true new queries
    results["adapted"]["mean_cosine_sim"] = compute_cosine_similarity(
        X_adapted_queries, X_new_queries
    )

    return results
