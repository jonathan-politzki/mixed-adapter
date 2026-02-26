"""Dataset loading utilities for the local-drift-adapter experiments.

Provides functions to load and preprocess common information-retrieval
datasets (MS MARCO, Natural Questions, BEIR benchmarks) as well as a
heterogeneous multi-domain corpus used to evaluate whether local
adapters outperform global ones on mixed-domain data.

All loaders return plain Python lists so they remain model-agnostic.
Actual embedding is handled by :mod:`src.data.pair_generator`.
"""

from __future__ import annotations

import logging
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MS MARCO
# ---------------------------------------------------------------------------

def load_msmarco(
    split: str = "train",
    max_samples: int = 100_000,
) -> list[str]:
    """Load passage texts from the MS MARCO v2.1 dataset.

    Parameters
    ----------
    split : str
        Dataset split to load (``"train"`` or ``"validation"``).
    max_samples : int
        Maximum number of passage texts to return.  Because each
        example contains a *list* of passages, we flatten first and
        then truncate to *max_samples*.

    Returns
    -------
    list[str]
        Unique passage text strings.
    """
    logger.info("Loading MS MARCO v2.1 passages (split=%s) ...", split)
    try:
        ds = load_dataset("ms_marco", "v2.1", split=split, trust_remote_code=True)
    except TypeError:
        ds = load_dataset("ms_marco", "v2.1", split=split)

    texts: list[str] = []
    for example in ds:
        # Each example has a "passages" dict with key "passage_text" (list[str])
        passages = example.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        for t in passage_texts:
            t_stripped = t.strip()
            if t_stripped:
                texts.append(t_stripped)
        if len(texts) >= max_samples:
            break

    texts = texts[:max_samples]
    logger.info("Loaded %d MS MARCO passages.", len(texts))
    return texts


# ---------------------------------------------------------------------------
# Natural Questions
# ---------------------------------------------------------------------------

def load_nq(
    split: str = "train",
    max_samples: int = 50_000,
) -> list[str]:
    """Load question texts from the NQ-Open dataset.

    Parameters
    ----------
    split : str
        Dataset split to load (``"train"`` or ``"validation"``).
    max_samples : int
        Maximum number of question texts to return.

    Returns
    -------
    list[str]
        Question strings.
    """
    logger.info("Loading NQ-Open questions (split=%s) ...", split)
    ds = load_dataset("nq_open", split=split, trust_remote_code=True)

    texts: list[str] = []
    for example in ds:
        q = example.get("question", "").strip()
        if q:
            texts.append(q)
        if len(texts) >= max_samples:
            break

    texts = texts[:max_samples]
    logger.info("Loaded %d NQ-Open questions.", len(texts))
    return texts


# ---------------------------------------------------------------------------
# BEIR benchmarks
# ---------------------------------------------------------------------------

def load_beir_dataset(
    name: str,
    split: str = "test",
    max_samples: int = 50_000,
) -> dict[str, object]:
    """Load a BEIR benchmark dataset from HuggingFace.

    Supported names include ``"scifact"``, ``"fiqa"``, ``"arguana"``,
    ``"nfcorpus"``, etc.  The dataset is fetched from the
    ``BeIR/<name>`` HuggingFace Hub path.

    Parameters
    ----------
    name : str
        BEIR dataset name (e.g. ``"scifact"``).
    split : str
        Corpus split to load.  For BEIR, the corpus is usually only
        available under ``"corpus"``; queries under ``"queries"``.  When
        those canonical splits exist they take precedence over *split*.
    max_samples : int
        Cap on the number of corpus documents returned.

    Returns
    -------
    dict
        ``{"corpus": list[str], "queries": list[str], "qrels": dict | None}``

        * **corpus** -- Document texts (``title + " " + text``).
        * **queries** -- Query texts.
        * **qrels** -- ``{query_id: {doc_id: relevance}}`` if the
          dataset provides relevance judgements, else ``None``.
    """
    hf_path = f"BeIR/{name}"
    logger.info("Loading BEIR dataset '%s' from %s ...", name, hf_path)

    # -- corpus ---------------------------------------------------------------
    try:
        corpus_ds = load_dataset(hf_path, "corpus", split="corpus", trust_remote_code=True)
    except (ValueError, KeyError):
        # Some BEIR datasets use a different config/split naming
        corpus_ds = load_dataset(hf_path, split="corpus", trust_remote_code=True)

    corpus_texts: list[str] = []
    corpus_ids: list[str] = []
    for row in corpus_ds:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        combined = f"{title} {text}".strip() if title else text
        if combined:
            corpus_texts.append(combined)
            corpus_ids.append(str(row.get("_id", len(corpus_ids))))
        if len(corpus_texts) >= max_samples:
            break

    # -- queries --------------------------------------------------------------
    try:
        queries_ds = load_dataset(hf_path, "queries", split="queries", trust_remote_code=True)
    except (ValueError, KeyError):
        queries_ds = load_dataset(hf_path, split="queries", trust_remote_code=True)

    query_texts: list[str] = []
    query_ids: list[str] = []
    for row in queries_ds:
        q = (row.get("text") or "").strip()
        if q:
            query_texts.append(q)
            query_ids.append(str(row.get("_id", len(query_ids))))
        if len(query_texts) >= max_samples:
            break

    # -- qrels ----------------------------------------------------------------
    qrels: Optional[dict[str, dict[str, int]]] = None
    try:
        qrels_split = f"{split}" if split != "test" else "test"
        qrels_ds = load_dataset(hf_path, "default", split=qrels_split, trust_remote_code=True)
        qrels = {}
        for row in qrels_ds:
            qid = str(row.get("query-id", ""))
            did = str(row.get("corpus-id", ""))
            score = int(row.get("score", 0))
            if qid and did:
                qrels.setdefault(qid, {})[did] = score
    except Exception:
        # qrels not available for every BEIR dataset / split
        logger.debug("No qrels found for %s (split=%s).", name, split)
        qrels = None

    logger.info(
        "Loaded BEIR '%s': %d corpus docs, %d queries, qrels=%s.",
        name,
        len(corpus_texts),
        len(query_texts),
        "yes" if qrels else "no",
    )
    return {"corpus": corpus_texts, "queries": query_texts, "qrels": qrels}


# ---------------------------------------------------------------------------
# Heterogeneous multi-domain corpus
# ---------------------------------------------------------------------------

def load_heterogeneous_corpus(
    max_per_domain: int = 25_000,
) -> tuple[list[str], list[str]]:
    """Load a mixed-domain corpus spanning science, finance, and general text.

    This corpus is designed to test the hypothesis that *local* adapters
    fitted per-cluster improve over a single *global* adapter when the
    embedding space contains semantically diverse regions.

    Domains
    -------
    * **science** -- passages from the SciFact BEIR corpus.
    * **finance** -- passages from the FiQA BEIR corpus.
    * **general** -- passages from MS MARCO.

    Parameters
    ----------
    max_per_domain : int
        Maximum number of texts to draw from each domain.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(texts, domain_labels)`` where *domain_labels[i]* is one of
        ``"science"``, ``"finance"``, or ``"general"``.
    """
    logger.info("Building heterogeneous corpus (max %d per domain) ...", max_per_domain)

    texts: list[str] = []
    labels: list[str] = []

    # Science -- SciFact
    try:
        sci = load_beir_dataset("scifact", max_samples=max_per_domain)
        for t in sci["corpus"]:
            texts.append(t)
            labels.append("science")
    except Exception as exc:
        logger.warning("Could not load SciFact: %s", exc)

    # Finance -- FiQA
    try:
        fin = load_beir_dataset("fiqa", max_samples=max_per_domain)
        for t in fin["corpus"]:
            texts.append(t)
            labels.append("finance")
    except Exception as exc:
        logger.warning("Could not load FiQA: %s", exc)

    # General -- MS MARCO subset
    try:
        gen = load_msmarco(max_samples=max_per_domain)
        for t in gen:
            texts.append(t)
            labels.append("general")
    except Exception as exc:
        logger.warning("Could not load MS MARCO: %s", exc)

    logger.info(
        "Heterogeneous corpus: %d texts (%d science, %d finance, %d general).",
        len(texts),
        labels.count("science"),
        labels.count("finance"),
        labels.count("general"),
    )
    return texts, labels
