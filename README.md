# mixed-adapter

Extends the work from [DRIFT-adapter](https://github.com/stanford-futuredata/DRIFT-adapter). Trains **embedding adapters** that map vectors from an old embedding model to a new one, so you can keep using existing embeddings when upgrading models—without re-embedding your entire corpus.

## What it does

When you upgrade an embedding model (e.g. MiniLM-L6 → MiniLM-L12), old stored embeddings no longer align with new ones. Re-embedding everything is expensive. Adapters learn a mapping from old → new space so you can keep old embeddings and still use them with the new model.

This project supports:

- **Global adapters** — a single mapping for the whole space
- **Local adapters** — a mixture-of-experts setup with per-cluster adapters and routing

## Quick start

```bash
# Install
pip install -e .

# 1. Generate paired embeddings (old model + new model on same texts)
python scripts/generate_pairs.py \
    --model-pair minilm-6-to-12 \
    --dataset msmarco \
    --max-samples 100000 \
    --output data/pairs/

# 2. Train and evaluate an adapter
python scripts/run_experiment.py \
    --config configs/global_procrustes.yaml \
    --pairs data/pairs/minilm-6-to-12_msmarco.npz
```

## Components

| Component | Description |
|-----------|-------------|
| **Adapters** | Procrustes (closed-form rotation), Affine (low-rank linear), Residual MLP (neural) |
| **Clustering** | None (global), K-means, or drift-aware (position + drift direction) |
| **Router** | Hard (one adapter) or soft (weighted blend) routing for MoE |
| **Evaluation** | Retrieval metrics (recall@k) for oracle, no-adapter, and adapted settings |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_pairs.py` | Generate (old, new) embedding pairs from a text corpus |
| `scripts/run_experiment.py` | Train adapters and evaluate retrieval quality |
| `scripts/run_all_experiments.py` | Run full experiment suite (generate pairs + all configs) |
| `scripts/analyze_drift.py` | Visualize drift patterns and cluster structure |

## Configuration

Experiments are configured via YAML (`configs/base.yaml` + experiment overrides). Override from CLI:

```bash
python scripts/run_experiment.py --config configs/local_drift_aware.yaml \
    --pairs data/pairs/minilm-6-to-12_msmarco.npz \
    --override clustering.n_clusters=16
```

## Project structure

```
configs/          # YAML configs (base + experiment variants)
scripts/          # Entry points: generate_pairs, run_experiment, run_all, analyze_drift
src/
  adapters/       # Procrustes, Affine, ResidualMLP
  clustering/     # KMeans, DriftAware, MoERouter
  data/           # Pair generator, dataset loaders
  evaluation/     # Retrieval metrics
  training/       # Trainer, losses
```

## Requirements

- Python ≥ 3.10
- PyTorch, sentence-transformers, faiss-cpu, scikit-learn, datasets, etc. (see `pyproject.toml`)
