# Local Drift-Adapters

Extends [Drift-Adapter](https://arxiv.org/abs/2509.23471) (Vejendla, EMNLP 2025) with **local adapters** — a mixture-of-experts approach where per-cluster adapters specialize in different regions of the embedding space, improving translation quality over a single global adapter.

Paper: `paper/local_drift_adapter.tex`

## The idea

When you upgrade an embedding model, old stored embeddings no longer align with new ones. Re-embedding is expensive. Drift-Adapter learns a lightweight mapping from old → new space.

**The problem**: a single global adapter assumes the transformation is uniform everywhere. It isn't — different document types occupy different regions with different drift characteristics.

**Our approach**: cluster the embedding space, train a separate adapter per cluster, and route queries via soft MoE blending. This transplants a known technique from cross-lingual alignment (Dan et al., COLING 2020) to the model upgrade setting.

## Key results

On MS MARCO (100K passages), local adapters consistently improve over global baselines, with gains scaling with model-pair difficulty:

| Model Pair | Dims | Global Procrustes | Global Affine | Best Local Affine | Local Procrustes |
|---|---|---|---|---|---|
| MiniLM-L6 → MiniLM-L12 | 384→384 | 0.992 | 0.986 | 0.989 (k=32) | 0.993 |
| MiniLM-L6 → BGE-small | 384→384 | 0.989 | 0.887 | **0.964** (k=32) | 0.990 |
| BGE-small → BGE-base | 384→768 | 0.992 | 0.556 | **0.666** (k=16) | 0.992 |
| MiniLM-L6 → E5-large | 384→1024 | 0.979 | 0.199 | **0.523** (k=32) | **0.984** |

On the hardest pair (MiniLM → E5-large), local Procrustes **outperforms global Procrustes** (0.984 vs 0.979 R@1).

## Quick start

```bash
# Install
pip install -e .

# Run all experiments (generates pairs, trains adapters, evaluates)
python scripts/run_all_experiments.py --device cuda

# Or run a single experiment
python scripts/run_experiment.py \
    --config configs/local_drift_aware.yaml \
    --pairs data/pairs/minilm-to-bge_msmarco.npz

# Generate embedding pairs only
python scripts/generate_pairs.py \
    --model-pair minilm-to-bge \
    --dataset msmarco \
    --max-samples 100000 \
    --output data/pairs/
```

## Model pairs

| Name | Old → New | Dimensions | Scenario |
|------|-----------|-----------|----------|
| `minilm-6-to-12` | MiniLM-L6-v2 → MiniLM-L12-v2 | 384 → 384 | Same family |
| `minilm-to-bge` | MiniLM-L6-v2 → BGE-small-en-v1.5 | 384 → 384 | Cross-family |
| `bge-small-to-base` | BGE-small-en-v1.5 → BGE-base-en-v1.5 | 384 → 768 | Cross-dimension |
| `minilm-to-e5-large` | MiniLM-L6-v2 → E5-large-v2 | 384 → 1024 | Cross-family + cross-dim |

## Project structure

```
configs/            YAML experiment configs
scripts/
  generate_pairs.py   Embed corpus with both models, save paired embeddings
  run_experiment.py    Train adapter(s) and evaluate retrieval quality
  run_all_experiments.py   Full experiment suite with cluster sweep
  analyze_drift.py    Drift visualization and spatial analysis
src/
  adapters/         Procrustes, LowRankAffine, ResidualMLP
  clustering/       KMeans, DriftAwareClustering, MoERouter
  data/             Dataset loaders, EmbeddingPairGenerator
  evaluation/       Retrieval metrics (Recall@k, MRR, cosine sim)
  training/         AdapterTrainer, loss functions
  config.py         Config loader with deep merge + CLI overrides
paper/              LaTeX paper and references
```

## Configuration

Experiments use YAML configs (`configs/base.yaml` + experiment overlays). Override from CLI:

```bash
python scripts/run_experiment.py \
    --config configs/local_drift_aware.yaml \
    --pairs data/pairs/minilm-to-bge_msmarco.npz \
    --clustering.n_clusters=16
```

## Requirements

- Python >= 3.10
- PyTorch, sentence-transformers, faiss-cpu, scikit-learn, datasets
- See `pyproject.toml` for full list
