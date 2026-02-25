"""Loss functions for embedding-space adapter training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target embedding batches.

    Parameters
    ----------
    predicted:
        Predicted embeddings of shape ``(batch, d)``.
    target:
        Target embeddings of shape ``(batch, d)``.

    Returns
    -------
    Scalar MSE loss.
    """
    return F.mse_loss(predicted, target)


def cosine_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus mean cosine similarity between predicted and target embeddings.

    Minimising this loss pushes predicted vectors to align directionally
    with their targets, which is especially important for retrieval where
    cosine similarity is the dominant distance metric.

    Parameters
    ----------
    predicted:
        Predicted embeddings of shape ``(batch, d)``.
    target:
        Target embeddings of shape ``(batch, d)``.

    Returns
    -------
    Scalar cosine loss in ``[0, 2]``.
    """
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    return 1.0 - cos_sim.mean()


def combined_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.5,
) -> torch.Tensor:
    """Weighted combination of MSE and cosine loss.

    ``loss = mse_weight * MSE + (1 - mse_weight) * cosine_loss``

    The MSE term preserves magnitude fidelity while the cosine term
    preserves directional alignment.

    Parameters
    ----------
    predicted:
        Predicted embeddings of shape ``(batch, d)``.
    target:
        Target embeddings of shape ``(batch, d)``.
    mse_weight:
        Weight for the MSE component; cosine weight is ``1 - mse_weight``.
        Must be in ``[0, 1]``.

    Returns
    -------
    Scalar combined loss.
    """
    if not 0.0 <= mse_weight <= 1.0:
        raise ValueError(f"mse_weight must be in [0, 1], got {mse_weight}")

    return mse_weight * mse_loss(predicted, target) + (1.0 - mse_weight) * cosine_loss(
        predicted, target
    )
