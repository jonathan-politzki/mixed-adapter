"""Training loop for PyTorch-based embedding adapters (Affine and ResidualMLP).

Procrustes alignment is solved in closed form and does not require iterative
training -- use :class:`src.adapters.procrustes.OrthogonalProcrustes` directly.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.training.losses import combined_loss, cosine_loss, mse_loss

logger = logging.getLogger(__name__)

_LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "mse": mse_loss,
    "cosine": cosine_loss,
    "combined": combined_loss,
}


class AdapterTrainer:
    """Train a PyTorch adapter module to map old embeddings to new embeddings.

    Parameters
    ----------
    adapter:
        A ``torch.nn.Module`` whose ``forward`` accepts ``(batch, d)`` tensors
        (e.g. :class:`~src.adapters.affine.LowRankAffine` or
        :class:`~src.adapters.residual_mlp.ResidualMLP`).
    lr:
        Learning rate for Adam.
    weight_decay:
        L2 regularisation coefficient.
    loss_fn:
        One of ``"mse"``, ``"cosine"``, or ``"combined"`` (default).
    device:
        Device string passed to ``torch.device``.
    """

    def __init__(
        self,
        adapter: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_fn: str = "combined",
        device: str = "cpu",
    ) -> None:
        if loss_fn not in _LOSS_REGISTRY:
            raise ValueError(f"Unknown loss_fn '{loss_fn}'. Choose from {list(_LOSS_REGISTRY)}")

        self.device = torch.device(device)
        self.adapter = adapter.to(self.device)
        self.loss_fn = _LOSS_REGISTRY[loss_fn]
        self.optimizer = torch.optim.Adam(
            self.adapter.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        epochs: int = 100,
        batch_size: int = 512,
        val_split: float = 0.1,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Train the adapter with early stopping.

        Parameters
        ----------
        X_old:
            Source embeddings of shape ``(n, d_old)``.
        X_new:
            Target embeddings of shape ``(n, d_new)``.
        epochs:
            Maximum number of training epochs.
        batch_size:
            Mini-batch size.
        val_split:
            Fraction of data reserved for validation (used for early stopping).
        patience:
            Number of epochs without improvement before stopping.
        verbose:
            Whether to display a tqdm progress bar.

        Returns
        -------
        dict
            Training history with keys ``"train_loss"``, ``"val_loss"``, and
            ``"best_epoch"``.
        """
        X_old_t = torch.as_tensor(X_old, dtype=torch.float32)
        X_new_t = torch.as_tensor(X_new, dtype=torch.float32)

        # Train / validation split (deterministic, last chunk = val)
        n = X_old_t.shape[0]
        n_val = max(1, int(n * val_split))
        n_train = n - n_val

        indices = torch.randperm(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_loader = DataLoader(
            TensorDataset(X_old_t[train_idx], X_new_t[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )
        val_X_old = X_old_t[val_idx].to(self.device)
        val_X_new = X_new_t[val_idx].to(self.device)

        # History tracking
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0

        epoch_iter = tqdm(range(1, epochs + 1), desc="Training", disable=not verbose)
        for epoch in epoch_iter:
            # --- train ---
            self.adapter.train()
            running_loss = 0.0
            n_batches = 0
            for batch_old, batch_new in train_loader:
                batch_old = batch_old.to(self.device)
                batch_new = batch_new.to(self.device)

                predicted = self.adapter(batch_old)
                loss = self.loss_fn(predicted, batch_new)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg_train_loss = running_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # --- validation ---
            self.adapter.eval()
            with torch.no_grad():
                val_pred = self.adapter(val_X_old)
                val_loss = self.loss_fn(val_pred, val_X_new).item()
            val_losses.append(val_loss)

            # --- early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.adapter.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            epoch_iter.set_postfix(train=f"{avg_train_loss:.5f}", val=f"{val_loss:.5f}")

            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

        # Restore best weights
        if best_state is not None:
            self.adapter.load_state_dict(best_state)
        self.adapter.to(self.device)

        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the adapter in inference mode.

        Parameters
        ----------
        X:
            Input embeddings of shape ``(n, d)``.

        Returns
        -------
        Transformed embeddings as a numpy array of the same shape.
        """
        self.adapter.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.adapter(X_t)
        return out.cpu().numpy()
