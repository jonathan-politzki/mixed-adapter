from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Residual MLP adapter for embedding-space translation.

    When ``input_dim == output_dim`` (the default), the transform is a
    standard residual connection::

        f(x) = x + MLP(x)

    where ``MLP`` is a stack of ``Linear -> LayerNorm -> GELU -> Dropout``
    blocks followed by a final linear projection back to the input
    dimension.  The residual connection ensures the transform starts
    near identity.

    When ``input_dim != output_dim`` (e.g. 384 -> 768), the identity
    shortcut is replaced by a learned linear projection so that
    dimensions are compatible::

        f(x) = projection(x) + MLP(x)

    where ``projection`` is ``Linear(input_dim, output_dim)`` and the
    MLP's final layer outputs ``output_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._same_dim = self.input_dim == self.output_dim

        # --- MLP trunk ---
        layers: List[nn.Module] = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.mlp = nn.Sequential(*layers)

        # --- Skip connection ---
        # When dimensions differ, use a learned linear projection instead
        # of the identity shortcut.
        if not self._same_dim:
            self.skip_projection = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual MLP transform.

        Parameters
        ----------
        x:
            Input embeddings of shape ``(*, d_in)``.

        Returns
        -------
        Transformed embeddings of shape ``(*, d_out)``.
        """
        if self._same_dim:
            return x + self.mlp(x)
        return self.skip_projection(x) + self.mlp(x)

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
