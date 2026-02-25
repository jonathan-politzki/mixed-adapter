from __future__ import annotations

import torch
import torch.nn as nn


class LowRankAffine(nn.Module):
    """Low-rank affine adapter for embedding-space translation.

    When ``input_dim == output_dim`` (the default), the transform is
    parameterised as an identity-plus-low-rank map::

        f(x) = x + (x @ U) @ V^T + b          (U, V: d x r,  b: d)

    The identity shortcut ensures the transform starts near identity,
    making optimisation stable for small drift.

    When ``input_dim != output_dim`` (e.g. 384 -> 768), the identity
    shortcut is not applicable so the transform becomes a pure low-rank
    affine map::

        f(x) = (x @ U) @ V^T + b              (U: d_in x r, V: d_out x r, b: d_out)

    This is equivalent to ``x @ W + b`` where ``W = U @ V^T`` is a
    ``(d_in, d_out)`` matrix of rank at most *r*.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int = 32,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.rank = rank
        self._same_dim = self.input_dim == self.output_dim

        self.U = nn.Parameter(torch.empty(input_dim, rank))
        self.V = nn.Parameter(torch.empty(self.output_dim, rank))
        self.b = nn.Parameter(torch.zeros(self.output_dim))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank affine transform.

        Parameters
        ----------
        x:
            Input embeddings of shape ``(*, d_in)``.

        Returns
        -------
        Transformed embeddings of shape ``(*, d_out)``.
        """
        low_rank = (x @ self.U) @ self.V.T + self.b
        if self._same_dim:
            return x + low_rank
        return low_rank

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
