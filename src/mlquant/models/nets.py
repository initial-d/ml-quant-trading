"""Two compact baselines: an MLP and a thin Transformer encoder.

Both consume a 1-D feature vector per stock-day and produce a scalar
prediction. They are intentionally small — the paper argues that the
*factor engineering* and *portfolio construction* layers carry most of
the alpha; the model's job is non-linear feature combination, not
deep representation learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Three-layer MLP with GELU and dropout."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Apply a transformer encoder across the *factor* axis.

    We tokenise each factor as one position, project to ``d_model``,
    add a learned positional embedding, and reduce with a CLS token.
    For ~50-200 factors this is cheap and lets the network learn
    pairwise factor interactions without manual cross terms.
    """

    def __init__(self, in_dim: int, d_model: int = 64, n_heads: int = 4,
                 depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.randn(in_dim, d_model) * 0.02)
        self.cls  = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> tokens [B, F, d_model]
        tokens = self.proj(x.unsqueeze(-1)) + self.pos
        cls = self.cls.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        h = self.encoder(seq)
        return self.head(h[:, 0]).squeeze(-1)
