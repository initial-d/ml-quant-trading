"""Cross-sectional dataset for daily-rebalance models.

Each sample is *one stock-day* — features at date ``t``, label is the
next-day return at ``t+1``. The dataset takes care of:

* aligning the factor tensor with the bias-corrected mask,
* shifting the label by one day,
* dropping cells where either side of the prediction is non-tradable,
* exposing batches as plain ``(features, labels)`` pairs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class FactorDataset(Dataset):
    """A flat dataset of (factor vector, next-day return) pairs."""

    def __init__(
        self,
        factors: torch.Tensor,            # [T, N, F]
        mask:    torch.Tensor,            # [T, N]
        forward_returns: torch.Tensor,    # [T, N]   already bias-corrected
        *,
        date_index: Optional[torch.Tensor] = None,
        stock_index: Optional[torch.Tensor] = None,
    ) -> None:
        if factors.dim() != 3:
            raise ValueError("factors must be [T, N, F]")
        T, N, _ = factors.shape

        # Drop the last day (no forward return available) and any masked
        # cells.
        valid = mask[:-1] & mask[1:] & ~torch.isnan(forward_returns[:-1])
        flat_idx = valid.nonzero(as_tuple=False)        # [K, 2] (t, n)

        self.features = factors[:-1][flat_idx[:, 0], flat_idx[:, 1]]
        self.labels   = forward_returns[:-1][flat_idx[:, 0], flat_idx[:, 1]]

        self.dates  = (date_index[:-1] [flat_idx[:, 0]] if date_index  is not None else flat_idx[:, 0])
        self.stocks = (stock_index    [flat_idx[:, 1]] if stock_index is not None else flat_idx[:, 1])

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
