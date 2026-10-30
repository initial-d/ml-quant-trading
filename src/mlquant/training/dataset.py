"""Cross-sectional dataset for daily-rebalance models.

Each sample is *one stock-day* — features at date ``t``, label is the
next-day return at ``t+1``. The dataset takes care of:

* selecting features with the signal-time factor mask,
* consuming labels already aligned to signal date ``t`` (no second shift),
* selecting labels with a separate forward-label mask when supplied,
* exposing batches as plain ``(features, labels)`` pairs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class FactorDataset(Dataset):
    """A flat dataset of (factor vector, next-day return) pairs.

    ``label_mask[t, n]`` describes the validity of the already-aligned label
    at ``t``. It need not require usable factors at ``t+1``. Omitting it keeps
    the legacy rule ``mask[t, n] & mask[t+1, n]``. These offline sample filters
    are not order-fill rules and must not select live portfolio holdings.
    """

    def __init__(
        self,
        factors: torch.Tensor,            # [T, N, F]
        mask:    torch.Tensor,            # [T, N]
        forward_returns: torch.Tensor,    # [T, N]   label for signal date t
        *,
        label_mask: Optional[torch.Tensor] = None,  # [T, N], already aligned to t
        date_index: Optional[torch.Tensor] = None,
        stock_index: Optional[torch.Tensor] = None,
    ) -> None:
        if factors.dim() != 3:
            raise ValueError("factors must be [T, N, F]")
        T, N, _ = factors.shape
        if mask.shape != (T, N) or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [T, N]")
        if forward_returns.shape != (T, N):
            raise ValueError("forward_returns must be [T, N]")
        if label_mask is not None and (
            label_mask.shape != (T, N) or label_mask.dtype != torch.bool
        ):
            raise ValueError("label_mask must be bool [T, N]")

        # Drop the last day (no forward return available) and any masked
        # cells.
        label_valid = mask[1:] if label_mask is None else label_mask[:-1]
        valid = (
            mask[:-1] & label_valid
            & torch.isfinite(forward_returns[:-1])
            & torch.isfinite(factors[:-1]).all(dim=-1)
        )
        flat_idx = valid.nonzero(as_tuple=False)        # [K, 2] (t, n)

        self.features = factors[:-1][flat_idx[:, 0], flat_idx[:, 1]]
        self.labels   = forward_returns[:-1][flat_idx[:, 0], flat_idx[:, 1]]

        self.dates  = (date_index[:-1] [flat_idx[:, 0]] if date_index  is not None else flat_idx[:, 0])
        self.stocks = (stock_index    [flat_idx[:, 1]] if stock_index is not None else flat_idx[:, 1])

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
