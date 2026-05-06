"""Forward-return labels (training targets).

The legacy ``cuda_features.py`` ships a single label, ``target_01``,
which the paper uses throughout::

    target_01[t, i] = close[t+1, i] / close[t, i] - 1

masked by ``mask[t, i] & mask[t+1, i]``. Concretely, this is the
realised one-day forward simple return, computed only on stock-days
where the stock was tradable both at signal time *t* and at execution
time *t+1*. The last row is always masked because there is no ``t+1``.

We expose two helpers:

* :func:`target_01` — vectorised tensor implementation.
* :func:`forward_return` — generic ``k``-day forward return.

Both return ``(values, mask)`` to fit the rest of the feature engine.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..data.panel import Panel


Tensor = torch.Tensor


def forward_return(panel: Panel, horizon: int = 1) -> Tuple[Tensor, Tensor]:
    """``close[t+horizon] / close[t] - 1`` masked by tradability at both ends.

    Parameters
    ----------
    panel : Panel
    horizon : int
        Number of periods to look ahead. Must be >= 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    T, N = panel.close.shape
    out = torch.zeros_like(panel.close)
    out_mask = torch.zeros_like(panel.mask)

    fut_close = panel.close[horizon:]
    cur_close = panel.close[:-horizon].clamp_min(1e-9)
    fut_mask  = panel.mask[horizon:]
    cur_mask  = panel.mask[:-horizon]

    out[:-horizon] = (fut_close / cur_close - 1.0) * (fut_mask & cur_mask).float()
    out_mask[:-horizon] = fut_mask & cur_mask
    return out, out_mask


def target_01(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day forward simple return — the canonical training label."""
    return forward_return(panel, horizon=1)


__all__ = ["target_01", "forward_return"]
