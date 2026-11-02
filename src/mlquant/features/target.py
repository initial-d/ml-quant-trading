"""Forward-return labels (training targets).

The legacy ``cuda_features.py`` ships a single label, ``target_01``,
which the paper uses throughout::

    target_01[t, i] = close[t+1, i] / close[t, i] - 1

masked by ``mask[t, i] & mask[t+1, i]``. Concretely, this is the
realised one-day forward simple return, computed only where the panel's
observation mask accepts both endpoints. This is label availability, not
proof of order execution. The last row is masked because there is no ``t+1``.

We expose two helpers:

* :func:`target_01` — vectorised tensor implementation.
* :func:`forward_return` — generic ``k``-day forward return.

Both return ``(values, mask)`` to fit the rest of the feature engine.
"""
from __future__ import annotations

from numbers import Integral
from typing import Tuple

import torch

from ..data.panel import Panel


Tensor = torch.Tensor


def forward_return(panel: Panel, horizon: int = 1) -> Tuple[Tensor, Tensor]:
    """``close[t+horizon] / close[t] - 1`` masked by observation availability.

    Parameters
    ----------
    panel : Panel
    horizon : int
        Positive integer number of panel rows to look ahead. A horizon at or
        beyond the panel length returns all-zero values and an all-false mask.

    Both endpoints must be accepted by the observation mask and have finite,
    positive closes. Nonfinite computed returns are invalid too. Invalid labels
    are exactly zero, including when a masked source cell contains NaN/Inf.
    For multi-row horizons only endpoints are checked; this is not a claim
    about intermediate-session observations or execution feasibility.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
    horizon = int(horizon)
    out = torch.zeros_like(panel.close)
    out_mask = torch.zeros_like(panel.mask)
    if horizon >= panel.close.shape[0]:
        return out, out_mask

    fut_close = panel.close[horizon:]
    cur_close = panel.close[:-horizon]
    fut_mask  = panel.mask[horizon:]
    cur_mask  = panel.mask[:-horizon]

    valid = (
        fut_mask & cur_mask & torch.isfinite(cur_close) & torch.isfinite(fut_close)
        & (cur_close > 0) & (fut_close > 0)
    )
    # Do not use NaN * 0 as masking, or clamp a zero/negative price into an
    # apparently valid extreme return. Neutral operands keep invalid cells safe.
    numerator = torch.where(valid, fut_close, torch.ones_like(fut_close))
    denominator = torch.where(valid, cur_close, torch.ones_like(cur_close))
    values = numerator / denominator - 1.0
    valid = valid & torch.isfinite(values)
    out[:-horizon] = torch.where(valid, values, torch.zeros_like(values))
    out_mask[:-horizon] = valid
    return out, out_mask


def target_01(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day forward simple return — the canonical training label."""
    return forward_return(panel, horizon=1)


__all__ = ["target_01", "forward_return"]
