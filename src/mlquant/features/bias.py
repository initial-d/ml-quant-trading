"""Bias-correction utilities for A-share microstructure quirks.

The two most common look-ahead biases in A-share alpha research are:

1. **Limit-up days** — when a stock closes at the +10 % cap, the close
   you see *did not actually fill any unsatisfied buy orders*. Treating
   such a day as tradable inflates returns.
2. **Limit-down days** — symmetric problem on the downside.

Both must be excluded from training labels and from the tradable
universe at portfolio construction time.
"""
from __future__ import annotations

import torch

from ..data.panel import Panel


def limit_move_mask(panel: Panel, *, limit_pct: float = 0.098) -> torch.Tensor:
    """Boolean mask, ``True`` where the day's move is **inside** ±``limit_pct``.

    A return outside the band is the unambiguous signature of a limit
    move on most A-shares (10 % main-board, 20 % STAR/ChiNext — pick
    whichever applies via ``limit_pct``).
    """
    prev = torch.roll(panel.close, shifts=1, dims=0)
    prev[0] = panel.close[0]
    move = (panel.close / prev.clamp_min(1e-9)) - 1.0
    inside = move.abs() <= limit_pct
    inside[0] = False                 # day 0 has no previous close
    return inside & panel.mask
