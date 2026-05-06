"""Bias-correction utilities for A-share microstructure quirks.

The two most common look-ahead biases in A-share alpha research are:

1. **Limit-up days** — when a stock closes at the +10 % cap, the close
   you see *did not actually fill any unsatisfied buy orders*. Treating
   such a day as tradable inflates returns.
2. **Limit-down days** — symmetric problem on the downside.

Both must be excluded from training labels and from the tradable
universe at portfolio construction time.

Two regimes
-----------
* **Real limits available** (``Panel.limit_up`` / ``Panel.limit_down`` set):
  use the exchange-published prices — this is the regime real Wind
  data lands in. A cell is masked if ``close >= limit_up - eps`` or
  ``close <= limit_down + eps``.
* **Proxy regime** (synthetic data, or any feed without explicit limits):
  fall back to the ``|return| > limit_pct`` heuristic, which is the
  unambiguous signature of a limit move on most A-shares.

Both code paths are exercised by the test suite via the synthetic
generator (which fills both real and derived fields) so this module
behaves identically on synthetic and real feeds.
"""
from __future__ import annotations

import torch

from ..data.panel import Panel


def limit_move_mask(
    panel: Panel,
    *,
    limit_pct: float = 0.098,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Boolean mask, ``True`` where the day's price is **inside** the limit band.

    When :attr:`Panel.has_real_limits` is True we use the exchange-published
    ``limit_up`` / ``limit_down`` prices and require ``limit_down + eps <
    close < limit_up - eps``. Otherwise we fall back to the
    ``|return| <= limit_pct`` proxy.

    Day 0 is always masked False because no previous close is available
    to evaluate the return proxy. (Real-limit cells on day 0 are still
    valid but we follow the legacy convention of masking the very first
    row to keep the two regimes aligned.)
    """
    if panel.has_real_limits:
        up   = panel.limit_up.clamp_min(0.0)
        down = panel.limit_down.clamp_min(0.0)
        # Treat zero/missing limit prices as "no limit" (fall through).
        has_band = (up > 0) & (down > 0)
        # Inside the band on every cell that has one, plus a tolerance
        # so float round-off doesn't trip the comparison.
        inside_band = (panel.close < up - eps) & (panel.close > down + eps)
        # Where the band is missing, defer to the return-proxy rule.
        ret_proxy = _return_inside(panel, limit_pct)
        inside = torch.where(has_band, inside_band, ret_proxy)
    else:
        inside = _return_inside(panel, limit_pct)

    inside[0] = False                 # day 0 has no previous close
    return inside & panel.mask


def _return_inside(panel: Panel, limit_pct: float) -> torch.Tensor:
    """``|return| <= limit_pct`` proxy mask."""
    if panel.last_close is not None:
        prev = panel.last_close.clamp_min(1e-9)
    else:
        prev = torch.roll(panel.close, shifts=1, dims=0).clamp_min(1e-9)
        prev[0] = panel.close[0]
    move = (panel.close / prev) - 1.0
    return move.abs() <= limit_pct


__all__ = ["limit_move_mask"]
