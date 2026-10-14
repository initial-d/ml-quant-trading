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

A single scalar ``limit_pct`` only fits Main Board names. ChiNext and STAR
Market names move up to ±20 % and Beijing Stock Exchange names up to ±30 %,
so a 9.8 % proxy wrongly masks ordinary moves on those boards while never
catching their real limit days. :func:`board_limit_pct` builds a per-cell
threshold from ticker codes and dates; pass it as ``limit_pct``. It cannot
see ST status (±5 %) or the no-limit first days after an IPO — feeds that
care about those should supply real ``limit_up`` / ``limit_down`` prices.

Both code paths are exercised by the test suite via the synthetic
generator (which fills both real and derived fields) so this module
behaves identically on synthetic and real feeds.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ..data.panel import Panel


# Proxy thresholds sit slightly inside each board's band so a move that
# closes exactly at the limit (after price rounding) is still caught.
MAIN_BOARD_LIMIT_PCT = 0.098
GROWTH_BOARD_LIMIT_PCT = 0.198      # ChiNext (since 2020-08-24) and STAR Market
BSE_LIMIT_PCT = 0.298               # Beijing Stock Exchange
CHINEXT_20PCT_START = np.datetime64("2020-08-24")

_CODE_RE = re.compile(r"(\d{6})")


def _board_of(ticker: str) -> str:
    """Classify an A-share ticker by board; non-A-share tickers return ``"other"``."""
    text = str(ticker)
    match = _CODE_RE.search(text)
    if match is None:
        return "other"
    code = match.group(1)
    lowered = text.lower()
    if lowered.startswith("bj") or lowered.endswith(".bj") or code.startswith(("4", "8", "92")):
        return "bse"
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main"


def board_limit_pct(
    stocks: Sequence[str],
    dates: Optional[Sequence] = None,
    *,
    default: float = MAIN_BOARD_LIMIT_PCT,
) -> torch.Tensor:
    """Per-cell limit-move proxy thresholds from A-share ticker codes.

    Accepts plain codes (``"300750"``), Baostock style (``"sz.300750"``) and
    Yahoo style (``"300750.SZ"``). Tickers without a six-digit code (US
    tickers, synthetic ``SYN00001``) get ``default``.

    Returns ``[N]`` when ``dates`` is ``None``. With ``dates`` it returns
    ``[T, N]`` and applies the ChiNext 10 % → 20 % change on 2020-08-24, so
    ChiNext rows before that date use the Main Board threshold.
    """
    boards = [_board_of(s) for s in stocks]
    per_stock = np.array(
        [
            GROWTH_BOARD_LIMIT_PCT if b in ("star", "chinext")
            else BSE_LIMIT_PCT if b == "bse"
            else MAIN_BOARD_LIMIT_PCT if b == "main"
            else default
            for b in boards
        ],
        dtype=np.float32,
    )
    if dates is None:
        return torch.from_numpy(per_stock)

    day = np.asarray(dates).astype("datetime64[D]")
    out = np.broadcast_to(per_stock, (day.shape[0], per_stock.shape[0])).copy()
    chinext = np.array([b == "chinext" for b in boards])
    before = day < CHINEXT_20PCT_START
    out[np.ix_(before, chinext)] = MAIN_BOARD_LIMIT_PCT
    return torch.from_numpy(out)


def limit_move_mask(
    panel: Panel,
    *,
    limit_pct: Union[float, torch.Tensor] = MAIN_BOARD_LIMIT_PCT,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Boolean mask, ``True`` where the day's price is **inside** the limit band.

    When :attr:`Panel.has_real_limits` is True we use the exchange-published
    ``limit_up`` / ``limit_down`` prices and require ``limit_down + eps <
    close < limit_up - eps``. Otherwise we fall back to the
    ``|return| <= limit_pct`` proxy. ``limit_pct`` may be a scalar, an
    ``[N]`` per-stock tensor or a ``[T, N]`` tensor such as the output of
    :func:`board_limit_pct`.

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


def _return_inside(panel: Panel, limit_pct: Union[float, torch.Tensor]) -> torch.Tensor:
    """``|return| <= limit_pct`` proxy mask."""
    if panel.last_close is not None:
        prev = panel.last_close.clamp_min(1e-9)
    else:
        prev = torch.roll(panel.close, shifts=1, dims=0).clamp_min(1e-9)
        prev[0] = panel.close[0]
    move = (panel.close / prev) - 1.0
    if isinstance(limit_pct, torch.Tensor):
        limit_pct = limit_pct.to(device=move.device, dtype=move.dtype)
    return move.abs() <= limit_pct


__all__ = ["limit_move_mask", "board_limit_pct"]
