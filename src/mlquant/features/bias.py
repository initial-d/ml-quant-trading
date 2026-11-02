"""Conservative price-limit exclusion policy for factor research.

A close at a price limit is an observed price, and a position already held
still earns its return. This module excludes such observations symmetrically
from factor windows as a research policy. It does not establish lookahead
bias, invalidate the observed price, or simulate directional order fills.
Buy/sell feasibility depends on order timing, side, liquidity and queue state.

With explicit ``limit_up`` / ``limit_down`` prices, cells near either band
edge are excluded. Missing bands fall back to ``|return| <= limit_pct``;
that proxy is a heuristic, not an unambiguous identification of limit days.
``eps`` is an absolute price tolerance and must suit the feed's units.

A single scalar threshold only fits some boards. :func:`board_limit_pct`
handles board/date rules, but cannot identify ST status or no-limit IPO days.
Prefer reliable explicit bands and document the exclusion policy in reports.
"""
from __future__ import annotations

import re
from dataclasses import replace
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
        move = (panel.close / prev) - 1.0
    else:
        # Halt-aware: measured from the last tradable close, so a zero-filled
        # halt row does not turn the resumption day into a fake limit move.
        move = panel.returns
    if isinstance(limit_pct, torch.Tensor):
        limit_pct = limit_pct.to(device=move.device, dtype=move.dtype)
    return move.abs() <= limit_pct


def apply_limit_mask(
    panel: Panel,
    *,
    limit_pct: Union[float, torch.Tensor] = MAIN_BOARD_LIMIT_PCT,
    eps: float = 1e-3,
) -> Panel:
    """Return ``panel`` with limit-move days removed from its ``mask``.

    This is the mask-first step. Factor code reads ``panel.mask``, so
    computing factors on the returned panel keeps limit-move prices out of
    every rolling window; intersecting :func:`limit_move_mask` with the
    factor mask afterwards only drops the limit day itself and still lets
    that excluded observation enter later rolling windows. This difference
    demonstrates exclusion-policy propagation, not invalid market prices.

    Keep using the original panel for P&L: a position held into a limit
    move still earns that return.
    """
    return replace(panel, mask=limit_move_mask(panel, limit_pct=limit_pct, eps=eps))


__all__ = ["limit_move_mask", "board_limit_pct", "apply_limit_mask"]
