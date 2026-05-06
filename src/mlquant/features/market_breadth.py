"""Market-breadth / cross-sectional return-distribution features.

These are the "大盘涨跌分布信号" the legacy ``cuda_features.py`` exposes
under names like ``CR_-10 .. CR_10``, ``OR_-10 .. OR_10`` and
``cs_rank_{close,open,high,low,avg,amount}``. They are *cross-sectional
broadcast* signals — every stock on date ``t`` sees the *same value*
because the signal is a property of the whole market on that day.

Two families are produced:

1. **Per-stock cross-sectional ranks** of OHLCV/avg/amount within each
   trading day, mirroring the legacy ``calc_basedata`` output.
2. **Cross-sectional return-distribution buckets**: for each integer
   bucket ``b ∈ [-10, 10]`` we publish the fraction of tradable stocks
   whose rounded close-on-close return (in %) equals ``b``. These 21
   numbers form the daily "rise/fall histogram" that the legacy
   pipeline used as a market regime descriptor. The same is computed
   for open-on-close (``OR_*``).

Why broadcast?
    The downstream alpha set takes ``[T, N]`` inputs, so a market-only
    signal is replicated across the stock axis. This wastes a bit of
    memory but keeps the data layout uniform and mask-aware.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..data.panel import Panel
from .tensor_factors import cs_rank


Tensor = torch.Tensor

# Bucket grid: -10 % .. +10 % in 1 % steps -> 21 buckets.
_BUCKETS: tuple[int, ...] = tuple(range(-10, 11))


# ---------------------------------------------------------------------------
# Per-stock cross-sectional ranks
# ---------------------------------------------------------------------------
def cs_rank_field(panel: Panel, name: str) -> Tuple[Tensor, Tensor]:
    """Cross-sectional percentile rank of ``panel.<name>`` within each date.

    The legacy code uses ``ascending=False`` (largest → rank 1.0). We
    keep that convention for parity with the pickled feature tables.
    """
    field = getattr(panel, name, None)
    if field is None:
        raise KeyError(f"panel has no field {name!r}")
    return cs_rank(field, panel.mask, descending=True)


def cs_rank_avg(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of the daily VWAP (``S_DQ_AVGPRICE``)."""
    return cs_rank(panel.vwap, panel.mask, descending=True)


def cs_rank_amount(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of the daily turnover ``amount``.

    Falls back to ``vwap * volume`` when :attr:`Panel.amount` is absent.
    """
    if panel.amount is not None:
        amt = panel.amount
    else:
        amt = panel.vwap * panel.volume
    return cs_rank(amt, panel.mask, descending=True)


# ---------------------------------------------------------------------------
# Cross-sectional return-distribution buckets (CR_*, OR_*)
# ---------------------------------------------------------------------------
def _bucketize_pct(ret: Tensor, mask: Tensor) -> Tensor:
    """Round ``ret`` into integer percentage buckets, clipped to ``[-10, 10]``.

    Returns an ``int64`` tensor with the same shape as ``ret``; cells where
    ``mask`` is False are set to a sentinel value ``-100`` so the caller
    can drop them.
    """
    pct = (ret * 100.0).round().to(torch.int64)
    pct = pct.clamp(min=_BUCKETS[0], max=_BUCKETS[-1])
    return pct.masked_fill(~mask, -100)


def _broadcast_histogram(buckets: Tensor, mask: Tensor) -> Dict[int, Tensor]:
    """For every date, compute the fraction of tradable stocks in each bucket.

    Returns a dict ``{b: tensor[T, N]}`` where each tensor is the daily
    fraction broadcast across the stock axis.
    """
    T, N = buckets.shape
    out: Dict[int, Tensor] = {}
    n_valid = mask.sum(dim=1, keepdim=True).clamp_min(1).float()      # [T, 1]
    for b in _BUCKETS:
        count = ((buckets == b) & mask).sum(dim=1, keepdim=True).float()  # [T, 1]
        frac  = count / n_valid                                            # [T, 1]
        out[b] = frac.expand(T, N).contiguous()
    return out


def close_return_distribution(panel: Panel) -> Dict[int, Tensor]:
    """``CR_-10..CR_10`` — daily fraction of stocks at each rounded close-return.

    Definition (matches legacy)
    ---------------------------
    ``ret_t = close_t / last_close_t - 1``, clipped to ±10 %, rounded to
    the nearest integer percent. Buckets index 21 disjoint events; the
    sum of the 21 ``CR_*`` series at any date equals 1.

    Returns
    -------
    dict[int, Tensor]
        Mapping bucket → ``[T, N]`` tensor (broadcast across stocks).
    """
    ret = panel.returns                                # last_close-aware
    mask = panel.mask
    buckets = _bucketize_pct(ret, mask)
    return _broadcast_histogram(buckets, mask)


def open_return_distribution(panel: Panel) -> Dict[int, Tensor]:
    """``OR_-10..OR_10`` — same idea but on ``(open_t - last_close_t) / last_close_t``."""
    if panel.last_close is not None:
        prev = panel.last_close.clamp_min(1e-9)
        valid_prev = panel.last_close > 0
    else:
        prev = torch.roll(panel.close, shifts=1, dims=0).clamp_min(1e-9)
        valid_prev = torch.ones_like(panel.mask)
        valid_prev[0] = False
    ret = panel.open / prev - 1.0
    mask = panel.mask & valid_prev
    buckets = _bucketize_pct(ret, mask)
    return _broadcast_histogram(buckets, mask)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_market_breadth(panel: Panel) -> Tuple[Tensor, Tensor, List[str]]:
    """Compute every market-breadth feature into a stacked ``[T, N, F]`` tensor.

    Output columns
    --------------
    * ``cs_rank_close``, ``cs_rank_open``, ``cs_rank_high``,
      ``cs_rank_low``, ``cs_rank_avg``, ``cs_rank_amount``
    * ``CR_-10`` .. ``CR_10``  (21 buckets)
    * ``OR_-10`` .. ``OR_10``  (21 buckets)

    => 6 + 21 + 21 = 48 columns total.
    """
    cols: List[Tensor] = []
    names: List[str] = []
    joint_mask: Tensor = panel.mask.clone()

    for fname in ("close", "open", "high", "low"):
        v, m = cs_rank_field(panel, fname)
        cols.append(v)
        names.append(f"cs_rank_{fname}")
        joint_mask = joint_mask & m

    v, m = cs_rank_avg(panel)
    cols.append(v); names.append("cs_rank_avg"); joint_mask = joint_mask & m
    v, m = cs_rank_amount(panel)
    cols.append(v); names.append("cs_rank_amount"); joint_mask = joint_mask & m

    cr = close_return_distribution(panel)
    for b in _BUCKETS:
        cols.append(cr[b])
        names.append(f"CR_{b}")

    or_ = open_return_distribution(panel)
    for b in _BUCKETS:
        cols.append(or_[b])
        names.append(f"OR_{b}")

    factors = torch.stack(cols, dim=-1)       # [T, N, F]
    return factors, joint_mask, names


__all__ = [
    "cs_rank_field",
    "cs_rank_avg",
    "cs_rank_amount",
    "close_return_distribution",
    "open_return_distribution",
    "compute_market_breadth",
]
