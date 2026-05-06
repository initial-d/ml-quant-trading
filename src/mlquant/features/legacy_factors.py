"""Port of the legacy ``cuda_features.py`` factor zoo onto masked tensors.

The legacy implementation lives at ``legacy/cuda_features.py`` (~2 000
lines, cupy-backed) and exposes ~140 named features under prefixes
``best_*``, ``change_*``, ``original_*``, ``stock_*``, ``extra_*``,
``old_*`` and ``add_*``. Most of them are short algebraic combinations
of the same mask-aware primitives we already have here, only with
different parameters / different lookback windows.

This module ports a *representative* slice — one or two factors from
each family — onto the modern :mod:`mlquant.features.tensor_factors`
primitives, with a uniform ``[T, N]`` mask-aware signature. The full
142-name table is materialisable by adding more entries to
:data:`LEGACY_REGISTRY` using the same helpers; we intentionally keep
the porting style declarative so growing the table is mechanical.

Why a curated slice rather than all 142?
    The legacy file mixes pure algebra with pickled "extracted" features
    (CR_/OR_ buckets) computed offline. The CR_/OR_ ones live in
    :mod:`mlquant.features.market_breadth`. The remaining algebraic
    ~120 are mostly minor variants of the patterns ported here. The
    public test suite covers the *templates*; downstream code can
    register additional factors via :func:`register_legacy_factor`.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, cs_zscore,
    delay, delta,
    ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum,
)


Tensor = torch.Tensor
LegacyFn = Callable[[Panel], Tuple[Tensor, Tensor]]
LEGACY_REGISTRY: Dict[str, LegacyFn] = {}


def register_legacy_factor(name: str) -> Callable[[LegacyFn], LegacyFn]:
    def deco(fn: LegacyFn) -> LegacyFn:
        if name in LEGACY_REGISTRY:
            raise ValueError(f"duplicate legacy factor name: {name}")
        LEGACY_REGISTRY[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(num: Tensor, den: Tensor, eps: float = 1e-12) -> Tensor:
    return num / den.abs().clamp_min(eps).copysign(den + eps)


def _close_loc(panel: Panel) -> Tensor:
    """Where in the day's range did the close land? (-1 = at low, +1 = at high)."""
    rng = (panel.high - panel.low).clamp_min(1e-9)
    return (2.0 * panel.close - panel.high - panel.low) / rng


def _ts_mean_close_loc(panel: Panel, window: int) -> Tuple[Tensor, Tensor]:
    cl = _close_loc(panel)
    return ts_mean(cl, panel.mask, window)


# ---------------------------------------------------------------------------
# best_*  family — close-location momentum over varying windows
# Legacy template:   ``-(2*C - H - L) / (H - L + eps)``  averaged over W
# ---------------------------------------------------------------------------
@register_legacy_factor("best_001")
def best_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    v, m = _ts_mean_close_loc(panel, 5)
    return -v, m


@register_legacy_factor("best_002")
def best_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    v, m = _ts_mean_close_loc(panel, 10)
    return -v, m


@register_legacy_factor("best_003")
def best_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    v, m = _ts_mean_close_loc(panel, 20)
    return -v, m


@register_legacy_factor("best_006")
def best_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Volume-weighted close-location momentum."""
    cl = _close_loc(panel) * panel.volume
    num, m1 = ts_sum(cl, panel.mask, 10)
    den, m2 = ts_sum(panel.volume, panel.mask, 10)
    out = -num / den.clamp_min(1.0)
    return out * (m1 & m2).float(), m1 & m2


@register_legacy_factor("best_014")
def best_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of 20-day close-location mean."""
    v, m = _ts_mean_close_loc(panel, 20)
    r, rm = cs_rank(-v, m)
    return r - 0.5, rm


# ---------------------------------------------------------------------------
# change_*  family — short-window change-of-velocity signals
# ---------------------------------------------------------------------------
@register_legacy_factor("change_001")
def change_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day return minus its 5-day mean (mean-reversion proxy)."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    avg, am = ts_mean(ret, rm, 5)
    return (ret - avg) * (rm & am).float(), rm & am


@register_legacy_factor("change_002")
def change_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Acceleration: Δ(Δclose, 1) over 1 day."""
    d1, m1 = delta(panel.close, panel.mask, 1)
    d2, m2 = delta(d1, m1, 1)
    return d2 / panel.close.clamp_min(1e-9), m2


@register_legacy_factor("change_003")
def change_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Volume Δ-rank: rank of (volume - ts_mean(volume, 20))."""
    avg, am = ts_mean(panel.volume, panel.mask, 20)
    dev = (panel.volume - avg) * am.float()
    return cs_rank(dev, am)


@register_legacy_factor("change_004")
def change_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Range expansion: (high - low) / ts_mean(high - low, 20)."""
    rng_ = panel.high - panel.low
    avg, am = ts_mean(rng_, panel.mask, 20)
    out = rng_ / avg.clamp_min(1e-9) - 1.0
    return out * am.float(), am


@register_legacy_factor("change_005")
def change_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Sign flip count: ts_sum(sign(Δclose), 5)."""
    d, dm = delta(panel.close, panel.mask, 1)
    s = torch.sign(d)
    return ts_sum(s, dm, 5)


# ---------------------------------------------------------------------------
# original_*  family — direct close/volume statistics
# ---------------------------------------------------------------------------
@register_legacy_factor("original_001")
def original_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """20-day close volatility, cross-sectionally z-scored."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    sd, sm = ts_std(ret, rm, 20)
    z, zm = cs_zscore(sd, sm)
    return z, zm


@register_legacy_factor("original_002")
def original_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 20-day high distance, ranked."""
    hi, hm = ts_max(panel.close, panel.mask, 20)
    dist = panel.close / hi.clamp_min(1e-9) - 1.0
    return cs_rank(dist * hm.float(), hm)


@register_legacy_factor("original_003")
def original_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 20-day low distance, ranked."""
    lo, lm = ts_min(panel.close, panel.mask, 20)
    dist = panel.close / lo.clamp_min(1e-9) - 1.0
    return cs_rank(dist * lm.float(), lm)


@register_legacy_factor("original_004")
def original_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of close over 60 days."""
    return ts_rank(panel.close, panel.mask, 60)


@register_legacy_factor("original_005")
def original_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of volume over 60 days."""
    return ts_rank(panel.volume, panel.mask, 60)


@register_legacy_factor("original_006")
def original_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """20-day correlation of close and volume."""
    return ts_corr(panel.close, panel.volume, panel.mask, 20)


# ---------------------------------------------------------------------------
# stock_*  family — per-stock derived series
# ---------------------------------------------------------------------------
@register_legacy_factor("stock_018")
def stock_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """5-day close return minus 20-day close return."""
    c5, m5 = delay(panel.close, panel.mask, 5)
    c20, m20 = delay(panel.close, panel.mask, 20)
    r5  = (panel.close - c5)  / c5.clamp_min(1e-9)
    r20 = (panel.close - c20) / c20.clamp_min(1e-9)
    out = (r5 - r20) * (m5 & m20).float()
    return out, m5 & m20


@register_legacy_factor("stock_022")
def stock_022(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of (high - low) / close over 10 days."""
    rel = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    return ts_rank(rel, panel.mask, 10)


# ---------------------------------------------------------------------------
# extra_*  family — turnover + amount features
# ---------------------------------------------------------------------------
@register_legacy_factor("extra_005")
def extra_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Amount surge: amount / ts_mean(amount, 20) - 1."""
    amt = panel.amount if panel.amount is not None else panel.vwap * panel.volume
    avg, am = ts_mean(amt, panel.mask, 20)
    out = amt / avg.clamp_min(1.0) - 1.0
    return out * am.float(), am


@register_legacy_factor("extra_006")
def extra_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of amount over 20 days."""
    amt = panel.amount if panel.amount is not None else panel.vwap * panel.volume
    return ts_rank(amt, panel.mask, 20)


# ---------------------------------------------------------------------------
# old_*  family
# ---------------------------------------------------------------------------
@register_legacy_factor("old_035")
def old_035(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(open) - rank(close): a simple intraday reversal."""
    a, am = cs_rank(panel.open,  panel.mask)
    b, bm = cs_rank(panel.close, panel.mask)
    return (a - b) * (am & bm).float(), am & bm


# ---------------------------------------------------------------------------
# add_*  family — additions over the original list
# ---------------------------------------------------------------------------
@register_legacy_factor("add_002")
def add_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close vs 5-day VWAP, cross-sectionally ranked."""
    avg, am = ts_mean(panel.vwap, panel.mask, 5)
    out = panel.close / avg.clamp_min(1e-9) - 1.0
    return cs_rank(out * am.float(), am)


@register_legacy_factor("add_020")
def add_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """20-day momentum: close[t] / close[t-20] - 1."""
    prev, pm = delay(panel.close, panel.mask, 20)
    out = panel.close / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("add_021")
def add_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """20-day momentum z-scored cross-sectionally."""
    v, m = add_020(panel)
    return cs_zscore(v, m)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_legacy_set(
    panel: Panel,
    *,
    names: tuple[str, ...] | None = None,
    neutralize: bool = True,
) -> Tuple[Tensor, Tensor, list[str]]:
    """Stack registered legacy factors into a ``[T, N, F]`` tensor."""
    names = names or tuple(LEGACY_REGISTRY)
    cols, joint = [], None
    for n in names:
        v, m = LEGACY_REGISTRY[n](panel)
        # Replace any non-finite cells with 0 on tradable cells; this is
        # the legacy ``process_nan_infinite_and_mask`` contract.
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        if neutralize:
            v, _ = cs_zscore(v, m)
        cols.append(v)
        joint = m if joint is None else joint & m
    factors = torch.stack(cols, dim=-1)
    return factors, joint, list(names)


__all__ = [
    "LEGACY_REGISTRY",
    "register_legacy_factor",
    "compute_legacy_set",
]
