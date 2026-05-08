"""alpha_* family: curated WorldQuant Alpha101 signals (9 factors).

These factors are adapted from the Alpha101 formulary (Kakushadze, 2016)
and re-implemented on masked-tensor primitives. They cover momentum,
reversal, volume--price correlation, volatility, and intraday patterns.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, cs_zscore, delta, ts_corr, ts_rank, ts_sum,
)
from .legacy_factors import register_legacy_factor

Tensor = torch.Tensor


@register_legacy_factor("alpha_001")
def alpha_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank(ts_rank(close, 5)) - 0.5 — momentum via rank of recent highs."""
    z, m = ts_rank(panel.close, panel.mask, 5)
    rank, _ = cs_rank(z, m)
    return rank - 0.5, m


@register_legacy_factor("alpha_002")
def alpha_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-corr(rank(Δ log volume, 2), rank((C-O)/O), 6) — volume-vs-intraday reversal."""
    log_vol = (panel.volume.clamp_min(1.0)).log()
    d, dm = delta(log_vol, panel.mask, 2)
    a, am = cs_rank(d, dm)

    co = (panel.close - panel.open) / panel.open.clamp_min(1e-9)
    b, bm = cs_rank(co, panel.mask)

    common = am & bm
    corr, cm = ts_corr(a, b, common, 6)
    return -corr, cm


@register_legacy_factor("alpha_003")
def alpha_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-corr(rank(open), rank(volume), 10) — open/volume rank divergence."""
    a, am = cs_rank(panel.open,   panel.mask)
    b, bm = cs_rank(panel.volume, panel.mask)
    corr, cm = ts_corr(a, b, am & bm, 10)
    return -corr, cm


@register_legacy_factor("alpha_004")
def alpha_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-ts_rank(rank(low), 9) — low-quantile mean reversion."""
    r, m = cs_rank(panel.low, panel.mask)
    z, zm = ts_rank(r, m, 9)
    return -z, zm


@register_legacy_factor("alpha_006")
def alpha_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-corr(open, volume, 10) — open/volume rolling correlation."""
    corr, cm = ts_corr(panel.open, panel.volume, panel.mask, 10)
    return -corr, cm


@register_legacy_factor("alpha_007")
def alpha_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Deviation from 20-day mean, cross-sectionally z-scored."""
    summed, sm = ts_sum(panel.close, panel.mask, 20)
    mean = summed / 20.0
    dev = (panel.close - mean) * sm
    z, zm = cs_zscore(dev, sm)
    return z, zm


@register_legacy_factor("alpha_012")
def alpha_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sign(Δvolume) × (−Δclose) — volume-change reversal."""
    dv, dvm = delta(panel.volume, panel.mask, 1)
    dc, dcm = delta(panel.close,  panel.mask, 1)
    out = torch.sign(dv) * (-dc)
    return out, dvm & dcm


@register_legacy_factor("alpha_053")
def alpha_053(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-Δ(close-location, 9) — 9-day change in intraday close position."""
    closeloc = ((panel.close - panel.low) - (panel.high - panel.close)) / (panel.close - panel.low + 1e-12)
    d, dm = delta(closeloc, panel.mask, 9)
    return -d, dm


@register_legacy_factor("alpha_101")
def alpha_101(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close - open) / (high - low + eps) — intraday close location within range."""
    out = (panel.close - panel.open) / (panel.high - panel.low + 1e-3)
    return out * panel.mask.float(), panel.mask
