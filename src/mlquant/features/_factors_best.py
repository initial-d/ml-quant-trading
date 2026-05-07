"""best_* family: close-location momentum over varying windows (21 factors).

Legacy template: -(2*C - H - L) / (H - L + eps) with various transforms.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, delay, delta, ewma, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_sum,
)
from .legacy_factors import register_legacy_factor, _close_loc, _ts_mean_close_loc, _amount

Tensor = torch.Tensor


@register_legacy_factor("best_001")
def best_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of -close_loc over 5 days."""
    cl = -_close_loc(panel)
    return ts_rank(cl, panel.mask, 5)


@register_legacy_factor("best_002")
def best_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of -close_loc over 10 days."""
    cl = -_close_loc(panel)
    return ts_rank(cl, panel.mask, 10)


@register_legacy_factor("best_003")
def best_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of -close_loc over 20 days."""
    cl = -_close_loc(panel)
    return ts_rank(cl, panel.mask, 20)


@register_legacy_factor("best_004")
def best_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(close_loc, 3)."""
    cl = -_close_loc(panel)
    return delta(cl, panel.mask, 3)


@register_legacy_factor("best_005")
def best_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(close_loc, 5)."""
    cl = -_close_loc(panel)
    return delta(cl, panel.mask, 5)


@register_legacy_factor("best_006")
def best_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """EWMA of -close_loc, alpha=1/10."""
    cl = -_close_loc(panel)
    return ewma(cl, panel.mask, alpha=1.0 / 10.0)


@register_legacy_factor("best_007")
def best_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of -close_loc."""
    cl = -_close_loc(panel)
    return cs_rank(cl, panel.mask)


@register_legacy_factor("best_008")
def best_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_mean(vwap / close, 3)."""
    ratio = panel.vwap / panel.close.clamp_min(1e-9)
    return ts_mean(ratio, panel.mask, 3)


@register_legacy_factor("best_009")
def best_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """EWMA(vwap, 5) / EWMA(close, 5)."""
    ew_vwap, m1 = ewma(panel.vwap, panel.mask, alpha=1.0 / 5.0)
    ew_close, m2 = ewma(panel.close, panel.mask, alpha=1.0 / 5.0)
    m = m1 & m2
    out = ew_vwap / ew_close.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("best_010")
def best_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(vwap / close - 1) * volume."""
    out = (panel.vwap / panel.close.clamp_min(1e-9) - 1.0) * panel.volume
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("best_011")
def best_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank(open / delay(close, 1) - 1)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = panel.open / prev.clamp_min(1e-9) - 1.0
    return cs_rank(gap * pm.float(), pm)


@register_legacy_factor("best_012")
def best_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_mean(open / delay(close, 1) - 1, 5)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = (panel.open / prev.clamp_min(1e-9) - 1.0) * pm.float()
    return ts_mean(gap, pm, 5)


@register_legacy_factor("best_013")
def best_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """EWMA(open / delay(close, 1) - 1, alpha=1/5)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = (panel.open / prev.clamp_min(1e-9) - 1.0) * pm.float()
    return ewma(gap, pm, alpha=1.0 / 5.0)


@register_legacy_factor("best_014")
def best_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Correlation of ts_rank(volume, 7) and ts_rank(range/close, 7) over 5 days."""
    vol_r, vm = ts_rank(panel.volume, panel.mask, 7)
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    rng_r, rm = ts_rank(rng, panel.mask, 7)
    m = vm & rm
    return ts_corr(rng_r, vol_r, m, 5)


@register_legacy_factor("best_015")
def best_015(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank((high - low) / close)."""
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    return cs_rank(rng, panel.mask)


@register_legacy_factor("best_016")
def best_016(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high - low) / close / (1 + sqrt(volume))."""
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    out = rng / (1.0 + panel.volume.clamp_min(0).sqrt())
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("best_017")
def best_017(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(EWMA(close, alpha=1/5), 1) - moving average breakout."""
    ew, em = ewma(panel.close, panel.mask, alpha=1.0 / 5.0)
    return delta(ew, em, 1)


@register_legacy_factor("best_018")
def best_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(EWMA(close, alpha=1/10), 1)."""
    ew, em = ewma(panel.close, panel.mask, alpha=1.0 / 10.0)
    return delta(ew, em, 1)


@register_legacy_factor("best_019")
def best_019(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(EWMA(close, alpha=1/20), 1)."""
    ew, em = ewma(panel.close, panel.mask, alpha=1.0 / 20.0)
    return delta(ew, em, 1)


@register_legacy_factor("best_020")
def best_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """EWMA((close - vwap) * volume, alpha=1/5) - money flow."""
    x = (panel.close - panel.vwap) * panel.volume
    return ewma(x, panel.mask, alpha=1.0 / 5.0)


@register_legacy_factor("best_021")
def best_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """max((vwap-close)*vol, 3) + min((vwap-close)*vol, 3) * delta(volume, 3)."""
    spread = (panel.vwap - panel.close) * panel.volume
    mx, mm1 = ts_max(spread, panel.mask, 3)
    mn, mm2 = ts_min(spread, panel.mask, 3)
    dv, dm = delta(panel.volume, panel.mask, 3)
    m = mm1 & mm2 & dm
    out = (mx + mn * dv) * m.float()
    return out, m
