"""extra_* family: turnover + amount features (14 factors).

Ported from legacy Feature.py extra_001-extra_014.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, delay, delta, ewma, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum,
)
from .legacy_factors import register_legacy_factor, _close_loc, _amount

Tensor = torch.Tensor


@register_legacy_factor("extra_001")
def extra_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Delta(close_loc, 1)."""
    cl = -_close_loc(panel)
    return delta(cl, panel.mask, 1)


@register_legacy_factor("extra_002")
def extra_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(max(vwap-close, 3)) + rank(min(vwap-close, 3)) * rank(Δvolume, 3)."""
    spread = panel.vwap - panel.close
    mx, mm1 = ts_max(spread, panel.mask, 3)
    mn, mm2 = ts_min(spread, panel.mask, 3)
    dv, dm = delta(panel.volume, panel.mask, 3)
    m = mm1 & mm2 & dm
    mx_r, _ = cs_rank(mx, m)
    mn_r, _ = cs_rank(mn, m)
    dv_r, _ = cs_rank(dv, m)
    out = mx_r + mn_r * dv_r
    return out * m.float(), m


@register_legacy_factor("extra_003")
def extra_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """open / delay(close, 1) - 1 (overnight gap)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    out = panel.open / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("extra_004")
def extra_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Volume-weighted close-loc EWMA difference (AD oscillator)."""
    cl = _close_loc(panel)
    data = panel.volume * cl
    x, m1 = ewma(data, panel.mask, alpha=2.0 / 9.0)
    y, m2 = ewma(data, panel.mask, alpha=2.0 / 4.0)
    m = m1 & m2
    return (x - y) * m.float(), m


@register_legacy_factor("extra_005")
def extra_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Amount surge: amount / ts_mean(amount, 20) - 1."""
    amt = _amount(panel)
    avg, am = ts_mean(amt, panel.mask, 20)
    out = amt / avg.clamp_min(1.0) - 1.0
    return out * am.float(), am


@register_legacy_factor("extra_006")
def extra_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank of volume / mean(volume, 20)."""
    avg, am = ts_mean(panel.volume, panel.mask, 20)
    ratio = panel.volume / avg.clamp_min(1.0)
    return cs_rank(ratio * am.float(), am)


@register_legacy_factor("extra_007")
def extra_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma(high-low, 2/5) / ewma(ewma(high-low, 2/5), 2/20) - range expansion."""
    rng = panel.high - panel.low
    sma1, m1 = ewma(rng, panel.mask, alpha=2.0 / 5.0)
    sma2, m2 = ewma(sma1, m1, alpha=2.0 / 20.0)
    m = m1 & m2
    out = sma1 / sma2.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("extra_008")
def extra_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sum(max(high-delay(close,1), 0), 5) / sum(max(delay(close,1)-low, 0), 10) * 100."""
    prev, pm = delay(panel.close, panel.mask, 1)
    up = torch.clamp(panel.high - prev, min=0.0)
    down = torch.clamp(prev - panel.low, min=0.0)
    s_up, um = ts_sum(up, pm, 5)
    s_down, dm = ts_sum(down, pm, 10)
    m = um & dm
    out = s_up * 100.0 / s_down.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("extra_009")
def extra_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional volatility: ewma(std(close,20) * (close<=delay(close,1)), 1/5) - ewma(..., 1/20)."""
    sd, sm = ts_std(panel.close, panel.mask, 20)
    prev, pm = delay(panel.close, panel.mask, 1)
    cond = (panel.close <= prev).float() * pm.float()
    signal = sd * cond
    m = sm & pm
    ew1, m1 = ewma(signal, m, alpha=1.0 / 5.0)
    ew2, m2 = ewma(signal, m, alpha=1.0 / 20.0)
    m_out = m1 & m2
    return (ew1 - ew2) * m_out.float(), m_out


@register_legacy_factor("extra_010")
def extra_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """close / delay(close, 5) - 5-day momentum."""
    prev, pm = delay(panel.close, panel.mask, 5)
    out = panel.close / prev.clamp_min(1e-9)
    return out * pm.float(), pm


@register_legacy_factor("extra_011")
def extra_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Raw amount (turnover)."""
    amt = _amount(panel)
    return amt * panel.mask.float(), panel.mask


@register_legacy_factor("extra_012")
def extra_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """high / open."""
    out = panel.high / panel.open.clamp_min(1e-9)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("extra_013")
def extra_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """vwap / close."""
    out = panel.vwap / panel.close.clamp_min(1e-9)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("extra_014")
def extra_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high - low) / close - intraday range."""
    out = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    return out * panel.mask.float(), panel.mask
