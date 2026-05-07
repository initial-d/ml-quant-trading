"""add_* family: additional factor variants (30 factors).

Ported from legacy Feature.py add_001-add_030.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, delay, delta, ewma, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum,
)
from .legacy_factors import register_legacy_factor, _close_loc, _amount

Tensor = torch.Tensor


@register_legacy_factor("add_001")
def add_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(Δclose, 5) * rank(Δvolume, 5)."""
    dc, dm1 = delta(panel.close, panel.mask, 5)
    dv, dm2 = delta(panel.volume, panel.mask, 5)
    m = dm1 & dm2
    r1, _ = cs_rank(dc, m)
    r2, _ = cs_rank(dv, m)
    return (r1 * r2) * m.float(), m


@register_legacy_factor("add_002")
def add_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-(close - ts_mean(close, 10)) / ts_mean(close, 10) * rank(volume)."""
    avg, am = ts_mean(panel.close, panel.mask, 10)
    dev = -(panel.close - avg) / avg.clamp_min(1e-9)
    vr, vm = cs_rank(panel.volume, panel.mask)
    m = am & vm
    return (dev * vr) * m.float(), m


@register_legacy_factor("add_003")
def add_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sign(Δclose, 1) * (1 + |Δclose/close|) * Δvolume/volume."""
    dc, dm = delta(panel.close, panel.mask, 1)
    dv, dvm = delta(panel.volume, panel.mask, 1)
    m = dm & dvm
    sign_dc = torch.sign(dc)
    mag = 1.0 + (dc / panel.close.clamp_min(1e-9)).abs()
    vol_chg = dv / panel.volume.clamp_min(1.0)
    return (sign_dc * mag * vol_chg) * m.float(), m


@register_legacy_factor("add_004")
def add_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank(volume, 20) * ts_rank(-close_loc, 10)."""
    vr, vm = ts_rank(panel.volume, panel.mask, 20)
    cl = -_close_loc(panel)
    cr, cm = ts_rank(cl, panel.mask, 10)
    m = vm & cm
    return (vr * cr) * m.float(), m


@register_legacy_factor("add_005")
def add_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_sum(close > delay(close, 1), 12) - ts_sum(close < delay(close, 1), 12)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    up = (panel.close > prev).float() * pm.float()
    dn = (panel.close < prev).float() * pm.float()
    s_up, um = ts_sum(up, pm, 12)
    s_dn, dm2 = ts_sum(dn, pm, 12)
    m = um & dm2
    return (s_up - s_dn) * m.float(), m


@register_legacy_factor("add_006")
def add_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma(close/delay(close,1)-1, 1/12) - close/delay(close,12)-1."""
    prev1, pm1 = delay(panel.close, panel.mask, 1)
    ret1 = (panel.close / prev1.clamp_min(1e-9) - 1.0) * pm1.float()
    ew, em = ewma(ret1, pm1, alpha=1.0 / 12.0)
    prev12, pm12 = delay(panel.close, panel.mask, 12)
    ret12 = (panel.close / prev12.clamp_min(1e-9) - 1.0) * pm12.float()
    m = em & pm12
    return (ew - ret12) * m.float(), m


@register_legacy_factor("add_007")
def add_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """corr(rank(close), rank(volume), 5)."""
    cr, cm = cs_rank(panel.close, panel.mask)
    vr, vm = cs_rank(panel.volume, panel.mask)
    m = cm & vm
    return ts_corr(cr, vr, m, 5)


@register_legacy_factor("add_008")
def add_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_std(close/delay(close,1)-1, 20)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = (panel.close / prev.clamp_min(1e-9) - 1.0) * pm.float()
    return ts_std(ret, pm, 20)


@register_legacy_factor("add_009")
def add_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high + low) / 2 - delay((high + low) / 2, 3)."""
    mid = (panel.high + panel.low) * 0.5
    prev, pm = delay(mid, panel.mask, 3)
    return (mid - prev) * pm.float(), pm


@register_legacy_factor("add_010")
def add_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_max(high, 5) - ts_min(low, 5) scaled by close."""
    mx, mm = ts_max(panel.high, panel.mask, 5)
    mn, mnm = ts_min(panel.low, panel.mask, 5)
    m = mm & mnm
    out = (mx - mn) / panel.close.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("add_011")
def add_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_corr(vwap, volume, 10)."""
    return ts_corr(panel.vwap, panel.volume, panel.mask, 10)


@register_legacy_factor("add_012")
def add_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(open - mean(open,10)) * rank(abs(close-vwap))."""
    avg_o, am = ts_mean(panel.open, panel.mask, 10)
    r1, rm1 = cs_rank((panel.open - avg_o) * am.float(), am)
    diff = (panel.close - panel.vwap).abs()
    r2, rm2 = cs_rank(diff, panel.mask)
    m = rm1 & rm2
    return (r1 * r2) * m.float(), m


@register_legacy_factor("add_013")
def add_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high * low)^0.5 - vwap."""
    out = (panel.high * panel.low).clamp_min(0).sqrt() - panel.vwap
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("add_014")
def add_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """close - delay(close, 5) scaled by delay(close, 5)."""
    prev, pm = delay(panel.close, panel.mask, 5)
    out = (panel.close - prev) / prev.clamp_min(1e-9)
    return out * pm.float(), pm


@register_legacy_factor("add_015")
def add_015(panel: Panel) -> Tuple[Tensor, Tensor]:
    """open / delay(close, 1) - 1 (intraday gap rank)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = (panel.open / prev.clamp_min(1e-9) - 1.0)
    return cs_rank(gap * pm.float(), pm)


@register_legacy_factor("add_016")
def add_016(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_max(vwap, 10) - vwap."""
    mx, mm = ts_max(panel.vwap, panel.mask, 10)
    out = mx - panel.vwap
    return out * mm.float(), mm


@register_legacy_factor("add_017")
def add_017(panel: Panel) -> Tuple[Tensor, Tensor]:
    """vwap - ts_min(vwap, 10)."""
    mn, mm = ts_min(panel.vwap, panel.mask, 10)
    out = panel.vwap - mn
    return out * mm.float(), mm


@register_legacy_factor("add_018")
def add_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean(abs(close - open) / (high - low + 1e-9), 5)."""
    ratio = (panel.close - panel.open).abs() / (panel.high - panel.low).clamp_min(1e-9)
    return ts_mean(ratio, panel.mask, 5)


@register_legacy_factor("add_019")
def add_019(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sum(close > open, 5) / 5 - bullishness ratio."""
    bull = (panel.close > panel.open).float()
    s, sm = ts_sum(bull, panel.mask, 5)
    return (s / 5.0) * sm.float(), sm


@register_legacy_factor("add_020")
def add_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """std(close, 10) / mean(close, 10) - coefficient of variation."""
    sd, sm = ts_std(panel.close, panel.mask, 10)
    avg, am = ts_mean(panel.close, panel.mask, 10)
    m = sm & am
    out = sd / avg.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("add_021")
def add_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(close, volume, 20))."""
    c, cm = ts_corr(panel.close, panel.volume, panel.mask, 20)
    return cs_rank(c * cm.float(), cm)


@register_legacy_factor("add_022")
def add_022(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma(volume, 2/6) / ewma(volume, 2/24) - volume wave."""
    ew1, m1 = ewma(panel.volume, panel.mask, alpha=2.0 / 6.0)
    ew2, m2 = ewma(panel.volume, panel.mask, alpha=2.0 / 24.0)
    m = m1 & m2
    out = ew1 / ew2.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("add_023")
def add_023(panel: Panel) -> Tuple[Tensor, Tensor]:
    """close - ts_mean(close, 5) (deviation from 5-day MA)."""
    avg, am = ts_mean(panel.close, panel.mask, 5)
    out = (panel.close - avg) / panel.close.clamp_min(1e-9)
    return out * am.float(), am


@register_legacy_factor("add_024")
def add_024(panel: Panel) -> Tuple[Tensor, Tensor]:
    """close - ts_mean(close, 20) (deviation from 20-day MA)."""
    avg, am = ts_mean(panel.close, panel.mask, 20)
    out = (panel.close - avg) / panel.close.clamp_min(1e-9)
    return out * am.float(), am


@register_legacy_factor("add_025")
def add_025(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean(close, 5) / mean(close, 20)."""
    m5, mm5 = ts_mean(panel.close, panel.mask, 5)
    m20, mm20 = ts_mean(panel.close, panel.mask, 20)
    m = mm5 & mm20
    out = m5 / m20.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("add_026")
def add_026(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_std(volume, 5) / ts_mean(volume, 5) (volume CV)."""
    sd, sm = ts_std(panel.volume, panel.mask, 5)
    avg, am = ts_mean(panel.volume, panel.mask, 5)
    m = sm & am
    out = sd / avg.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("add_027")
def add_027(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(-delta(close, 3) * Δvolume)."""
    dc, dm1 = delta(panel.close, panel.mask, 3)
    dv, dm2 = delta(panel.volume, panel.mask, 3)
    m = dm1 & dm2
    out = -dc * dv
    return cs_rank(out * m.float(), m)


@register_legacy_factor("add_028")
def add_028(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close - open) / (high - low + 1e-9)."""
    out = (panel.close - panel.open) / (panel.high - panel.low).clamp_min(1e-9)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("add_029")
def add_029(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_mean((close - open) / (high - low + eps), 10)."""
    ratio = (panel.close - panel.open) / (panel.high - panel.low).clamp_min(1e-9)
    return ts_mean(ratio, panel.mask, 10)


@register_legacy_factor("add_030")
def add_030(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(sum(ret, 5)) * rank(sum(ret, 20))."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    s5, m5 = ts_sum(ret, rm, 5)
    s20, m20 = ts_sum(ret, rm, 20)
    m = m5 & m20
    r1, _ = cs_rank(s5, m)
    r2, _ = cs_rank(s20, m)
    return (r1 * r2) * m.float(), m
