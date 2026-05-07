"""stock_* family: per-stock derived series (22 factors).

Ported from legacy Feature.py stock001-stock022.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, delay, delta, ewma, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum,
)
from .legacy_factors import register_legacy_factor, _amount

Tensor = torch.Tensor


@register_legacy_factor("stock_001")
def stock_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Corr(Δlog(volume), (close-open)/open, 4)."""
    log_vol = torch.log(panel.volume + 1.0)
    d_lv, dm = delta(log_vol, panel.mask, 1)
    intra_ret = (panel.close - panel.open) / panel.open.clamp_min(1e-9)
    m = dm & panel.mask
    return ts_corr(d_lv, intra_ret, m, 4)


@register_legacy_factor("stock_002")
def stock_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Corr(Δlog(volume), (close-open)/open, 6)."""
    log_vol = torch.log(panel.volume + 1.0)
    d_lv, dm = delta(log_vol, panel.mask, 1)
    intra_ret = (panel.close - panel.open) / panel.open.clamp_min(1e-9)
    m = dm & panel.mask
    return ts_corr(d_lv, intra_ret, m, 6)


@register_legacy_factor("stock_003")
def stock_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(close - ts_max(vwap, 15)) ^ delta(close, 5)."""
    mx, mm = ts_max(panel.vwap, panel.mask, 15)
    diff = (panel.close - mx) * mm.float()
    r, rm = cs_rank(diff, mm)
    dc, dcm = delta(panel.close, panel.mask, 5)
    m = rm & dcm
    # Use sign-preserving power: sign(r) * |r|^|dc/close|
    dc_norm = dc / panel.close.clamp_min(1e-9)
    out = torch.sign(r) * r.abs().clamp_min(1e-12).pow(dc_norm.abs().clamp(0, 2))
    return out * m.float(), m


@register_legacy_factor("stock_004")
def stock_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """EWMA((close - mean(close,6))/mean(close,6) - delay(..., 3), alpha=1/12)."""
    avg, am = ts_mean(panel.close, panel.mask, 6)
    dev = (panel.close - avg) / avg.clamp_min(1e-9)
    dev_d, dm = delay(dev, am, 3)
    diff = (dev - dev_d) * (am & dm).float()
    return ewma(diff, am & dm, alpha=1.0 / 12.0)


@register_legacy_factor("stock_005")
def stock_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close - delay(close, 6)) / delay(close, 6) * (volume + 1)."""
    prev, pm = delay(panel.close, panel.mask, 6)
    out = (panel.close - prev) / prev.clamp_min(1e-9) * (panel.volume + 1.0)
    return out * pm.float(), pm


@register_legacy_factor("stock_006")
def stock_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close - mean(close, 12)) / mean(close, 12)."""
    avg, am = ts_mean(panel.close, panel.mask, 12)
    out = (panel.close - avg) / avg.clamp_min(1e-9)
    return out * am.float(), am


@register_legacy_factor("stock_007")
def stock_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(delay(ts_min(low,5), 5) - ts_min(low,5)) * rank((sum(ret,60)-sum(ret,20))/40) * rank(volume)."""
    mn, mm = ts_min(panel.low, panel.mask, 5)
    mn_d, md = delay(mn, mm, 5)
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    s60, s60m = ts_sum(ret, rm, 60)
    s20, s20m = ts_sum(ret, rm, 20)
    momentum = (s60 - s20) / 40.0
    mom_r, momm = cs_rank(momentum, s60m & s20m)
    vol_r, volm = cs_rank(panel.volume, panel.mask)
    m = md & momm & volm
    out = (mn_d - mn) * mom_r * vol_r
    return out * m.float(), m


@register_legacy_factor("stock_008")
def stock_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-(sum(open,5)*sum(ret,5) - delay(sum(open,5)*sum(ret,5), 10)) ranked."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    so, som = ts_sum(panel.open, panel.mask, 5)
    sr, srm = ts_sum(ret, rm, 5)
    prod = so * sr
    m_prod = som & srm
    prod_d, pdm = delay(prod, m_prod, 10)
    diff = prod - prod_d
    m = m_prod & pdm
    r, rm2 = cs_rank(-diff * m.float(), m)
    return r, rm2


@register_legacy_factor("stock_009")
def stock_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24)) / (4*close)."""
    m3, mm3 = ts_mean(panel.close, panel.mask, 3)
    m6, mm6 = ts_mean(panel.close, panel.mask, 6)
    m12, mm12 = ts_mean(panel.close, panel.mask, 12)
    m24, mm24 = ts_mean(panel.close, panel.mask, 24)
    m = mm3 & mm6 & mm12 & mm24
    out = (m3 + m6 + m12 + m24) * 0.25 / panel.close.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("stock_010")
def stock_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """std(log(amount+1), 6)."""
    amt = _amount(panel)
    log_amt = torch.log(amt + 1.0)
    return ts_std(log_amt, panel.mask, 6)


@register_legacy_factor("stock_011")
def stock_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """CCI-like: (typical - mean(typical,12)) / (0.015 * mean(|close - mean(typical,12)|, 12))."""
    typical = (panel.high + panel.low + panel.close) / 3.0
    avg_t, am = ts_mean(typical, panel.mask, 12)
    dev = (panel.close - avg_t).abs()
    avg_dev, adm = ts_mean(dev, am, 12)
    m = am & adm
    out = (typical - avg_t) / (0.015 * avg_dev.clamp_min(1e-9))
    return out * m.float(), m


@register_legacy_factor("stock_012")
def stock_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """RSI-like: ewma(max(Δvol, 0), 1/6) / ewma(|Δvol|, 1/6) * 100."""
    dv, dm = delta(panel.volume, panel.mask, 1)
    pos = torch.clamp(dv, min=0.0)
    abs_dv = dv.abs()
    ew_pos, m1 = ewma(pos, dm, alpha=1.0 / 6.0)
    ew_abs, m2 = ewma(abs_dv, dm, alpha=1.0 / 6.0)
    m = m1 & m2
    out = ew_pos * 100.0 / ew_abs.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("stock_013")
def stock_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Δ(vwap - close) / Δ(vwap + close)."""
    diff = panel.vwap - panel.close
    summ = panel.vwap + panel.close
    d_diff, dm1 = delta(diff, panel.mask, 1)
    d_sum, dm2 = delta(summ, panel.mask, 1)
    m = dm1 & dm2
    out = d_diff / d_sum.abs().clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("stock_014")
def stock_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close / delay(close, 12) - 1) * volume."""
    prev, pm = delay(panel.close, panel.mask, 12)
    out = (panel.close / prev.clamp_min(1e-9) - 1.0) * panel.volume
    return out * pm.float(), pm


@register_legacy_factor("stock_015")
def stock_015(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional down-vol: sum(|ret|/log(amt+1) * (close<delay(close,1)), 20) / count(down, 20)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret_abs = (panel.close / prev.clamp_min(1e-9) - 1.0).abs()
    amt = _amount(panel)
    log_amt = torch.log(amt + 1.0).clamp_min(1e-9)
    down = (panel.close < prev).float() * pm.float()
    signal = ret_abs / log_amt * down
    s_sum, sm = ts_sum(signal, pm, 20)
    d_count, cm = ts_sum(down, pm, 20)
    m = sm & cm
    out = s_sum / d_count.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("stock_016")
def stock_016(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Williams %R multi-timeframe composite."""
    prev, pm = delay(panel.close, panel.mask, 1)
    true_low = torch.min(panel.low, prev)
    true_high = torch.max(panel.high, prev)
    # 6-day
    sl6, slm6 = ts_sum(true_low, pm, 6)
    sh6, shm6 = ts_sum(true_high - true_low, pm, 6)
    p1 = (panel.close - sl6) / sh6.clamp_min(1e-9) * 12 * 24
    # 12-day
    sl12, slm12 = ts_sum(true_low, pm, 12)
    sh12, shm12 = ts_sum(true_high - true_low, pm, 12)
    p2 = (panel.close - sl12) / sh12.clamp_min(1e-9) * 6 * 24
    # 24-day
    sl24, slm24 = ts_sum(true_low, pm, 24)
    sh24, shm24 = ts_sum(true_high - true_low, pm, 24)
    p3 = (panel.close - sl24) / sh24.clamp_min(1e-9) * 6 * 12
    m = slm6 & slm12 & slm24
    out = (p1 + p2 + p3) * 100.0 / (6*12 + 6*24 + 12*24)
    return out * m.float(), m


@register_legacy_factor("stock_017")
def stock_017(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-ret * mean(volume, 20) * vwap * (high - close)."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    avg_v, am = ts_mean(panel.volume, panel.mask, 20)
    m = rm & am
    out = -ret * avg_v * panel.vwap * (panel.high - panel.close)
    return out * m.float(), m


@register_legacy_factor("stock_018")
def stock_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """5-day close return minus 20-day close return."""
    c5, m5 = delay(panel.close, panel.mask, 5)
    c20, m20 = delay(panel.close, panel.mask, 20)
    r5 = (panel.close - c5) / c5.clamp_min(1e-9)
    r20 = (panel.close - c20) / c20.clamp_min(1e-9)
    m = m5 & m20
    return (r5 - r20) * m.float(), m


@register_legacy_factor("stock_019")
def stock_019(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(low - close) * open^5 / ((close - high) * close^5)."""
    num = -(panel.low - panel.close) * panel.open.pow(5)
    den = (panel.close - panel.high) * panel.close.pow(5)
    out = num / den.abs().clamp_min(1e-12).copysign(den + 1e-12)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("stock_020")
def stock_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(close - delay(close,1)) / delay(close,1) * volume."""
    prev, pm = delay(panel.close, panel.mask, 1)
    out = (panel.close - prev) / prev.clamp_min(1e-9) * panel.volume
    return out * pm.float(), pm


@register_legacy_factor("stock_021")
def stock_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """((high-low) - ewma(high-low, 2/11)) / ewma(high-low, 2/11) * 100."""
    rng = panel.high - panel.low
    sma, sm = ewma(rng, panel.mask, alpha=2.0 / 11.0)
    out = (rng - sma) / sma.clamp_min(1e-9) * 100.0
    return out * sm.float(), sm


@register_legacy_factor("stock_022")
def stock_022(panel: Panel) -> Tuple[Tensor, Tensor]:
    """corr(mean(volume,20), low, 5) + (high+low)/2 - close."""
    avg_v, am = ts_mean(panel.volume, panel.mask, 20)
    corr, cm = ts_corr(avg_v, panel.low, am, 5)
    m = am & cm
    out = corr + (panel.high + panel.low) / 2.0 - panel.close
    return out * m.float(), m
