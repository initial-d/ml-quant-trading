"""better_* family: vwap-based alpha signals (28 factors).

Ported from legacy cuda_features.py better_001-better_028.
These factors heavily use vwap (S_DQ_AVGPRICE) as a reference price.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, delay, delta, ewma, ts_corr, ts_max, ts_mean, ts_min,
    ts_rank, ts_std, ts_sum,
)
from .legacy_factors import register_legacy_factor, _close_loc, _amount

Tensor = torch.Tensor


@register_legacy_factor("better_001")
def better_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(open/delay(close,1) - 1) * log2(volume + 1)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = panel.open / prev.clamp_min(1e-9) - 1.0
    out = gap * torch.log2(panel.volume + 1.0)
    m = pm & panel.mask
    return out * m.float(), m


@register_legacy_factor("better_002")
def better_002(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(open/delay(close,1) - 1) * volume / mean(volume, 5)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = panel.open / prev.clamp_min(1e-9) - 1.0
    avg, am = ts_mean(panel.volume, panel.mask, 5)
    m = pm & am & panel.mask
    out = gap * panel.volume / avg.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("better_003")
def better_003(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(open/delay(close,1)-1) * (vwap-close) * (high-low) / close."""
    prev, pm = delay(panel.close, panel.mask, 1)
    gap = panel.open / prev.clamp_min(1e-9) - 1.0
    spread = (panel.vwap - panel.close) * (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    m = pm & panel.mask
    return (gap * spread) * m.float(), m


@register_legacy_factor("better_004")
def better_004(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-(vwap_loc - delay(vwap_loc, 1)) where vwap_loc = (2*vwap-low-high)/(high-low)."""
    rng = (panel.high - panel.low).clamp_min(1e-9)
    vwap_loc = (2.0 * panel.vwap - panel.low - panel.high) / rng
    prev, pm = delay(vwap_loc, panel.mask, 1)
    out = -(vwap_loc - prev)
    return out * pm.float(), pm


@register_legacy_factor("better_005")
def better_005(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-(vwap_loc - delay(vwap_loc, 5))."""
    rng = (panel.high - panel.low).clamp_min(1e-9)
    vwap_loc = (2.0 * panel.vwap - panel.low - panel.high) / rng
    prev, pm = delay(vwap_loc, panel.mask, 5)
    out = -(vwap_loc - prev)
    return out * pm.float(), pm


@register_legacy_factor("better_006")
def better_006(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean((2*vwap - low - high) / (high - low), 5)."""
    rng = (panel.high - panel.low).clamp_min(1e-9)
    vwap_loc = (2.0 * panel.vwap - panel.low - panel.high) / rng
    return ts_mean(vwap_loc, panel.mask, 5)


@register_legacy_factor("better_007")
def better_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high-low)/close * (volume - delay(volume, 1))."""
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    dv, dm = delta(panel.volume, panel.mask, 1)
    m = dm & panel.mask
    return (rng * dv) * m.float(), m


@register_legacy_factor("better_008")
def better_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high-low)/close / log2(volume + 1)."""
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    out = rng / torch.log2(panel.volume + 1.0).clamp_min(1e-9)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("better_009")
def better_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional sum: if vwap[t]<vwap[t-1]: |vwap/delay(vwap,1)-1|/log(amount+1), window=20."""
    prev, pm = delay(panel.vwap, panel.mask, 1)
    ratio = (panel.vwap / prev.clamp_min(1e-9) - 1.0).abs()
    amt = _amount(panel)
    signal = ratio / torch.log(amt + 1.0).clamp_min(1e-9)
    cond = (panel.vwap < prev).float() * pm.float()
    weighted = signal * cond
    s, sm = ts_sum(weighted, pm, 20)
    cnt, cm = ts_sum(cond, pm, 20)
    m = sm & cm
    out = s / cnt.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("better_010")
def better_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Same as better_009 but window=10."""
    prev, pm = delay(panel.vwap, panel.mask, 1)
    ratio = (panel.vwap / prev.clamp_min(1e-9) - 1.0).abs()
    amt = _amount(panel)
    signal = ratio / torch.log(amt + 1.0).clamp_min(1e-9)
    cond = (panel.vwap < prev).float() * pm.float()
    weighted = signal * cond
    s, sm = ts_sum(weighted, pm, 10)
    cnt, cm = ts_sum(cond, pm, 10)
    m = sm & cm
    out = s / cnt.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("better_011")
def better_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Multi-period support/resistance using min(low, delay(low,1)) and max(high, delay(high,1))."""
    prev_l, plm = delay(panel.low, panel.mask, 1)
    prev_h, phm = delay(panel.high, panel.mask, 1)
    m_base = plm & phm
    lo = torch.minimum(panel.low, prev_l)
    hi = torch.maximum(panel.high, prev_h)
    rng = (hi - lo).clamp_min(1e-9)
    pos_6 = (panel.close - lo) / rng
    pos_12 = pos_6  # simplified: same ratio, different sum windows
    pos_24 = pos_6
    s6, m6 = ts_sum(pos_6, m_base, 6)
    s12, m12 = ts_sum(pos_12, m_base, 12)
    s24, m24 = ts_sum(pos_24, m_base, 24)
    m = m6 & m12 & m24
    out = (s6 / 6.0 + s12 / 12.0 + s24 / 24.0) / 3.0
    return out * m.float(), m


@register_legacy_factor("better_012")
def better_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Multi-period support/resistance using min(low, delay(vwap,1)) and max(high, delay(vwap,1))."""
    prev_v, pvm = delay(panel.vwap, panel.mask, 1)
    lo = torch.minimum(panel.low, prev_v)
    hi = torch.maximum(panel.high, prev_v)
    rng = (hi - lo).clamp_min(1e-9)
    pos = (panel.vwap - lo) / rng
    m_base = pvm & panel.mask
    s6, m6 = ts_sum(pos, m_base, 6)
    s12, m12 = ts_sum(pos, m_base, 12)
    s24, m24 = ts_sum(pos, m_base, 24)
    m = m6 & m12 & m24
    out = (s6 / 6.0 + s12 / 12.0 + s24 / 24.0) / 3.0
    return out * m.float(), m


@register_legacy_factor("better_013")
def better_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(vwap/close - 1) * log2(amount + 1)."""
    amt = _amount(panel)
    out = (panel.vwap / panel.close.clamp_min(1e-9) - 1.0) * torch.log2(amt + 1.0)
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("better_014")
def better_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean((vwap/close - 1) * log2(amount + 1), 4)."""
    amt = _amount(panel)
    signal = (panel.vwap / panel.close.clamp_min(1e-9) - 1.0) * torch.log2(amt + 1.0)
    return ts_mean(signal, panel.mask, 4)


@register_legacy_factor("better_015")
def better_015(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma((vwap - mean(vwap,10))/mean(vwap,10) delta 5, 1/20)."""
    avg, am = ts_mean(panel.vwap, panel.mask, 10)
    dev = (panel.vwap - avg) / avg.clamp_min(1e-9)
    # delta 5
    T = dev.shape[0]
    if T > 5:
        d = dev[5:] - dev[:-5]
        dm = am[5:] & am[:-5]
    else:
        d = torch.zeros_like(dev)
        dm = torch.zeros_like(am)
    ew, em = ewma(d, dm, alpha=1.0 / 20.0)
    # Pad back to [T, N]
    pad = torch.zeros(5, dev.shape[1], device=dev.device)
    pad_m = torch.zeros(5, dev.shape[1], dtype=torch.bool, device=dev.device)
    out = torch.cat([pad, ew], dim=0)
    mask_out = torch.cat([pad_m, em], dim=0)
    return out * mask_out.float(), mask_out


@register_legacy_factor("better_016")
def better_016(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma((vwap - mean(vwap,6))/mean(vwap,6) delta 3, 1/12)."""
    avg, am = ts_mean(panel.vwap, panel.mask, 6)
    dev = (panel.vwap - avg) / avg.clamp_min(1e-9)
    T = dev.shape[0]
    if T > 3:
        d = dev[3:] - dev[:-3]
        dm = am[3:] & am[:-3]
    else:
        d = torch.zeros_like(dev)
        dm = torch.zeros_like(am)
    ew, em = ewma(d, dm, alpha=1.0 / 12.0)
    pad = torch.zeros(3, dev.shape[1], device=dev.device)
    pad_m = torch.zeros(3, dev.shape[1], dtype=torch.bool, device=dev.device)
    out = torch.cat([pad, ew], dim=0)
    mask_out = torch.cat([pad_m, em], dim=0)
    return out * mask_out.float(), mask_out


@register_legacy_factor("better_017")
def better_017(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma((amount - mean(amount,6))/mean(amount,6) delta 3, 1/12)."""
    amt = _amount(panel)
    avg, am = ts_mean(amt, panel.mask, 6)
    dev = (amt - avg) / avg.clamp_min(1.0)
    T = dev.shape[0]
    if T > 3:
        d = dev[3:] - dev[:-3]
        dm = am[3:] & am[:-3]
    else:
        d = torch.zeros_like(dev)
        dm = torch.zeros_like(am)
    ew, em = ewma(d, dm, alpha=1.0 / 12.0)
    pad = torch.zeros(3, dev.shape[1], device=dev.device)
    pad_m = torch.zeros(3, dev.shape[1], dtype=torch.bool, device=dev.device)
    out = torch.cat([pad, ew], dim=0)
    mask_out = torch.cat([pad_m, em], dim=0)
    return out * mask_out.float(), mask_out


@register_legacy_factor("better_018")
def better_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma((amount - mean(amount,10))/mean(amount,10) delta 5, 1/20)."""
    amt = _amount(panel)
    avg, am = ts_mean(amt, panel.mask, 10)
    dev = (amt - avg) / avg.clamp_min(1.0)
    T = dev.shape[0]
    if T > 5:
        d = dev[5:] - dev[:-5]
        dm = am[5:] & am[:-5]
    else:
        d = torch.zeros_like(dev)
        dm = torch.zeros_like(am)
    ew, em = ewma(d, dm, alpha=1.0 / 20.0)
    pad = torch.zeros(5, dev.shape[1], device=dev.device)
    pad_m = torch.zeros(5, dev.shape[1], dtype=torch.bool, device=dev.device)
    out = torch.cat([pad, ew], dim=0)
    mask_out = torch.cat([pad_m, em], dim=0)
    return out * mask_out.float(), mask_out


@register_legacy_factor("better_019")
def better_019(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Triple-smoothed EMA of log(volume) change."""
    lv = torch.log(panel.volume + 1.0)
    ew1, m1 = ewma(lv, panel.mask, alpha=2.0 / 13.0)
    ew2, m2 = ewma(ew1, m1, alpha=2.0 / 13.0)
    ew3, m3 = ewma(ew2, m2, alpha=2.0 / 13.0)
    # Change: ew3[t] / ew3[t-1] - 1
    prev, pm = delay(ew3, m3, 1)
    out = ew3 / prev.clamp_min(1e-9) - 1.0
    m = m3 & pm
    return out * m.float(), m


@register_legacy_factor("better_020")
def better_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """std(|ret|/volume, 10) / mean(|ret|/volume, 10) - Amihud illiquidity CV."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = (panel.close / prev.clamp_min(1e-9) - 1.0).abs()
    signal = ret / panel.volume.clamp_min(1.0)
    sd, sm = ts_std(signal, pm, 10)
    avg, am = ts_mean(signal, pm, 10)
    m = sm & am
    out = sd / avg.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("better_021")
def better_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """vwap / delay(close, 1)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    out = panel.vwap / prev.clamp_min(1e-9)
    return out * pm.float(), pm


@register_legacy_factor("better_022")
def better_022(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(vwap/delay(close,1) - 1) * log2(amount + 1)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ratio = panel.vwap / prev.clamp_min(1e-9) - 1.0
    amt = _amount(panel)
    out = ratio * torch.log2(amt + 1.0)
    m = pm & panel.mask
    return out * m.float(), m


@register_legacy_factor("better_023")
def better_023(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean((vwap/delay(close,1) - 1) * log2(amount + 1), 4)."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ratio = panel.vwap / prev.clamp_min(1e-9) - 1.0
    amt = _amount(panel)
    signal = ratio * torch.log2(amt + 1.0)
    m = pm & panel.mask
    return ts_mean(signal * m.float(), m, 4)


@register_legacy_factor("better_024")
def better_024(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(max(vwap-close, 5) + min(vwap-close, 5)) * Δvolume_5."""
    spread = panel.vwap - panel.close
    mx, mxm = ts_max(spread, panel.mask, 5)
    mn, mnm = ts_min(spread, panel.mask, 5)
    dv, dvm = delta(panel.volume, panel.mask, 5)
    m = mxm & mnm & dvm
    out = (mx + mn) * dv
    return out * m.float(), m


@register_legacy_factor("better_025")
def better_025(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(max(vwap-close, 5) + min(vwap-close, 5)) * (volume - mean(volume, 5))."""
    spread = panel.vwap - panel.close
    mx, mxm = ts_max(spread, panel.mask, 5)
    mn, mnm = ts_min(spread, panel.mask, 5)
    avg, am = ts_mean(panel.volume, panel.mask, 5)
    m = mxm & mnm & am
    out = (mx + mn) * (panel.volume - avg)
    return out * m.float(), m


@register_legacy_factor("better_026")
def better_026(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ewma((high-low)/close, 2/5) / ewma(ewma((high-low)/close, 2/5), 2/20)."""
    rng = (panel.high - panel.low) / panel.close.clamp_min(1e-9)
    sma1, m1 = ewma(rng, panel.mask, alpha=2.0 / 5.0)
    sma2, m2 = ewma(sma1, m1, alpha=2.0 / 20.0)
    m = m1 & m2
    out = sma1 / sma2.clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("better_027")
def better_027(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional sum: if vwap < delay(close,1): |vwap/delay(close,1)-1|/log(amount+1), window=20."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ratio = (panel.vwap / prev.clamp_min(1e-9) - 1.0).abs()
    amt = _amount(panel)
    signal = ratio / torch.log(amt + 1.0).clamp_min(1e-9)
    cond = (panel.vwap < prev).float() * pm.float()
    weighted = signal * cond
    s, sm = ts_sum(weighted, pm, 20)
    cnt, cm = ts_sum(cond, pm, 20)
    m = sm & cm
    out = s / cnt.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("better_028")
def better_028(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional sum: if vwap < close: |vwap/close-1|/log(amount+1), window=20."""
    ratio = (panel.vwap / panel.close.clamp_min(1e-9) - 1.0).abs()
    amt = _amount(panel)
    signal = ratio / torch.log(amt + 1.0).clamp_min(1e-9)
    cond = (panel.vwap < panel.close).float() * panel.mask.float()
    weighted = signal * cond
    s, sm = ts_sum(weighted, panel.mask, 20)
    cnt, cm = ts_sum(cond, panel.mask, 20)
    m = sm & cm
    out = s / cnt.clamp_min(1.0)
    return out * m.float(), m
