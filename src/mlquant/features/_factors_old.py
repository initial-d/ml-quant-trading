"""old_* family: legacy alpha signals (50 factors, old_027 through old_076).

Ported from legacy Feature.py old_027-old_076.
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


@register_legacy_factor("old_027")
def old_027(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(volume) * rank(vwap - close)."""
    vr, vm = cs_rank(panel.volume, panel.mask)
    diff = panel.vwap - panel.close
    dr, dm = cs_rank(diff, panel.mask)
    m = vm & dm
    return (vr * dr) * m.float(), m


@register_legacy_factor("old_028")
def old_028(panel: Panel) -> Tuple[Tensor, Tensor]:
    """corr(rank(open), rank(volume), 10)."""
    ro, om = cs_rank(panel.open, panel.mask)
    rv, vm = cs_rank(panel.volume, panel.mask)
    m = om & vm
    return ts_corr(ro, rv, m, 10)


@register_legacy_factor("old_029")
def old_029(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_min(rank(corr(close, volume, 5)), 12)."""
    c, cm = ts_corr(panel.close, panel.volume, panel.mask, 5)
    r, rm = cs_rank(c * cm.float(), cm)
    return ts_min(r, rm, 12)


@register_legacy_factor("old_030")
def old_030(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sign(Δclose, 5) * (1 - rank(Δvolume, 5))."""
    dc, dm = delta(panel.close, panel.mask, 5)
    dv, dvm = delta(panel.volume, panel.mask, 5)
    m = dm & dvm
    dvr, _ = cs_rank(dv, m)
    out = torch.sign(dc) * (1.0 - dvr)
    return out * m.float(), m


@register_legacy_factor("old_031")
def old_031(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(Δclose, 10) * rank(Δvolume/vol, 10)."""
    dc, dm = delta(panel.close, panel.mask, 10)
    dv, dvm = delta(panel.volume, panel.mask, 10)
    m = dm & dvm
    vol_r = dv / panel.volume.clamp_min(1.0)
    r1, _ = cs_rank(dc, m)
    r2, _ = cs_rank(vol_r, m)
    return (r1 * r2) * m.float(), m


@register_legacy_factor("old_032")
def old_032(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_mean(close, 7) - close + corr(vwap, delay(close,5), 230)."""
    avg, am = ts_mean(panel.close, panel.mask, 7)
    dev = avg - panel.close
    prev, pm = delay(panel.close, panel.mask, 5)
    # Use shorter window (20) as proxy for 230 (data may be short)
    c, cm = ts_corr(panel.vwap, prev, pm, 20)
    m = am & cm
    return (dev + c) * m.float(), m


@register_legacy_factor("old_033")
def old_033(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(-1 + open/close)."""
    out = -1.0 + panel.open / panel.close.clamp_min(1e-9)
    return cs_rank(out, panel.mask)


@register_legacy_factor("old_034")
def old_034(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(1 - rank(std(ret, 2)/std(ret, 5)) + 1 - rank(Δclose, 1))."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    s2, sm2 = ts_std(ret, rm, 2)
    s5, sm5 = ts_std(ret, rm, 5)
    m = sm2 & sm5
    ratio = s2 / s5.clamp_min(1e-9)
    rr, _ = cs_rank(ratio, m)
    dc, dcm = delta(panel.close, panel.mask, 1)
    dcr, _ = cs_rank(dc, dcm)
    m2 = m & dcm
    out = 2.0 - rr - dcr
    return cs_rank(out * m2.float(), m2)


@register_legacy_factor("old_035")
def old_035(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16))."""
    vr, vm = ts_rank(panel.volume, panel.mask, 32)
    x = panel.close + panel.high - panel.low
    xr, xm = ts_rank(x, panel.mask, 16)
    m = vm & xm
    out = vr * (1.0 - xr)
    return out * m.float(), m


@register_legacy_factor("old_036")
def old_036(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(delay(open-close,1), close, 20)) + rank(open-close)."""
    oc = panel.open - panel.close
    oc_d, dm = delay(oc, panel.mask, 1)
    c, cm = ts_corr(oc_d, panel.close, dm, 20)
    rc, rcm = cs_rank(c * cm.float(), cm)
    roc, rocm = cs_rank(oc, panel.mask)
    m = rcm & rocm
    return (rc + roc) * m.float(), m


@register_legacy_factor("old_037")
def old_037(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))."""
    dh, m1 = delay(panel.high, panel.mask, 1)
    dc, m2 = delay(panel.close, panel.mask, 1)
    dl, m3 = delay(panel.low, panel.mask, 1)
    m = m1 & m2 & m3
    r1, _ = cs_rank(panel.open - dh, m)
    r2, _ = cs_rank(panel.open - dc, m)
    r3, _ = cs_rank(panel.open - dl, m)
    out = -(r1 * r2 * r3)
    return out * m.float(), m


@register_legacy_factor("old_038")
def old_038(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-(rank(open) ^ rank(close/vwap))."""
    ro, om = cs_rank(panel.open, panel.mask)
    ratio = panel.close / panel.vwap.clamp_min(1e-9)
    rr, rm = cs_rank(ratio, panel.mask)
    m = om & rm
    out = -(ro.abs().clamp_min(1e-12).pow(rr.clamp(0.01, 2)))
    return out * m.float(), m


@register_legacy_factor("old_039")
def old_039(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-rank(Δclose, 7) * (1 - rank(ewma(volume*ret/close, 1/20)))."""
    dc, dm = delta(panel.close, panel.mask, 7)
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    signal = panel.volume * ret
    ew, em = ewma(signal, rm, alpha=1.0 / 20.0)
    ewr, ewrm = cs_rank(ew, em)
    dcr, dcrm = cs_rank(dc, dm)
    m = dcrm & ewrm
    out = -dcr * (1.0 - ewr)
    return out * m.float(), m


@register_legacy_factor("old_040")
def old_040(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-rank(std(high, 10)) * corr(high, volume, 10)."""
    sd, sm = ts_std(panel.high, panel.mask, 10)
    sdr, sdrm = cs_rank(sd, sm)
    c, cm = ts_corr(panel.high, panel.volume, panel.mask, 10)
    m = sdrm & cm
    out = -sdr * c
    return out * m.float(), m


@register_legacy_factor("old_041")
def old_041(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high * low)^0.5 - vwap."""
    out = (panel.high * panel.low).clamp_min(0).sqrt() - panel.vwap
    return out * panel.mask.float(), panel.mask


@register_legacy_factor("old_042")
def old_042(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(vwap - close) / rank(vwap + close)."""
    r1, m1 = cs_rank(panel.vwap - panel.close, panel.mask)
    r2, m2 = cs_rank(panel.vwap + panel.close, panel.mask)
    m = m1 & m2
    out = r1 / r2.abs().clamp_min(1e-9)
    return out * m.float(), m


@register_legacy_factor("old_043")
def old_043(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank(volume / mean(volume, 20), 20) * ts_rank(-Δclose, 7)."""
    avg, am = ts_mean(panel.volume, panel.mask, 20)
    ratio = panel.volume / avg.clamp_min(1.0)
    vr, vm = ts_rank(ratio, am, 20)
    dc, dm = delta(panel.close, panel.mask, 7)
    dcr, dcm = ts_rank(-dc, dm, 20)
    m = vm & dcm
    return (vr * dcr) * m.float(), m


@register_legacy_factor("old_044")
def old_044(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-corr(high, rank(volume), 5)."""
    rv, vm = cs_rank(panel.volume, panel.mask)
    c, cm = ts_corr(panel.high, rv, vm, 5)
    return (-c) * cm.float(), cm


@register_legacy_factor("old_045")
def old_045(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-rank(sum(delay(close,5), 20)/20) * corr(close, volume, 2)."""
    prev, pm = delay(panel.close, panel.mask, 5)
    s, sm = ts_mean(prev, pm, 20)
    sr, srm = cs_rank(s, sm)
    c, cm = ts_corr(panel.close, panel.volume, panel.mask, 2)
    m = srm & cm
    return (-sr * c) * m.float(), m


@register_legacy_factor("old_046")
def old_046(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/(4*close) - 1."""
    m3, mm3 = ts_mean(panel.close, panel.mask, 3)
    m6, mm6 = ts_mean(panel.close, panel.mask, 6)
    m12, mm12 = ts_mean(panel.close, panel.mask, 12)
    m24, mm24 = ts_mean(panel.close, panel.mask, 24)
    m = mm3 & mm6 & mm12 & mm24
    out = (m3 + m6 + m12 + m24) / (4.0 * panel.close.clamp_min(1e-9)) - 1.0
    return out * m.float(), m


@register_legacy_factor("old_047")
def old_047(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6) + eps) * 100."""
    mx, mxm = ts_max(panel.high, panel.mask, 6)
    mn, mnm = ts_min(panel.low, panel.mask, 6)
    m = mxm & mnm
    rng = (mx - mn).clamp_min(1e-9)
    out = (mx - panel.close) / rng * 100.0
    return out * m.float(), m


@register_legacy_factor("old_048")
def old_048(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(-Δclose, 1) * volume / ewma(volume, 1/20)."""
    dc, dm = delta(panel.close, panel.mask, 1)
    ew, em = ewma(panel.volume, panel.mask, alpha=1.0 / 20.0)
    m = dm & em
    out = -dc * panel.volume / ew.clamp_min(1.0)
    return out * m.float(), m


@register_legacy_factor("old_049")
def old_049(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Conditional: if close<delay(close,1): Δclose else 0 over 20 days sum."""
    prev, pm = delay(panel.close, panel.mask, 1)
    dc = panel.close - prev
    down = (dc * (dc < 0).float()) * pm.float()
    return ts_sum(down, pm, 20)


@register_legacy_factor("old_050")
def old_050(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)."""
    rv, vm = cs_rank(panel.volume, panel.mask)
    rvw, vwm = cs_rank(panel.vwap, panel.mask)
    m = vm & vwm
    c, cm = ts_corr(rv, rvw, m, 5)
    cr, crm = cs_rank(c * cm.float(), cm)
    mx, mxm = ts_max(cr, crm, 5)
    return (-mx) * mxm.float(), mxm


@register_legacy_factor("old_051")
def old_051(panel: Panel) -> Tuple[Tensor, Tensor]:
    """If close<delay(close,1): Δclose else 0 - conditional neg momentum sum 12."""
    prev, pm = delay(panel.close, panel.mask, 1)
    dc = panel.close - prev
    neg = (dc * (dc < 0).float()) * pm.float()
    return ts_sum(neg, pm, 12)


@register_legacy_factor("old_052")
def old_052(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(sum(max(0, high-delay(close,1)), 26) / sum(max(0, delay(close,1)-low), 26) - 1) * 100."""
    prev, pm = delay(panel.close, panel.mask, 1)
    up = torch.clamp(panel.high - prev, min=0)
    dn = torch.clamp(prev - panel.low, min=0)
    su, um = ts_sum(up, pm, 26)
    sd, dm2 = ts_sum(dn, pm, 26)
    m = um & dm2
    out = (su / sd.clamp_min(1e-9) - 1.0) * 100.0
    return out * m.float(), m


@register_legacy_factor("old_053")
def old_053(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_sum(close > delay(close, 1), 12) / 12 * 100."""
    prev, pm = delay(panel.close, panel.mask, 1)
    up = (panel.close > prev).float() * pm.float()
    s, sm = ts_sum(up, pm, 12)
    return (s / 12.0 * 100.0) * sm.float(), sm


@register_legacy_factor("old_054")
def old_054(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(-1 * rank(std(|close-open|,10) + |close-open|) + rank(corr(close,open,10)))."""
    co = (panel.close - panel.open).abs()
    sd, sm = ts_std(co, panel.mask, 10)
    x = sd + co
    xr, xm = cs_rank(x, sm)
    c, cm = ts_corr(panel.close, panel.open, panel.mask, 10)
    cr, crm = cs_rank(c * cm.float(), cm)
    m = xm & crm
    out = -xr + cr
    return out * m.float(), m


@register_legacy_factor("old_055")
def old_055(panel: Panel) -> Tuple[Tensor, Tensor]:
    """corr(rank(close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12)), rank(volume), 6)."""
    mn, mnm = ts_min(panel.low, panel.mask, 12)
    mx, mxm = ts_max(panel.high, panel.mask, 12)
    m1 = mnm & mxm
    rng = (mx - mn).clamp_min(1e-9)
    pos = (panel.close - mn) / rng
    rp, rpm = cs_rank(pos * m1.float(), m1)
    rv, rvm = cs_rank(panel.volume, panel.mask)
    m = rpm & rvm
    return ts_corr(rp, rv, m, 6)


@register_legacy_factor("old_056")
def old_056(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-rank(sum(ret, 10)) * rank(corr(close, volume, 5)) * rank(returns volatility 20)."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    s, sm = ts_sum(ret, rm, 10)
    sr, srm = cs_rank(s, sm)
    c, cm = ts_corr(panel.close, panel.volume, panel.mask, 5)
    cr, crm = cs_rank(c * cm.float(), cm)
    sd, sdm = ts_std(ret, rm, 20)
    sdr, sdrm = cs_rank(sd, sdm)
    m = srm & crm & sdrm
    out = -(sr * cr * sdr)
    return out * m.float(), m


@register_legacy_factor("old_057")
def old_057(panel: Panel) -> Tuple[Tensor, Tensor]:
    """close / ewma(close, 1/30) - 1."""
    ew, em = ewma(panel.close, panel.mask, alpha=1.0 / 30.0)
    out = panel.close / ew.clamp_min(1e-9) - 1.0
    return out * em.float(), em


@register_legacy_factor("old_058")
def old_058(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-volume * (close - delay(close,1))."""
    prev, pm = delay(panel.close, panel.mask, 1)
    out = -panel.volume * (panel.close - prev)
    return out * pm.float(), pm


@register_legacy_factor("old_059")
def old_059(panel: Panel) -> Tuple[Tensor, Tensor]:
    """mean(volume * |ret|, 20) ranked."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = (ret / panel.close.clamp_min(1e-9)).abs()
    signal = panel.volume * ret
    avg, am = ts_mean(signal, rm, 20)
    return cs_rank(avg * am.float(), am)


@register_legacy_factor("old_060")
def old_060(panel: Panel) -> Tuple[Tensor, Tensor]:
    """2 * (rank(open - min(open, 12)) - rank((sum(ret, 60)/60 + 1)^2))."""
    mn, mnm = ts_min(panel.open, panel.mask, 12)
    r1, r1m = cs_rank((panel.open - mn) * mnm.float(), mnm)
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    avg, am = ts_mean(ret, rm, 60)
    sq = (avg + 1.0).pow(2)
    r2, r2m = cs_rank(sq * am.float(), am)
    m = r1m & r2m
    out = 2.0 * (r1 - r2)
    return out * m.float(), m


@register_legacy_factor("old_061")
def old_061(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(vwap - ts_min(vwap, 16))."""
    mn, mm = ts_min(panel.vwap, panel.mask, 16)
    diff = (panel.vwap - mn) * mm.float()
    return cs_rank(diff, mm)


@register_legacy_factor("old_062")
def old_062(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(vwap, sum(mean(volume,20), 23), 5))."""
    avg, am = ts_mean(panel.volume, panel.mask, 20)
    s, sm = ts_sum(avg, am, 23)
    c, cm = ts_corr(panel.vwap, s, sm, 5)
    return cs_rank(c * cm.float(), cm)


@register_legacy_factor("old_063")
def old_063(panel: Panel) -> Tuple[Tensor, Tensor]:
    """max(rank(corr(rank(vwap), rank(volume), 4)), 8)."""
    rv, vm = cs_rank(panel.vwap, panel.mask)
    rvol, volm = cs_rank(panel.volume, panel.mask)
    m = vm & volm
    c, cm = ts_corr(rv, rvol, m, 4)
    cr, crm = cs_rank(c * cm.float(), cm)
    return ts_max(cr, crm, 8)


@register_legacy_factor("old_064")
def old_064(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(sum(open*0.178 + low*0.822, 15), sum(mean(volume,60), 9), 11))."""
    x = panel.open * 0.178 + panel.low * 0.822
    sx, sxm = ts_sum(x, panel.mask, 15)
    avg, am = ts_mean(panel.volume, panel.mask, 60)
    sa, sam = ts_sum(avg, am, 9)
    m = sxm & sam
    c, cm = ts_corr(sx, sa, m, 11)
    return cs_rank(c * cm.float(), cm)


@register_legacy_factor("old_065")
def old_065(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(((open*0.147) + (vwap*0.853)), sum(mean(vol,6), 2), 6))."""
    x = panel.open * 0.147 + panel.vwap * 0.853
    avg, am = ts_mean(panel.volume, panel.mask, 6)
    s, sm = ts_sum(avg, am, 2)
    m = panel.mask & sm
    c, cm = ts_corr(x, s, m, 6)
    return cs_rank(c * cm.float(), cm)


@register_legacy_factor("old_066")
def old_066(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(delta(vwap, 4))."""
    d, dm = delta(panel.vwap, panel.mask, 4)
    return cs_rank(d * dm.float(), dm)


@register_legacy_factor("old_067")
def old_067(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(high - ts_max(high, 6)) / (ts_max(high, 6) + eps) * rank(corr(vwap, mean(vol,20), 4))."""
    mx, mm = ts_max(panel.high, panel.mask, 6)
    dist = (panel.high - mx) / mx.clamp_min(1e-9)
    avg, am = ts_mean(panel.volume, panel.mask, 20)
    c, cm = ts_corr(panel.vwap, avg, am, 4)
    cr, crm = cs_rank(c * cm.float(), cm)
    m = mm & crm
    out = dist * cr
    return out * m.float(), m


@register_legacy_factor("old_068")
def old_068(panel: Panel) -> Tuple[Tensor, Tensor]:
    """(rank(corr(high, mean(volume,15), 9)) * rank(corr(close, volume, 4)))."""
    avg, am = ts_mean(panel.volume, panel.mask, 15)
    c1, cm1 = ts_corr(panel.high, avg, am, 9)
    r1, rm1 = cs_rank(c1 * cm1.float(), cm1)
    c2, cm2 = ts_corr(panel.close, panel.volume, panel.mask, 4)
    r2, rm2 = cs_rank(c2 * cm2.float(), cm2)
    m = rm1 & rm2
    return (r1 * r2) * m.float(), m


@register_legacy_factor("old_069")
def old_069(panel: Panel) -> Tuple[Tensor, Tensor]:
    """sum(max(rank(corr(ts_rank(close,5), ts_rank(volume,5), 4)), 0), 3)."""
    cr, crm = ts_rank(panel.close, panel.mask, 5)
    vr, vrm = ts_rank(panel.volume, panel.mask, 5)
    m = crm & vrm
    c, cm = ts_corr(cr, vr, m, 4)
    rc, rcm = cs_rank(c * cm.float(), cm)
    pos = torch.clamp(rc, min=0)
    return ts_sum(pos, rcm, 3)


@register_legacy_factor("old_070")
def old_070(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(delta(vwap, 2))."""
    d, dm = delta(panel.vwap, panel.mask, 2)
    return cs_rank(d * dm.float(), dm)


@register_legacy_factor("old_071")
def old_071(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank(corr(ts_rank(close, 3), ts_rank(volume, 3), 18), 4)."""
    cr, crm = ts_rank(panel.close, panel.mask, 3)
    vr, vrm = ts_rank(panel.volume, panel.mask, 3)
    m = crm & vrm
    c, cm = ts_corr(cr, vr, m, 18)
    return ts_rank(c * cm.float(), cm, 4)


@register_legacy_factor("old_072")
def old_072(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(ewma(corr(mean(volume,40), low, 4), 1/20)) + rank(ewma(corr(rank(vwap), rank(volume), 4), 1/7))."""
    avg, am = ts_mean(panel.volume, panel.mask, 40)
    c1, cm1 = ts_corr(avg, panel.low, am, 4)
    ew1, em1 = ewma(c1 * cm1.float(), cm1, alpha=1.0 / 20.0)
    r1, rm1 = cs_rank(ew1 * em1.float(), em1)
    rv, vm = cs_rank(panel.vwap, panel.mask)
    rvol, volm = cs_rank(panel.volume, panel.mask)
    m2 = vm & volm
    c2, cm2 = ts_corr(rv, rvol, m2, 4)
    ew2, em2 = ewma(c2 * cm2.float(), cm2, alpha=1.0 / 7.0)
    r2, rm2 = cs_rank(ew2 * em2.float(), em2)
    m = rm1 & rm2
    return (r1 + r2) * m.float(), m


@register_legacy_factor("old_073")
def old_073(panel: Panel) -> Tuple[Tensor, Tensor]:
    """-ts_rank(ewma(Δ(vwap, 5), 1/2), 3)."""
    d, dm = delta(panel.vwap, panel.mask, 5)
    ew, em = ewma(d, dm, alpha=0.5)
    r, rm = ts_rank(ew, em, 3)
    return (-r) * rm.float(), rm


@register_legacy_factor("old_074")
def old_074(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(close, sum(mean(volume,30), 37), 15))."""
    avg, am = ts_mean(panel.volume, panel.mask, 30)
    s, sm = ts_sum(avg, am, 37)
    c, cm = ts_corr(panel.close, s, sm, 15)
    return cs_rank(c * cm.float(), cm)


@register_legacy_factor("old_075")
def old_075(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(corr(rank(vwap), rank(volume), 5)) vs rank(corr(rank(close), rank(volume), 5))."""
    rv, vm = cs_rank(panel.vwap, panel.mask)
    rvol, volm = cs_rank(panel.volume, panel.mask)
    m = vm & volm
    c1, cm1 = ts_corr(rv, rvol, m, 5)
    rc, rcm = cs_rank(panel.close, panel.mask)
    c2, cm2 = ts_corr(rc, rvol, rcm & volm, 5)
    r1, rm1 = cs_rank(c1 * cm1.float(), cm1)
    r2, rm2 = cs_rank(c2 * cm2.float(), cm2)
    m_out = rm1 & rm2
    return (r1 - r2) * m_out.float(), m_out


@register_legacy_factor("old_076")
def old_076(panel: Panel) -> Tuple[Tensor, Tensor]:
    """rank(Δ(corr(vwap, volume, 4), 3))."""
    c, cm = ts_corr(panel.vwap, panel.volume, panel.mask, 4)
    d, dm = delta(c * cm.float(), cm, 3)
    return cs_rank(d * dm.float(), dm)
