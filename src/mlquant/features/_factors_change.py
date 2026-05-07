"""change_* family: short-window change-of-velocity signals (5 factors)."""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import cs_rank, delta, ts_mean, ts_sum
from .legacy_factors import register_legacy_factor

Tensor = torch.Tensor


@register_legacy_factor("change_001")
def change_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day return minus its 5-day mean (mean-reversion proxy)."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    avg, am = ts_mean(ret, rm, 5)
    m = rm & am
    return (ret - avg) * m.float(), m


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
