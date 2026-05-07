"""original_* family: direct close/volume statistics (28 factors)."""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, cs_zscore, delay, delta, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std,
)
from .legacy_factors import register_legacy_factor

Tensor = torch.Tensor


@register_legacy_factor("original_001")
def original_001(panel: Panel) -> Tuple[Tensor, Tensor]:
    """20-day close volatility, cross-sectionally z-scored."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    sd, sm = ts_std(ret, rm, 20)
    return cs_zscore(sd, sm)


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


@register_legacy_factor("original_007")
def original_007(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of close over 20 days."""
    return ts_rank(panel.close, panel.mask, 20)


@register_legacy_factor("original_008")
def original_008(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of volume over 20 days."""
    return ts_rank(panel.volume, panel.mask, 20)


@register_legacy_factor("original_009")
def original_009(panel: Panel) -> Tuple[Tensor, Tensor]:
    """10-day correlation of close and volume."""
    return ts_corr(panel.close, panel.volume, panel.mask, 10)


@register_legacy_factor("original_010")
def original_010(panel: Panel) -> Tuple[Tensor, Tensor]:
    """5-day correlation of close and volume."""
    return ts_corr(panel.close, panel.volume, panel.mask, 5)


@register_legacy_factor("original_011")
def original_011(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of close over 10 days."""
    return ts_rank(panel.close, panel.mask, 10)


@register_legacy_factor("original_012")
def original_012(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of volume over 10 days."""
    return ts_rank(panel.volume, panel.mask, 10)


@register_legacy_factor("original_013")
def original_013(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of close over 5 days."""
    return ts_rank(panel.close, panel.mask, 5)


@register_legacy_factor("original_014")
def original_014(panel: Panel) -> Tuple[Tensor, Tensor]:
    """ts_rank of volume over 5 days."""
    return ts_rank(panel.volume, panel.mask, 5)


@register_legacy_factor("original_015")
def original_015(panel: Panel) -> Tuple[Tensor, Tensor]:
    """10-day close volatility."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    return ts_std(ret, rm, 10)


@register_legacy_factor("original_016")
def original_016(panel: Panel) -> Tuple[Tensor, Tensor]:
    """5-day close volatility."""
    ret, rm = delta(panel.close, panel.mask, 1)
    ret = ret / panel.close.clamp_min(1e-9)
    return ts_std(ret, rm, 5)


@register_legacy_factor("original_017")
def original_017(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 10-day high distance."""
    hi, hm = ts_max(panel.close, panel.mask, 10)
    dist = panel.close / hi.clamp_min(1e-9) - 1.0
    return dist * hm.float(), hm


@register_legacy_factor("original_018")
def original_018(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 10-day low distance."""
    lo, lm = ts_min(panel.close, panel.mask, 10)
    dist = panel.close / lo.clamp_min(1e-9) - 1.0
    return dist * lm.float(), lm


@register_legacy_factor("original_019")
def original_019(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 5-day high distance."""
    hi, hm = ts_max(panel.close, panel.mask, 5)
    dist = panel.close / hi.clamp_min(1e-9) - 1.0
    return dist * hm.float(), hm


@register_legacy_factor("original_020")
def original_020(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Close to 5-day low distance."""
    lo, lm = ts_min(panel.close, panel.mask, 5)
    dist = panel.close / lo.clamp_min(1e-9) - 1.0
    return dist * lm.float(), lm


@register_legacy_factor("original_021")
def original_021(panel: Panel) -> Tuple[Tensor, Tensor]:
    """10-day momentum: close[t] / close[t-10] - 1."""
    prev, pm = delay(panel.close, panel.mask, 10)
    out = panel.close / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("original_022")
def original_022(panel: Panel) -> Tuple[Tensor, Tensor]:
    """5-day momentum: close[t] / close[t-5] - 1."""
    prev, pm = delay(panel.close, panel.mask, 5)
    out = panel.close / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("original_023")
def original_023(panel: Panel) -> Tuple[Tensor, Tensor]:
    """3-day momentum: close[t] / close[t-3] - 1."""
    prev, pm = delay(panel.close, panel.mask, 3)
    out = panel.close / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("original_024")
def original_024(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day momentum: close[t] / close[t-1] - 1."""
    prev, pm = delay(panel.close, panel.mask, 1)
    out = panel.close / prev.clamp_min(1e-9) - 1.0
    return out * pm.float(), pm


@register_legacy_factor("original_025")
def original_025(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank of 10-day momentum."""
    prev, pm = delay(panel.close, panel.mask, 10)
    mom = panel.close / prev.clamp_min(1e-9) - 1.0
    return cs_rank(mom * pm.float(), pm)


@register_legacy_factor("original_026")
def original_026(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank of 5-day momentum."""
    prev, pm = delay(panel.close, panel.mask, 5)
    mom = panel.close / prev.clamp_min(1e-9) - 1.0
    return cs_rank(mom * pm.float(), pm)


@register_legacy_factor("original_027")
def original_027(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank of 1-day momentum."""
    prev, pm = delay(panel.close, panel.mask, 1)
    mom = panel.close / prev.clamp_min(1e-9) - 1.0
    return cs_rank(mom * pm.float(), pm)


@register_legacy_factor("original_028")
def original_028(panel: Panel) -> Tuple[Tensor, Tensor]:
    """cs_rank of 20-day momentum."""
    prev, pm = delay(panel.close, panel.mask, 20)
    mom = panel.close / prev.clamp_min(1e-9) - 1.0
    return cs_rank(mom * pm.float(), pm)
