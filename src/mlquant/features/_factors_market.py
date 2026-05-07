"""Market breadth signals: cs_rank_* (6 factors).

Cross-sectional rank of daily price changes relative to previous close.
Ported from legacy/cuda_features.py.
"""
from __future__ import annotations
from typing import Tuple
import torch

from ..data.panel import Panel
from .tensor_factors import cs_rank, delay
from .legacy_factors import register_legacy_factor, _amount

Tensor = torch.Tensor


@register_legacy_factor("cs_rank_close")
def cs_rank_close(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of close return."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = (panel.close / prev.clamp_min(1e-9) - 1.0).clamp(-0.1, 0.1)
    return cs_rank(ret, pm)


@register_legacy_factor("cs_rank_open")
def cs_rank_open(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of open gap."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = (panel.open / prev.clamp_min(1e-9) - 1.0).clamp(-0.1, 0.1)
    return cs_rank(ret, pm)


@register_legacy_factor("cs_rank_high")
def cs_rank_high(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of high return."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = panel.high / prev.clamp_min(1e-9) - 1.0
    return cs_rank(ret, pm)


@register_legacy_factor("cs_rank_low")
def cs_rank_low(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of low return."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = panel.low / prev.clamp_min(1e-9) - 1.0
    return cs_rank(ret, pm)


@register_legacy_factor("cs_rank_avg")
def cs_rank_avg(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of vwap return."""
    prev, pm = delay(panel.close, panel.mask, 1)
    ret = panel.vwap / prev.clamp_min(1e-9) - 1.0
    return cs_rank(ret, pm)


@register_legacy_factor("cs_rank_amount")
def cs_rank_amount(panel: Panel) -> Tuple[Tensor, Tensor]:
    """Cross-sectional rank of amount."""
    amt = _amount(panel)
    return cs_rank(amt, panel.mask)
