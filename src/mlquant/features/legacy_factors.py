"""Complete port of legacy factor zoo onto masked tensors.

Ports all ~150 factors from ``legacy/features/Feature.py`` to the modern
:mod:`mlquant.features.tensor_factors` primitives, with a uniform
``[T, N]`` mask-aware signature.

Each factor:
  - Takes a :class:`Panel` object
  - Returns ``(values_tensor, mask_tensor)``
  - Uses primitives from tensor_factors.py (ts_mean, cs_rank, delta, etc.)

Factor families:
  - best_*    : close-location momentum variants (21 factors)
  - stock_*   : per-stock derived series (22 factors)
  - extra_*   : turnover + amount features (14 factors)
  - add_*     : additional factor variants (30 factors)
  - change_*  : short-window change-of-velocity (5 factors)
  - original_*: direct close/volume statistics (28 factors)
  - old_*     : legacy alpha signals (50 factors)

All factors are auto-registered into :data:`LEGACY_REGISTRY` via
:func:`register_legacy_factor`. Use :func:`compute_legacy_set` to
compute all or a subset as a stacked ``[T, N, F]`` tensor.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, cs_zscore,
    delay, delta,
    ewma,
    ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum,
)


Tensor = torch.Tensor
LegacyFn = Callable[[Panel], Tuple[Tensor, Tensor]]
LEGACY_REGISTRY: Dict[str, LegacyFn] = {}


def register_legacy_factor(name: str) -> Callable[[LegacyFn], LegacyFn]:
    """Decorator to register a factor function into the global registry."""
    def deco(fn: LegacyFn) -> LegacyFn:
        if name in LEGACY_REGISTRY:
            raise ValueError(f"duplicate legacy factor name: {name}")
        LEGACY_REGISTRY[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(num: Tensor, den: Tensor, eps: float = 1e-12) -> Tensor:
    return num / den.abs().clamp_min(eps).copysign(den + eps)


def _close_loc(panel: Panel) -> Tensor:
    """Where in the day's range did the close land? (-1 = at low, +1 = at high)."""
    rng = (panel.high - panel.low).clamp_min(1e-9)
    return (2.0 * panel.close - panel.high - panel.low) / rng


def _ts_mean_close_loc(panel: Panel, window: int) -> Tuple[Tensor, Tensor]:
    cl = _close_loc(panel)
    return ts_mean(cl, panel.mask, window)


def _ret(panel: Panel) -> Tuple[Tensor, Tensor]:
    """1-day return."""
    d, dm = delta(panel.close, panel.mask, 1)
    return d / panel.close.clamp_min(1e-9), dm


def _amount(panel: Panel) -> Tensor:
    """Amount (turnover) — fallback to vwap * volume."""
    return panel.amount if panel.amount is not None else panel.vwap * panel.volume


# ---------------------------------------------------------------------------
# Import factor groups (each group registers its factors on import)
# ---------------------------------------------------------------------------
from . import _factors_best      # noqa: E402, F401
from . import _factors_stock     # noqa: E402, F401
from . import _factors_extra     # noqa: E402, F401
from . import _factors_add       # noqa: E402, F401
from . import _factors_change    # noqa: E402, F401
from . import _factors_original  # noqa: E402, F401
from . import _factors_old       # noqa: E402, F401
from . import _factors_better    # noqa: E402, F401
from . import _factors_market    # noqa: E402, F401
from . import _factors_alpha101  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_legacy_set(
    panel: Panel,
    *,
    names: tuple[str, ...] | None = None,
    neutralize: bool = True,
) -> Tuple[Tensor, Tensor, list[str]]:
    """Stack registered legacy factors into a ``[T, N, F]`` tensor."""
    names = names or tuple(LEGACY_REGISTRY)
    cols, joint = [], None
    for n in names:
        v, m = LEGACY_REGISTRY[n](panel)
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        if neutralize:
            v, _ = cs_zscore(v, m)
        cols.append(v)
        joint = m if joint is None else joint & m
    factors = torch.stack(cols, dim=-1)
    return factors, joint, list(names)


__all__ = [
    "LEGACY_REGISTRY",
    "register_legacy_factor",
    "compute_legacy_set",
]
