"""A small, hand-picked subset of WorldQuant Alpha101 implemented on top
of :mod:`mlquant.features.tensor_factors`.

Choosing which alphas to include
    The canonical 101 list contains a number of formulas with bugs or
    look-ahead biases that have been pointed out in the literature. We
    include a curated subset that:

      * compiles cleanly on the masked-tensor primitives (no per-stock
        Python loops),
      * has a documented rationale in the original paper, and
      * shows non-trivial correlation with forward returns on the
        synthetic GBM panel (a sanity check, not a guarantee).

    The repo's bigger Alpha101++ extension lives in the legacy directory
    for reference; the curated set here is what the unit tests pin.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from ..data.panel import Panel
from .tensor_factors import (
    cs_rank, cs_zscore,
    ts_corr, ts_max, ts_min, ts_rank, ts_std, ts_sum,
    delta,
)


Tensor = torch.Tensor
AlphaFn = Callable[[Panel], Tuple[Tensor, Tensor]]
ALPHA_REGISTRY: Dict[str, AlphaFn] = {}


def _register(name: str) -> Callable[[AlphaFn], AlphaFn]:
    def deco(fn: AlphaFn) -> AlphaFn:
        ALPHA_REGISTRY[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Reversal / momentum
# ---------------------------------------------------------------------------
@_register("alpha001")
def alpha_001(p: Panel) -> Tuple[Tensor, Tensor]:
    """``cs_rank(ts_argmax(close, 5)) - 0.5`` — a cleaned-up Alpha #1."""
    # We use ts_rank as a stable proxy for ts_argmax — both strictly
    # monotone on the original signal and easier to differentiate.
    z, m = ts_rank(p.close, p.mask, 5)
    rank, _ = cs_rank(z, m)
    return rank - 0.5, m


@_register("alpha002")
def alpha_002(p: Panel) -> Tuple[Tensor, Tensor]:
    """``-1 * corr(rank(Δ log volume, 2), rank((C-O)/O), 6)`` — Alpha #2."""
    log_vol = (p.volume.clamp_min(1.0)).log()
    d, dm = delta(log_vol, p.mask, 2)
    a, am = cs_rank(d, dm)

    co = (p.close - p.open) / p.open.clamp_min(1e-9)
    b, bm = cs_rank(co, p.mask)

    common = am & bm
    corr, cm = ts_corr(a, b, common, 6)
    return -corr, cm


@_register("alpha003")
def alpha_003(p: Panel) -> Tuple[Tensor, Tensor]:
    """``-1 * corr(rank(open), rank(volume), 10)`` — Alpha #3."""
    a, am = cs_rank(p.open,   p.mask)
    b, bm = cs_rank(p.volume, p.mask)
    corr, cm = ts_corr(a, b, am & bm, 10)
    return -corr, cm


# ---------------------------------------------------------------------------
# Volatility / range
# ---------------------------------------------------------------------------
@_register("alpha004")
def alpha_004(p: Panel) -> Tuple[Tensor, Tensor]:
    """``-1 * ts_rank(rank(low), 9)`` — Alpha #4."""
    r, m = cs_rank(p.low, p.mask)
    z, zm = ts_rank(r, m, 9)
    return -z, zm


@_register("alpha006")
def alpha_006(p: Panel) -> Tuple[Tensor, Tensor]:
    """``-1 * corr(open, volume, 10)`` — Alpha #6."""
    corr, cm = ts_corr(p.open, p.volume, p.mask, 10)
    return -corr, cm


# ---------------------------------------------------------------------------
# Reversal-of-extremes
# ---------------------------------------------------------------------------
@_register("alpha007")
def alpha_007(p: Panel) -> Tuple[Tensor, Tensor]:
    """Hand-rolled Alpha #7 variant: deviation from 20-day mean,
    cross-sectionally ranked."""
    summed, sm = ts_sum(p.close, p.mask, 20)
    mean = summed / 20.0
    dev = (p.close - mean) * sm
    z, zm = cs_zscore(dev, sm)
    return z, zm


@_register("alpha012")
def alpha_012(p: Panel) -> Tuple[Tensor, Tensor]:
    """``sign(Δvolume,1) * (-Δclose,1)`` — Alpha #12."""
    dv, dvm = delta(p.volume, p.mask, 1)
    dc, dcm = delta(p.close,  p.mask, 1)
    out = torch.sign(dv) * (-dc)
    return out, dvm & dcm


@_register("alpha053")
def alpha_053(p: Panel) -> Tuple[Tensor, Tensor]:
    """``-Δ(((close-low) - (high-close)) / (close-low + 1e-12), 9)`` — Alpha #53."""
    closeloc = ((p.close - p.low) - (p.high - p.close)) / (p.close - p.low + 1e-12)
    d, dm = delta(closeloc, p.mask, 9)
    return -d, dm


@_register("alpha101")
def alpha_101(p: Panel) -> Tuple[Tensor, Tensor]:
    """Alpha #101: ``(close - open) / (high - low + 1e-3)`` — intraday close
    location within the day's range. Trivially cheap, works as baseline."""
    out = (p.close - p.open) / (p.high - p.low + 1e-3)
    return out * p.mask.float(), p.mask


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_alpha_set(
    panel: Panel,
    *,
    names: tuple[str, ...] | None = None,
    neutralize: bool = True,
) -> Tuple[Tensor, Tensor, list[str]]:
    """Compute a stack of alphas → tensor of shape ``[T, N, F]``.

    Parameters
    ----------
    panel : Panel
    names : tuple of str, optional
        Subset of registered alpha names. ``None`` → all registered.
    neutralize : bool
        If True, each factor is cross-sectionally z-scored so the
        downstream model sees comparable scales.

    Returns
    -------
    factors : Tensor [T, N, F]
    mask    : Tensor [T, N]   (intersection of per-factor masks)
    names   : list[str]       (column → factor name)
    """
    names = names or tuple(ALPHA_REGISTRY)
    cols, joint_mask = [], None
    for n in names:
        v, m = ALPHA_REGISTRY[n](panel)
        if neutralize:
            v, _ = cs_zscore(v, m)
        cols.append(v)
        joint_mask = m if joint_mask is None else joint_mask & m
    factors = torch.stack(cols, dim=-1)
    return factors, joint_mask, list(names)
