"""Mask-aware, GPU-vectorised factor primitives.

Every primitive takes a value tensor ``x`` of shape ``[T, N]`` plus a
boolean mask of the same shape and returns a ``(values, mask)`` pair.
``mask[t, i] == False`` means cell ``(t, i)`` is non-tradable (halt,
limit-up before fill, pre-IPO, …) — and *the result must not depend on
the value at masked cells*.

Why this shape?
    The whole repo treats markets as wide tensors. Operating in this
    layout means a ``ts_corr`` over 5 000 stocks × 252 days is a single
    fused GPU call, not a Python loop with ``df.rolling`` per stock.

Numerical conventions
    * Float32 throughout. Accumulators promote to float64 only inside
      :func:`ewma` because the recurrence is sensitive.
    * Cross-sectional ranks are reported in the open interval ``(0, 1]``
      with ``ascending=True`` (smaller value → smaller rank).
    * Time-series rolling windows use *trailing* windows: ``window=5``
      at date ``t`` looks at ``t-4 … t``.

This module deliberately avoids any IO; it only depends on torch.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

__all__ = [
    "cs_rank", "cs_zscore", "cs_scale",
    "ts_rank", "ts_sum", "ts_mean", "ts_std",
    "ts_min", "ts_max", "ts_corr", "ts_cov",
    "ts_argmax", "ts_argmin", "ts_product",
    "ewma", "delay", "delta",
    "decay_linear", "signedpower",
]


# =============================================================================
# Cross-sectional primitives  (reduce along stock axis = dim 1)
# =============================================================================
def cs_rank(x: Tensor, mask: Tensor, *, descending: bool = False) -> Tuple[Tensor, Tensor]:
    """Cross-sectional percentile rank in ``(0, 1]``.

    Tied values receive the average rank — this matches ``pandas
    rank(method='average')`` and is what alpha-101 expects.

    Implementation
        1. Replace masked cells with ±inf so ``argsort`` puts them at
           the end, then sort along ``dim=1``.
        2. Identify tie-groups by ``cumsum`` of strictly-greater jumps.
        3. Average ranks within each group via ``scatter_add``.
        4. Scatter back to original positions and divide by ``mask.sum``
           to get a percentile.
    """
    fill = float("-inf") if descending else float("inf")
    masked = x.masked_fill(~mask, fill)
    sorted_x, idx = torch.sort(masked, dim=1, descending=descending)

    # Tie groups: a new group starts where consecutive values differ.
    diffs = sorted_x[:, 1:] - sorted_x[:, :-1]
    new_group = (diffs > 0) if not descending else (diffs < 0)
    group_id = F.pad(new_group.long(), (1, 0)).cumsum(dim=1)

    # Average rank within each tie group.
    ranks = torch.arange(1, x.shape[1] + 1, device=x.device).expand_as(group_id).float()
    sum_ranks   = torch.zeros_like(ranks).scatter_add(1, group_id, ranks)
    count_ranks = torch.zeros_like(ranks).scatter_add(1, group_id, torch.ones_like(ranks))
    mean_ranks  = sum_ranks / count_ranks.clamp_min(1.0)
    avg_rank    = torch.gather(mean_ranks, 1, group_id)

    # Scatter back to original column order and normalise.
    out = torch.zeros_like(avg_rank).scatter(1, idx, avg_rank)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    out = (out / denom).masked_fill(~mask, 0.0)
    return out, mask


def cs_zscore(x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Cross-sectional z-score with masked-aware mean/std."""
    n = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    xm = x.masked_fill(~mask, 0.0)
    mean = xm.sum(dim=1, keepdim=True) / n
    var  = ((xm - mean) ** 2 * mask).sum(dim=1, keepdim=True) / n.clamp_min(2)
    std  = var.sqrt().clamp_min(1e-12)
    out  = ((x - mean) / std).masked_fill(~mask, 0.0)
    return out, mask


# =============================================================================
# Time-series primitives  (rolling along date axis = dim 0)
# =============================================================================
def _unfold_t(x: Tensor, window: int) -> Tensor:
    """Materialise rolling windows -> shape [T-window+1, window, N]."""
    return x.unfold(0, window, 1).permute(0, 2, 1)


def _unfold_mask(mask: Tensor, window: int) -> Tensor:
    return mask.unfold(0, window, 1).permute(0, 2, 1)


def _pad_front(out: Tensor, window: int, value: float = 0.0) -> Tensor:
    """Front-pad ``window-1`` rows with ``value`` so output shape == input shape."""
    pad = torch.full((window - 1, *out.shape[1:]), value, dtype=out.dtype, device=out.device)
    return torch.cat([pad, out], dim=0)


def ts_sum(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    mw = _unfold_mask(mask, window)
    out_mask = mw.all(dim=1)
    out = xw.sum(dim=1)
    return _pad_front(out, window), _pad_front(out_mask.bool(), window, 0).bool()


def ts_mean(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    mw = _unfold_mask(mask, window).float()
    n  = mw.sum(dim=1).clamp_min(1.0)
    out = xw.sum(dim=1) / n
    out_mask = (mw.sum(dim=1) > 0)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def ts_std(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    mw = _unfold_mask(mask, window).float()
    n  = mw.sum(dim=1).clamp_min(2.0)
    mean = xw.sum(dim=1, keepdim=True) / n.unsqueeze(1)
    var = (((xw - mean) ** 2) * mw).sum(dim=1) / (n - 1)
    out_mask = (mw.sum(dim=1) >= 2)
    return _pad_front(var.sqrt(), window), _pad_front(out_mask, window, 0).bool()


def ts_min(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, float("inf")), window)
    mw = _unfold_mask(mask, window)
    out = xw.min(dim=1).values
    out_mask = mw.any(dim=1)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def ts_max(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, float("-inf")), window)
    mw = _unfold_mask(mask, window)
    out = xw.max(dim=1).values
    out_mask = mw.any(dim=1)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def ts_rank(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """Rank of the *last* element in a trailing window, normalised to ``(0, 1]``."""
    xw = _unfold_t(x.masked_fill(~mask, float("-inf")), window)
    last = xw[:, -1:, :]
    rank = (xw <= last).sum(dim=1).float()
    n = _unfold_mask(mask, window).float().sum(dim=1).clamp_min(1.0)
    out = rank / n
    out_mask = _unfold_mask(mask, window).all(dim=1)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def ts_corr(x: Tensor, y: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """Rolling Pearson correlation of two series, masked-aware."""
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    yw = _unfold_t(y.masked_fill(~mask, 0.0), window)
    mw = _unfold_mask(mask, window).float()
    n = mw.sum(dim=1, keepdim=True).clamp_min(2.0)

    mx = (xw * mw).sum(dim=1, keepdim=True) / n
    my = (yw * mw).sum(dim=1, keepdim=True) / n
    cx = (xw - mx) * mw
    cy = (yw - my) * mw
    cov = (cx * cy).sum(dim=1)
    sx  = (cx ** 2).sum(dim=1).sqrt()
    sy  = (cy ** 2).sum(dim=1).sqrt()
    out = cov / (sx * sy).clamp_min(1e-12)
    out_mask = (mw.sum(dim=1) >= 2)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def ts_cov(x: Tensor, y: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    yw = _unfold_t(y.masked_fill(~mask, 0.0), window)
    mw = _unfold_mask(mask, window).float()
    n = mw.sum(dim=1, keepdim=True).clamp_min(2.0)
    mx = (xw * mw).sum(dim=1, keepdim=True) / n
    my = (yw * mw).sum(dim=1, keepdim=True) / n
    out = ((xw - mx) * (yw - my) * mw).sum(dim=1) / (n.squeeze(1) - 1)
    out_mask = (mw.sum(dim=1) >= 2)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


# =============================================================================
# Other primitives
# =============================================================================
def ewma(x: Tensor, mask: Tensor, alpha: float) -> Tuple[Tensor, Tensor]:
    """Mask-aware exponentially-weighted moving average.

    The recurrence is computed in float64 because the simple recurrence
    ``y_t = α x_t + (1-α) y_{t-1}`` accumulates round-off error in
    float32 over ~3000 steps.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    x64  = x.to(torch.float64).masked_fill(~mask, 0.0)
    m64  = mask.to(torch.float64)
    out  = torch.zeros_like(x64)
    last = torch.zeros(x64.shape[1], dtype=torch.float64, device=x.device)
    for t in range(x64.shape[0]):
        last = alpha * x64[t] + (1.0 - alpha) * last
        out[t] = last
    return out.to(x.dtype) * mask.float(), mask


def delay(x: Tensor, mask: Tensor, periods: int) -> Tuple[Tensor, Tensor]:
    """``x[t-periods]`` along the date axis. Front-pads with zeros.

    Values at originally-masked cells are zeroed *before* shifting, so the
    output never depends on garbage at non-tradable cells (see module
    docstring contract).
    """
    if periods < 0:
        raise ValueError("periods must be non-negative")
    if periods == 0:
        return x * mask.float(), mask
    x_masked = x.masked_fill(~mask, 0.0)
    out = torch.roll(x_masked, shifts=periods, dims=0)
    out[:periods] = 0.0
    new_mask = torch.roll(mask, shifts=periods, dims=0)
    new_mask[:periods] = False
    return out, new_mask


def delta(x: Tensor, mask: Tensor, periods: int) -> Tuple[Tensor, Tensor]:
    """``x[t] - x[t-periods]``."""
    prev, prev_mask = delay(x, mask, periods)
    return (x - prev) * (mask & prev_mask).float(), mask & prev_mask


# =============================================================================
# WorldQuant-101 helpers
# =============================================================================
def cs_scale(x: Tensor, mask: Tensor, a: float = 1.0) -> Tuple[Tensor, Tensor]:
    """``a * x / sum(|x|)`` cross-sectionally — the WQ ``scale`` primitive.

    Masked cells are treated as 0 in both numerator and denominator so they
    do not influence the L1 norm.
    """
    xm = x.masked_fill(~mask, 0.0)
    denom = xm.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    out = (a * xm / denom).masked_fill(~mask, 0.0)
    return out, mask


def signedpower(x: Tensor, p: float) -> Tensor:
    """``sign(x) * |x|**p`` — the WQ ``SignedPower`` primitive."""
    return torch.sign(x) * x.abs().clamp_min(1e-12).pow(p)


def ts_argmax(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """1-based position of the max within a trailing ``window``."""
    xw = _unfold_t(x.masked_fill(~mask, float("-inf")), window)
    pos = xw.argmax(dim=1).float() + 1.0
    out_mask = _unfold_mask(mask, window).any(dim=1)
    return _pad_front(pos, window), _pad_front(out_mask, window, 0).bool()


def ts_argmin(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """1-based position of the min within a trailing ``window``."""
    xw = _unfold_t(x.masked_fill(~mask, float("inf")), window)
    pos = xw.argmin(dim=1).float() + 1.0
    out_mask = _unfold_mask(mask, window).any(dim=1)
    return _pad_front(pos, window), _pad_front(out_mask, window, 0).bool()


def ts_product(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """Rolling product over a trailing ``window``. Masked cells -> 1."""
    xw = _unfold_t(x.masked_fill(~mask, 1.0), window)
    out = xw.prod(dim=1)
    out_mask = _unfold_mask(mask, window).all(dim=1)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()


def decay_linear(x: Tensor, mask: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """Linear-decay weighted moving average — the WQ ``decay_linear`` primitive.

    ``y_t = sum_{k=0..d-1} (d - k) * x_{t-k} / sum_{k=0..d-1}(d - k)``
    Masked cells are treated as 0 (their weight effectively drops out).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    xw = _unfold_t(x.masked_fill(~mask, 0.0), window)
    weights = torch.arange(window, 0, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    out = (xw * weights.view(1, window, 1)).sum(dim=1)
    out_mask = _unfold_mask(mask, window).all(dim=1)
    return _pad_front(out, window), _pad_front(out_mask, window, 0).bool()
