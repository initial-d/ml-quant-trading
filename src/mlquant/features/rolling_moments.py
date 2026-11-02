"""Opt-in O(T*N) trailing moments, without an expanded window dimension.

Float64 block prefixes/suffixes reduce cancellation for price levels and
series with large outliers. Each window combines at most two disjoint block
summaries, centred at an observation included in that piece. Results keep the
input dtype/device. Memory and arithmetic are O(T*N), independent of window
length; FP64 throughput (especially on GPUs) is a tradeoff. Extreme within-
window dynamic ranges can still lose accuracy: use direct FP64 windows as a
reference in that regime.

Masked or nonfinite observations are excluded. A full calendar window must
exist, but its observations may be partial: mean needs one, sample std/cov and
Pearson corr need two. Invalid outputs and constant-window correlations are 0.
These functions do not change the default factor implementations.
"""
from __future__ import annotations

import torch


def _validate(x, mask, window, y=None):
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    if x.ndim != 2 or mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError("x and boolean mask must have matching [T, N] shapes")
    if not x.is_floating_point() or mask.device != x.device:
        raise ValueError("x must be floating point on the mask device")
    if x.device.type == "mps":
        raise ValueError("linear moments need float64; use CPU or CUDA, not MPS")
    if y is not None and (y.shape != x.shape or y.dtype != x.dtype or y.device != x.device):
        raise ValueError("y must match x shape, dtype and device")


def _summary(x, valid, kind, y=None, *, reverse=False):
    # x is [blocks, window, N]: a reshape of O(T*N) input, not an expanded
    # rolling-window view. The prefix includes its first valid anchor; the
    # suffix includes its last valid anchor. Empty pieces have zero statistics.
    first = (valid.flip(1) if reverse else valid).to(torch.int64).argmax(1, keepdim=True)
    if reverse:
        first = valid.shape[1] - 1 - first

    def cumulative(a):
        return a.flip(1).cumsum(1).flip(1) if reverse else a.cumsum(1)

    def centered(a):
        anchor = a.gather(1, first).masked_fill(~valid.any(1, keepdim=True), 0)
        return (a.masked_fill(~valid, 0) - anchor).masked_fill(~valid, 0), anchor

    count = cumulative(valid.to(torch.float64))
    n = count.clamp_min(1)
    cx, ax = centered(x)
    sx = cumulative(cx)
    stats = {"count": count, "mx": (sx / n + ax).masked_fill(count == 0, 0)}
    if kind in ("std", "corr"):
        stats["vx"] = (cumulative(cx.square()) - sx.square() / n).clamp_min(0)
    if y is not None:
        cy, ay = centered(y)
        sy = cumulative(cy)
        stats["my"] = (sy / n + ay).masked_fill(count == 0, 0)
        stats["cross"] = cumulative(cx * cy) - sx * sy / n
        if kind == "corr":
            stats["vy"] = (cumulative(cy.square()) - sy.square() / n).clamp_min(0)
    return stats


def _previous_suffix(a):
    # At row r of block b, the trailing window combines prefix [0,r] of b
    # with suffix [r+1,window-1] of b-1. No total-prefix subtraction occurs.
    tail = torch.cat([a[:-1, 1:], torch.zeros_like(a[:-1, :1])], dim=1)
    return torch.cat([torch.zeros_like(a[:1]), tail], dim=0)


def _moments(x, mask, window, kind, y=None):
    _validate(x, mask, window, y)
    if x.shape[0] < window or x.shape[1] == 0:
        return torch.zeros_like(x), torch.zeros_like(mask)
    valid = mask & torch.isfinite(x)
    if y is not None:
        valid = valid & torch.isfinite(y)
    padding = (-x.shape[0]) % window

    def blocks(a):
        a = torch.cat([a, a.new_zeros((padding, a.shape[1]))], dim=0)
        return a.reshape(-1, window, a.shape[1])

    xb, vb = blocks(x.to(torch.float64)), blocks(valid)
    yb = blocks(y.to(torch.float64)) if y is not None else None
    prefix = _summary(xb, vb, kind, yb)
    suffix = {key: _previous_suffix(value) for key, value in
              _summary(xb, vb, kind, yb, reverse=True).items()}
    count = prefix["count"] + suffix["count"]
    n = count.clamp_min(1)
    dx = suffix["mx"] - prefix["mx"]
    weight = prefix["count"] * suffix["count"] / n
    if kind == "mean":
        values = prefix["mx"] + dx * suffix["count"] / n
        out_mask = count >= 1
    else:
        out_mask = count >= 2
        if kind in ("std", "corr"):
            vx = prefix["vx"] + suffix["vx"] + dx.square() * weight
        if kind == "std":
            values = (vx / (n - 1).clamp_min(1)).sqrt()
        else:
            dy = suffix["my"] - prefix["my"]
            cross = prefix["cross"] + suffix["cross"] + dx * dy * weight
            if kind == "cov":
                values = cross / (n - 1).clamp_min(1)
            else:
                vy = prefix["vy"] + suffix["vy"] + dy.square() * weight
                denom = (vx * vy).clamp_min(1e-24).sqrt()
                values = (cross / denom).clamp(-1, 1)
                values = values.masked_fill((vx == 0) | (vy == 0), 0)
    out_mask = out_mask.reshape(-1, x.shape[1])[:x.shape[0]]
    out_mask[:window - 1] = False
    values = values.reshape(-1, x.shape[1])[:x.shape[0]]
    return values.masked_fill(~out_mask, 0).to(x.dtype), out_mask


def ts_mean_linear(x, mask, window):
    """Trailing finite-observation mean using centred FP64 block summaries."""
    return _moments(x, mask, window, "mean")


def ts_std_linear(x, mask, window):
    """Trailing sample standard deviation; at least two observations."""
    return _moments(x, mask, window, "std")


def ts_cov_linear(x, y, mask, window):
    """Trailing sample covariance over jointly finite, unmasked pairs."""
    return _moments(x, mask, window, "cov", y)


def ts_corr_linear(x, y, mask, window):
    """Trailing Pearson correlation; constant windows have value zero."""
    return _moments(x, mask, window, "corr", y)
