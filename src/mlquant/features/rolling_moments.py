"""Opt-in O(T*N) trailing moments, without an expanded window dimension.

Float64, first-observation-centred prefix sums reduce cancellation for price
levels. Results keep the input dtype/device. Memory and arithmetic do not grow
with window length; FP64 throughput (especially on GPUs) is a tradeoff. Very
long, strongly drifting series can still lose accuracy when subtracting prefix
sums: use the direct-window primitives as a reference in that regime.

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


def _window_sum(x, window):
    prefix = torch.cat([torch.zeros_like(x[:1]), x.cumsum(dim=0)], dim=0)
    return prefix[window:] - prefix[:-window]


def _center(x, valid):
    # Before the first observed value the column has no valid output, so this
    # anchor never makes an earlier valid result depend on a later observation.
    first = valid.to(torch.int64).argmax(dim=0, keepdim=True)
    anchor = x.gather(0, first).masked_fill(~valid.any(dim=0, keepdim=True), 0.0)
    safe = x.masked_fill(~valid, 0.0)
    return (safe - anchor).masked_fill(~valid, 0.0), anchor


def _moments(x, mask, window, kind, y=None):
    _validate(x, mask, window, y)
    if x.shape[0] < window or x.shape[1] == 0:
        return torch.zeros_like(x), torch.zeros_like(mask)
    valid = mask & torch.isfinite(x)
    if y is not None:
        valid = valid & torch.isfinite(y)
    count = _window_sum(valid.to(torch.float64), window)
    n = count.clamp_min(1)
    cx, anchor = _center(x.to(torch.float64), valid)
    sx = _window_sum(cx, window)
    if kind == "mean":
        values = sx / n + anchor
        out_mask = count >= 1
    else:
        out_mask = count >= 2
        if kind in ("std", "corr"):
            vx = (_window_sum(cx.square(), window) - sx.square() / n).clamp_min(0)
        if kind == "std":
            values = (vx / (n - 1).clamp_min(1)).sqrt()
        else:
            cy, _ = _center(y.to(torch.float64), valid)
            sy = _window_sum(cy, window)
            cross = _window_sum(cx * cy, window) - sx * sy / n
            if kind == "cov":
                values = cross / (n - 1).clamp_min(1)
            else:
                vy = (_window_sum(cy.square(), window) - sy.square() / n).clamp_min(0)
                denom = (vx * vy).clamp_min(1e-24).sqrt()
                values = (cross / denom).clamp(-1, 1)
                values = values.masked_fill((vx == 0) | (vy == 0), 0)
    values = values.masked_fill(~out_mask, 0).to(x.dtype)
    pad = window - 1
    return (torch.cat([torch.zeros_like(x[:pad]), values]),
            torch.cat([torch.zeros_like(mask[:pad]), out_mask]))


def ts_mean_linear(x, mask, window):
    """Trailing finite-observation mean using centred float64 prefix sums."""
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
