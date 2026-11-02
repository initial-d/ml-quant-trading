"""Daily cross-sectional IC with pairwise masks and explicit invalid dates."""
from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class DailyFactorIC:
    ic: torch.Tensor
    rank_ic: torch.Tensor
    count: torch.Tensor
    valid: torch.Tensor


def daily_factor_ic(values, targets, factor_mask, label_mask, *, min_assets=3):
    """Return [T] Pearson/Spearman coefficients across assets, never across dates.

    Values/targets must be aligned [T,N] floating tensors. Masks are separate:
    factor availability is at the signal date; label availability is retrospective
    evaluation eligibility, not a signal or portfolio mask. Nonfinite pairs are
    excluded before ranking, including from tie groups. Too few assets or a
    constant cross-section produces NaN and valid=False, rather than a zero IC.
    FP64 reductions require CPU/CUDA; MPS is unsupported. No gradients retained.
    """
    if type(min_assets) is not int or min_assets < 2:
        raise ValueError("min_assets must be an integer >= 2")
    if values.ndim != 2 or targets.shape != values.shape or values.shape[1] == 0:
        raise ValueError("values and targets must share nonempty [T,N] asset dimensions")
    for tensor in (values, targets, factor_mask, label_mask):
        if tensor.device != values.device or tensor.shape != values.shape:
            raise ValueError("all inputs must have the same shape and device")
    if not values.is_floating_point() or not targets.is_floating_point():
        raise TypeError("values and targets must be floating point")
    if factor_mask.dtype != torch.bool or label_mask.dtype != torch.bool:
        raise TypeError("masks must be boolean")
    if values.device.type == "mps":
        raise ValueError("daily_factor_ic needs FP64 reductions on CPU or CUDA")
    with torch.no_grad():
        paired = factor_mask & label_mask & torch.isfinite(values) & torch.isfinite(targets)
        count = paired.sum(1)
        x = torch.where(paired, values.double(), 0.0)
        y = torch.where(paired, targets.double(), 0.0)

        def correlate(a, b):
            # Scale before centring to avoid overflow in squared magnitudes.
            a = a / a.abs().amax(1, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)
            b = b / b.abs().amax(1, keepdim=True).clamp_min(torch.finfo(b.dtype).tiny)
            n = count.clamp_min(1).unsqueeze(1)
            a = torch.where(paired, a - a.sum(1, keepdim=True) / n, 0.0)
            b = torch.where(paired, b - b.sum(1, keepdim=True) / n, 0.0)
            norm = a.square().sum(1).sqrt() * b.square().sum(1).sqrt()
            valid = (count >= min_assets) & (norm > 0) & torch.isfinite(norm)
            coefficient = (a * b).sum(1) / torch.where(valid, norm, 1.0)
            return coefficient.clamp(-1, 1).masked_fill(~valid, float("nan")), valid

        ic, valid = correlate(x, y)
        def ranks(a):
            sorted_values, index = a.masked_fill(~paired, float("inf")).sort(1)
            starts = torch.cat([torch.zeros_like(sorted_values[:, :1], dtype=torch.bool),
                                sorted_values[:, 1:] != sorted_values[:, :-1]], 1)
            group = starts.long().cumsum(1)
            positions = torch.arange(1, a.shape[1] + 1, device=a.device,
                                     dtype=torch.float64).expand_as(a)
            totals = torch.zeros_like(a).scatter_add(1, group, positions)
            sizes = torch.zeros_like(a).scatter_add(1, group, torch.ones_like(a))
            averaged = (totals / sizes.clamp_min(1)).gather(1, group)
            return torch.zeros_like(a).scatter(1, index, averaged).masked_fill(~paired, 0)

        # Average tie ranks in FP64 on exactly the pairwise-eligible assets.
        rank_ic, rank_valid = correlate(ranks(x), ranks(y))
        valid = valid & rank_valid
        return DailyFactorIC(ic.masked_fill(~valid, float("nan")),
                             rank_ic.masked_fill(~valid, float("nan")), count, valid)
