"""Cross-sectional and industry/size neutralisation.

The simplest and most defensible neutralisation is a per-date
cross-sectional z-score, which removes the market-mean drift while
preserving the cross-sectional ordering — exactly the information a
long-short alpha consumes downstream. We expose two variants:

* :func:`neutralize_cs`        — z-score across all tradable stocks per date.
* :func:`neutralize_industry`  — regress out a one-hot industry matrix
  per date and return the residuals.
"""
from __future__ import annotations

from typing import Optional

import torch

from .tensor_factors import cs_zscore


def neutralize_cs(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-date cross-sectional z-score, masked-aware."""
    out, _ = cs_zscore(x, mask)
    return out


def neutralize_industry(
    x: torch.Tensor,
    mask: torch.Tensor,
    industry_onehot: torch.Tensor,
    *,
    log_size: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-date OLS-residualise ``x`` against industry dummies (+ optionally
    log market cap).

    Parameters
    ----------
    x : Tensor [T, N]
    mask : Tensor [T, N] bool
    industry_onehot : Tensor [T, N, K]   one-hot per stock-day
    log_size : Tensor [T, N], optional   if given, added as a continuous
        regressor.

    Returns
    -------
    Tensor [T, N] — residuals, masked cells set to 0.
    """
    T, N = x.shape
    K = industry_onehot.shape[-1]
    out = torch.zeros_like(x)

    for t in range(T):
        m = mask[t]
        if m.sum() < 2:
            continue
        y = x[t][m]
        Xs = [industry_onehot[t][m].float()]
        if log_size is not None:
            Xs.append(log_size[t][m].unsqueeze(1))
        X = torch.cat(Xs, dim=1)
        # least-squares  Xβ = y   via lstsq for numerical stability
        beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
        residual = y - X @ beta
        # Degeneracy guard: if the signal is almost entirely explained by the
        # regressors (float32 lstsq residual std ∼ 1e-7 for a perfect fit),
        # return zero rather than amplifying tiny noise to unit variance.
        rms_residual = residual.std(unbiased=False)
        if rms_residual < 1e-6:
            out[t][m] = 0.0
            continue
        residual = (residual - residual.mean()) / rms_residual.clamp_min(1e-6)
        out[t][m] = residual

    return out
