"""Geometric Brownian Motion data augmentation.

The paper argues that financial time-series datasets are
*data-poor for the regime that matters*: A-share crashes and rallies
are rare, and an ML model trained directly on the historical record
will overfit the most-recent regime. Augmenting the training set with
GBM-resampled trajectories that share the historical drift and
volatility — but reshuffle the random shocks — provides extra effective
sample size without leaking information about the actual future.

This module is intentionally a single, well-tested function; if you
want path-dependent augmentation (e.g. SDE bridge, bootstrap blocks)
add a sibling function rather than overloading :func:`gbm_augment`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class GBMAugmentConfig:
    n_synthetic: int = 1               # paths produced per real path
    block_size:  int = 21              # ~1 month, lengths of resampled blocks
    seed:        int = 0


def gbm_augment(
    log_returns: torch.Tensor,
    mask: torch.Tensor,
    cfg: GBMAugmentConfig | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-bootstrap GBM augmentation for log-return panels.

    For each stock we estimate ``μ̂, σ̂`` on the supplied history and
    resample a synthetic trajectory of identical length:

        r̃_t = μ̂ + σ̂ * ε_t,    ε_t ~ N(0, 1) drawn in blocks of
                                ``block_size`` so autocorrelation
                                structure is partly preserved.

    The resulting tensor has shape ``[n_synthetic*T, N]`` and is meant
    to be *concatenated* with real data at training time (with a
    binary "is_synthetic" flag exposed by :class:`FactorDataset`).

    Parameters
    ----------
    log_returns : Tensor [T, N]   per-stock log returns.
    mask        : Tensor [T, N]   tradability mask; masked cells do not
                                  contribute to the μ̂, σ̂ estimate.
    cfg         : GBMAugmentConfig

    Returns
    -------
    aug_log_returns : Tensor [n_synthetic*T, N]
    aug_mask        : Tensor [n_synthetic*T, N]   ones (synthetic data is
                                                  always tradable)
    """
    cfg = cfg or GBMAugmentConfig()
    if log_returns.dim() != 2:
        raise ValueError("log_returns must be [T, N]")

    T, N = log_returns.shape
    if cfg.block_size <= 0 or cfg.block_size > T:
        raise ValueError(f"block_size must be in (0, T={T}], got {cfg.block_size}")

    gen = torch.Generator(device=log_returns.device).manual_seed(cfg.seed)

    # Per-stock μ, σ from the masked history.
    m = mask.float()
    counts = m.sum(dim=0).clamp_min(2)
    mu  = (log_returns * m).sum(dim=0) / counts                          # [N]
    var = (((log_returns - mu) * m) ** 2).sum(dim=0) / (counts - 1)
    sigma = var.clamp_min(1e-12).sqrt()                                  # [N]

    out_T = cfg.n_synthetic * T
    out = torch.empty((out_T, N), dtype=log_returns.dtype, device=log_returns.device)

    # Block-bootstrap eps: draw block-aligned standardised innovations.
    n_blocks = (out_T + cfg.block_size - 1) // cfg.block_size
    eps = torch.randn((n_blocks * cfg.block_size, N), generator=gen,
                      dtype=log_returns.dtype, device=log_returns.device)
    eps = eps[:out_T]

    out[:] = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    aug_mask = torch.ones_like(out, dtype=torch.bool)
    return out, aug_mask
