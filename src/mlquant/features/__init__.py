"""Tensor-vectorised factor engine + Alpha101-style library.

Design contract
---------------
* Every primitive in :mod:`tensor_factors` accepts and returns a
  ``(values, mask)`` pair; values at masked-out cells are arbitrary
  (typically 0) but must not contaminate aggregates.
* Cross-sectional functions reduce along ``dim=1`` (stock axis).
* Time-series functions slide along ``dim=0`` (date axis) with a
  user-supplied look-back window.
* All operations are torch ops — they run on CPU or GPU and are
  autograd-friendly when needed.
"""
from .tensor_factors import (
    cs_rank,
    cs_zscore,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
    ewma,
    delay,
    delta,
)
from .neutralize import neutralize_cs, neutralize_industry
from .rolling_moments import ts_mean_linear, ts_std_linear, ts_cov_linear, ts_corr_linear
from .rolling_backend import using_rolling_backend
from .bias import apply_limit_mask, board_limit_pct, limit_move_mask
from .alpha101 import compute_alpha_set, ALPHA_REGISTRY
from .legacy_factors import compute_legacy_set, LEGACY_REGISTRY

__all__ = [
    "cs_rank", "cs_zscore",
    "ts_corr", "ts_cov", "ts_max", "ts_mean", "ts_min",
    "ts_rank", "ts_std", "ts_sum", "ewma", "delay", "delta",
    "ts_mean_linear", "ts_std_linear", "ts_cov_linear", "ts_corr_linear",
    "using_rolling_backend",
    "neutralize_cs", "neutralize_industry",
    "limit_move_mask", "board_limit_pct", "apply_limit_mask",
    "compute_alpha_set", "ALPHA_REGISTRY",
    "compute_legacy_set", "LEGACY_REGISTRY",
]
