"""Cross-sectional and industry neutralisation tests."""
from __future__ import annotations

import torch
import pytest

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.neutralize import neutralize_cs, neutralize_industry


# =============================================================================
# neutralize_cs
# =============================================================================


def test_neutralize_cs_basic():
    """Per-date cross-sectional z-score: mean≈0, std≈1 on valid cells."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=50, n_dates=60, seed=1))
    out = neutralize_cs(p.close, p.mask)

    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 2:
            continue
        vals = out[t][valid]
        assert torch.abs(vals.mean()) < 1e-6, f"date {t}: mean {vals.mean().item():.6f}"
        assert torch.abs(vals.std() - 1.0) < 0.15, f"date {t}: std {vals.std().item():.6f}"


def test_neutralize_cs_mask():
    """Masked cells are zero in output and unmasked ones are untouched."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=2))
    out = neutralize_cs(p.close, p.mask)
    # masked cells → 0
    assert (out[~p.mask] == 0.0).all()
    # valid cells ≠ 0
    assert (out[p.mask] != 0.0).any()


def test_neutralize_cs_const_values():
    """Constant values across stocks → all-zero output (no cross-sectional variation)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=3))
    p.close[:] = 1.0
    out = neutralize_cs(p.close, p.mask)
    # z-score of identical values is NaN; clamped std gives 0
    assert torch.allclose(out[p.mask], torch.zeros(1), atol=1e-6)


def test_neutralize_cs_single_stock():
    """Single valid stock per date → output is zero (std=0, clamped)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=5, n_dates=5, seed=4))
    mask = torch.zeros_like(p.mask)
    mask[:, 0] = True  # only stock 0 is tradable
    out = neutralize_cs(p.close, mask)
    assert (out == 0.0).all()


def test_neutralize_cs_all_masked():
    """All stocks masked → output is all zero."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=5))
    mask = torch.zeros_like(p.mask)
    out = neutralize_cs(p.close, mask)
    assert (out == 0.0).all()


# =============================================================================
# neutralize_industry
# =============================================================================


def _make_industry_onehot(n_stocks: int, n_industries: int, device: str = "cpu") -> torch.Tensor:
    """Create a fixed per-stock industry one-hot matrix [1, N, K]."""
    onehot = torch.zeros(1, n_stocks, n_industries, device=device)
    for i in range(n_stocks):
        onehot[0, i, i % n_industries] = 1.0
    return onehot


def test_neutralize_industry_basic():
    """Residuals after industry neutralisation have per-date mean≈0."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=6))
    onehot = _make_industry_onehot(30, 5)
    out = neutralize_industry(p.close, p.mask, onehot.expand(p.close.shape[0], -1, -1))

    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 2:
            continue
        vals = out[t][valid]
        assert torch.abs(vals.mean()) < 1e-5, f"date {t}: mean {vals.mean().item():.6f}"


def test_neutralize_industry_with_size():
    """Industry neutralisation with log_size regressor still produces mean≈0."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=7))
    onehot = _make_industry_onehot(30, 5)
    log_size = torch.log(p.close.abs().clamp_min(1e-12))
    out = neutralize_industry(p.close, p.mask, onehot.expand(p.close.shape[0], -1, -1), log_size=log_size)

    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() <= 2:
            continue
        vals = out[t][valid]
        assert torch.abs(vals.mean()) < 1e-5, f"date {t}: mean {vals.mean().item():.6f}"


def test_neutralize_industry_mask():
    """Masked cells are zero in output."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=10, seed=8))
    onehot = _make_industry_onehot(20, 4)
    # mask out first 5 stocks each day
    mask = p.mask.clone()
    mask[:, :5] = False
    out = neutralize_industry(p.close, mask, onehot.expand(p.close.shape[0], -1, -1))
    assert (out[:, :5] == 0.0).all()
    # valid cells ≠ 0 (unless degenerate)
    assert (out[:, 5:].abs().sum() > 0)


def test_neutralize_industry_single_stock():
    """Single tradable stock → returned as-is (not enough for OLS)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=9))
    onehot = _make_industry_onehot(10, 3)
    mask = torch.zeros_like(p.mask)
    mask[:, 0] = True
    out = neutralize_industry(p.close, mask, onehot.expand(p.close.shape[0], -1, -1))
    # single stock → OLS skipped, output is zero (as per source: out initialized to zeros)
    assert (out == 0.0).all()


def test_neutralize_industry_perfect_collinear():
    """All stocks in same industry → OLS still works (rank-deficient handled by lstsq)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=10))
    # all stocks same industry
    onehot = torch.zeros(1, 10, 1)
    onehot[:] = 1.0
    out = neutralize_industry(p.close, p.mask, onehot.expand(p.close.shape[0], -1, -1))
    # should not crash; residuals are z-scored per date
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 2:
            continue
        vals = out[t][valid]
        assert torch.abs(vals.mean()) < 1e-5, f"date {t}: mean {vals.mean().item():.6f}"


def test_neutralize_industry_cross_sectional_output():
    """Per-date residuals have std normalised to 1."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=11))
    onehot = _make_industry_onehot(30, 5)
    out = neutralize_industry(p.close, p.mask, onehot.expand(p.close.shape[0], -1, -1))

    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 3:  # need at least 3 stocks for meaningful std
            continue
        vals = out[t][valid]
        assert torch.abs(vals.std() - 1.0) < 0.15, f"date {t}: std {vals.std().item():.6f}"
