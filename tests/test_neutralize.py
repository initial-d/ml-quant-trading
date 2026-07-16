"""Cross-sectional and industry neutralisation tests.

Tests verify both the mathematical properties (mean zero, orthogonality,
unit-variance) and numerical-stability edge cases.
"""
from __future__ import annotations

import torch
import pytest

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.neutralize import neutralize_cs, neutralize_industry


# =============================================================================
# Helpers
# =============================================================================

def _make_onehot(n_stocks: int, n_industries: int) -> torch.Tensor:
    """Per-stock industry one-hot [1, N, K] with round-robin assignment."""
    onehot = torch.zeros(1, n_stocks, n_industries)
    for i in range(n_stocks):
        onehot[0, i, i % n_industries] = 1.0
    return onehot


# =============================================================================
# neutralize_cs
# =============================================================================


def test_neutralize_cs_mean_zero():
    """Per-date cross-sectional z-score has mean≈0 on valid cells."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=50, n_dates=60, seed=1))
    out = neutralize_cs(p.close, p.mask)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 2:
            continue
        vals = out[t][valid]
        assert vals.mean().abs() < 1e-5


def test_neutralize_cs_unit_std():
    """cs_zscore uses population std; test with unbiased=False."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=50, n_dates=60, seed=1))
    out = neutralize_cs(p.close, p.mask)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 2:
            continue
        vals = out[t][valid]
        assert torch.allclose(vals.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_neutralize_cs_mask_is_zero():
    """Masked cells remain zero."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=2))
    out = neutralize_cs(p.close, p.mask)
    assert (out[~p.mask] == 0.0).all()


def test_neutralize_cs_const_input():
    """Constant values → zero output (no cross-sectional variation)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=3))
    p.close[:] = 1.0
    out = neutralize_cs(p.close, p.mask)
    assert torch.allclose(out[p.mask], torch.zeros(1), atol=1e-6)


def test_neutralize_cs_single_tradable():
    """Single valid stock → zero (insufficient cross-section for z-score)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=5, n_dates=5, seed=4))
    mask = torch.zeros_like(p.mask)
    mask[:, 0] = True
    out = neutralize_cs(p.close, mask)
    assert (out == 0.0).all()


def test_neutralize_cs_all_masked():
    """No tradable stocks → all zero."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=5))
    mask = torch.zeros_like(p.mask)
    out = neutralize_cs(p.close, mask)
    assert (out == 0.0).all()


def test_neutralize_cs_num_tradable_equals_one():
    """Exactly one stock tradable each day → output zero (std=0, clamped)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=10, seed=6))
    mask = torch.zeros_like(p.mask)
    for t in range(mask.shape[0]):
        mask[t, t % mask.shape[1]] = True
    out = neutralize_cs(p.close, mask)
    assert (out == 0.0).all()


def test_neutralize_cs_permutation():
    """Stock order permutation → output permuted identically."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=10, seed=7))
    out1 = neutralize_cs(p.close, p.mask)
    perm = torch.randperm(p.close.shape[1])
    out2 = neutralize_cs(p.close[:, perm], p.mask[:, perm])
    assert torch.allclose(out1[:, perm], out2, atol=1e-5)


# =============================================================================
# neutralize_industry — basic properties
# =============================================================================


def test_neutralize_industry_mean_zero():
    """Residuals after industry neutralisation have per-date mean≈0."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=8))
    onehot = _make_onehot(30, 5).expand(p.close.shape[0], -1, -1)
    out = neutralize_industry(p.close, p.mask, onehot)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 3:
            continue
        assert out[t][valid].mean().abs() < 1e-5


def test_neutralize_industry_orthogonal_to_industries():
    """Residuals should be (approximately) orthogonal to industry dummies."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=24, n_dates=10, seed=9))
    onehot = _make_onehot(24, 4).expand(p.close.shape[0], -1, -1)
    out = neutralize_industry(p.close, p.mask, onehot)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 5:
            continue
        X = onehot[t][valid].float()
        r = out[t][valid].unsqueeze(1)
        proj = (X.T @ r).abs().max()
        assert proj < 1e-3, f"date {t}: max industry exposure {proj.item():.6f}"


def test_neutralize_industry_orthogonal_to_size():
    """With log_size, residuals orthogonal to size."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=24, n_dates=10, seed=10))
    onehot = _make_onehot(24, 4).expand(p.close.shape[0], -1, -1)
    log_size = torch.log(p.close.abs().clamp_min(1e-12))
    out = neutralize_industry(p.close, p.mask, onehot, log_size=log_size)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 6:
            continue
        r = out[t][valid]
        s = log_size[t][valid]
        # covariance between residual and size
        cov = ((r - r.mean()) * (s - s.mean())).mean().abs()
        assert cov < 1e-3, f"date {t}: size covariance {cov.item():.6f}"


def test_neutralize_industry_mask_is_zero():
    """Masked cells remain zero."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=10, seed=11))
    onehot = _make_onehot(20, 4).expand(p.close.shape[0], -1, -1)
    mask = p.mask.clone()
    mask[:, :5] = False
    out = neutralize_industry(p.close, mask, onehot)
    assert (out[:, :5] == 0.0).all()


# =============================================================================
# neutralize_industry — edge cases
# =============================================================================


def test_neutralize_industry_insufficient_stocks():
    """Single tradable stock → zero (OLS not possible)."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=12))
    onehot = _make_onehot(10, 3).expand(p.close.shape[0], -1, -1)
    mask = torch.zeros_like(p.mask)
    mask[:, 0] = True
    out = neutralize_industry(p.close, mask, onehot)
    assert (out == 0.0).all()


def test_neutralize_industry_rank_deficient():
    """True rank-deficient design: two identical industry columns."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=10, seed=13))
    # create two identical dummy columns
    onehot_single = _make_onehot(20, 3)  # [1, N, 3]
    onehot = torch.cat([onehot_single, onehot_single[:, :, :1]], dim=-1)  # [1, N, 4], col3 == col0
    onehot = onehot.expand(p.close.shape[0], -1, -1)
    out = neutralize_industry(p.close, p.mask, onehot)
    # should not crash or produce NaN
    assert torch.isfinite(out).all()
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 5:
            continue
        assert out[t][valid].mean().abs() < 1e-5


def test_neutralize_industry_perfectly_explained():
    """Signal perfectly explained by industries → output zero (no amplified noise).

    Uses a full-rank design: each stock is in exactly one of K-1 industries
    (plus an implicit Kth baseline group). The signal x is then x = X @ beta,
    which is perfectly explained by the design. After neutralisation the
    residual should be near machine precision, not amplified unit-variance noise."""
    T, N = 10, 20
    K = 4  # K-1 = 3 columns in design (full rank)
    # Industry 0..K-2 each get N//K stocks; the remaining are "baseline" (all zeros)
    X = torch.zeros(1, N, K - 1)
    for i in range(N):
        grp = i % K
        if grp < K - 1:
            X[0, i, grp] = 1.0
    # full-rank by construction: no column is all-zeros, stock counts differ → no collinearity
    beta = torch.tensor([1.5, -2.0, 0.7])
    x = (X.squeeze(0) @ beta).unsqueeze(0).expand(T, -1)
    mask = torch.ones(T, N, dtype=torch.bool)
    onehot_expanded = X.expand(T, -1, -1)
    out = neutralize_industry(x, mask, onehot_expanded)
    assert out.abs().max() < 1e-4, f"max residual: {out.abs().max().item():.6e}"


def test_neutralize_industry_perfectly_explained_with_size():
    """Signal = industry*beta + size*gamma → output zero (perfectly explained)."""
    T, N = 10, 20
    K = 4
    X = torch.zeros(1, N, K - 1)
    for i in range(N):
        grp = i % K
        if grp < K - 1:
            X[0, i, grp] = 1.0
    size = torch.randn(1, N).expand(T, -1)
    beta = torch.tensor([1.5, -2.0, 0.7])
    gamma = 3.0
    x = (X.squeeze(0) @ beta).unsqueeze(0).expand(T, -1) + gamma * size
    mask = torch.ones(T, N, dtype=torch.bool)
    onehot_expanded = X.expand(T, -1, -1)
    out = neutralize_industry(x, mask, onehot_expanded, log_size=size)
    assert out.abs().max() < 1e-4, f"max residual: {out.abs().max().item():.6e}"


def test_neutralize_industry_std_normalized():
    """Non-degenerate residuals have unit population std."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=20, seed=16))
    onehot = _make_onehot(30, 5).expand(p.close.shape[0], -1, -1)
    out = neutralize_industry(p.close, p.mask, onehot)
    for t in range(p.mask.shape[0]):
        valid = p.mask[t]
        if valid.sum() < 3:
            continue
        vals = out[t][valid]
        # population std ≈ 1 after z-scoring (atol 0.02 accounts for regression residual noise)
        assert torch.allclose(vals.std(unbiased=False), torch.tensor(1.0), atol=0.02), \
            f"date {t}: std {vals.std(unbiased=False).item():.6f}"


# =============================================================================
# neutralize — shape / dtype / device
# =============================================================================


def test_neutralize_cs_dtype():
    """float32 input → float32 output."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=17))
    out = neutralize_cs(p.close.float(), p.mask)
    assert out.dtype == torch.float32


def test_neutralize_industry_dtype():
    """float32 input → float32 output."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=5, seed=18))
    onehot = _make_onehot(10, 3).expand(5, -1, -1)
    out = neutralize_industry(p.close.float(), p.mask, onehot.float())
    assert out.dtype == torch.float32


def test_neutralize_cs_shape():
    """Output shape matches input."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=15, n_dates=8, seed=19))
    out = neutralize_cs(p.close, p.mask)
    assert out.shape == p.close.shape


def test_neutralize_industry_shape():
    """Output shape matches input."""
    p = make_synthetic_panel(SyntheticConfig(n_stocks=15, n_dates=8, seed=20))
    onehot = _make_onehot(15, 3).expand(8, -1, -1)
    out = neutralize_industry(p.close, p.mask, onehot)
    assert out.shape == p.close.shape
