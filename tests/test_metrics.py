"""Backtest metric formulas."""
from __future__ import annotations

import numpy as np

from mlquant.backtest.metrics import (
    annualised_return, annualised_vol, calmar_ratio, max_drawdown,
    rank_information_coefficient, sharpe_ratio, sortino_ratio, turnover,
    information_coefficient,
)


def test_annualised_return_zero_on_zero_returns():
    assert annualised_return(np.zeros(252)) == 0.0


def test_sharpe_sign_matches_mean_return():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.001, 0.01, size=252)
    neg = -pos
    assert sharpe_ratio(pos) > 0
    assert sharpe_ratio(neg) < 0


def test_max_drawdown_known():
    r = np.array([0.10, -0.05, -0.10, 0.05, 0.02])
    mdd = max_drawdown(r)
    eq = np.cumprod(1 + r)
    expected = (eq.cummax() if hasattr(eq, "cummax") else np.maximum.accumulate(eq))
    expected = ((expected - eq) / expected).max()
    assert abs(mdd - expected) < 1e-9


def test_information_coefficient_perfect():
    p = np.linspace(0, 1, 50)
    assert abs(information_coefficient(p, p) - 1.0) < 1e-6
    assert abs(rank_information_coefficient(p, p) - 1.0) < 1e-6


def test_turnover_zero_for_constant_weights():
    w = np.tile(np.array([0.5, 0.5]), (10, 1))
    assert turnover(w) == 0.0


def test_sortino_and_calmar_finite_on_random():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, size=500)
    assert np.isfinite(sortino_ratio(r))
    assert np.isfinite(calmar_ratio(r))
    assert np.isfinite(annualised_vol(r))
