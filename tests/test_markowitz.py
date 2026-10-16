"""Properties of the cross-sectional Markowitz optimiser."""
from __future__ import annotations

import numpy as np

from mlquant.portfolio.markowitz import MarkowitzConfig, MarkowitzOptimizer


def _toy_problem(n: int = 5, T: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    history = rng.normal(0.0, 0.02, size=(T, n))
    mu = rng.normal(0.001, 0.0005, size=n)
    return mu, history


def test_long_only_weights_sum_to_one():
    mu, history = _toy_problem()
    opt = MarkowitzOptimizer(MarkowitzConfig(weight_cap=1.0))
    w = opt.solve(mu, history)
    assert (w >= -1e-7).all()
    assert abs(w.sum() - 1.0) < 1e-3


def test_weight_cap_respected():
    mu, history = _toy_problem(n=10)
    opt = MarkowitzOptimizer(MarkowitzConfig(weight_cap=0.2))
    w = opt.solve(mu, history)
    assert w.max() <= 0.2 + 1e-3


def test_higher_risk_aversion_lower_variance():
    mu, history = _toy_problem(n=10, T=400, seed=2)
    cov = np.cov(history, rowvar=False)
    w_low  = MarkowitzOptimizer(MarkowitzConfig(risk_aversion=0.1, weight_cap=1.0)).solve(mu, history)
    w_high = MarkowitzOptimizer(MarkowitzConfig(risk_aversion=10.0, weight_cap=1.0)).solve(mu, history)
    var_low  = float(w_low  @ cov @ w_low)
    var_high = float(w_high @ cov @ w_high)
    assert var_high <= var_low + 1e-8


def test_problem_is_dpp_and_cached_per_universe_size():
    """The risk term must be DPP, or cvxpy re-canonicalises on every solve."""
    import warnings

    mu, history = _toy_problem(n=30, T=60)
    opt = MarkowitzOptimizer(MarkowitzConfig(weight_cap=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opt.solve(mu, history)
        opt.solve(mu * 2, history * 1.1)
    problem = opt._problems[30][3]
    assert problem.is_dpp()
    assert len(opt._problems) == 1


def test_rank_deficient_history_solves():
    """More names than look-back days: the covariance is rank-deficient."""
    rng = np.random.default_rng(3)
    opt = MarkowitzOptimizer(MarkowitzConfig(risk_aversion=5.0, weight_cap=0.05))
    for _ in range(5):
        w = opt.solve(rng.normal(0, 0.01, 200), rng.normal(0, 0.02, size=(60, 200)))
        assert abs(w.sum() - 1.0) < 1e-3
        assert w.max() <= 0.05 + 1e-3


def test_infeasible_weight_cap_raises_clear_error():
    import pytest

    mu, history = _toy_problem(n=10)
    with pytest.raises(ValueError, match="infeasible"):
        MarkowitzOptimizer(MarkowitzConfig(weight_cap=0.05)).solve(mu, history)
