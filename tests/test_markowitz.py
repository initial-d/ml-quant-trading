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
