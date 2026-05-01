"""Sweep the efficient frontier over a grid of risk-aversion values.

The paper sweeps ``α ∈ {0, 0.25, 0.5, 0.75, 1, 2, …, 300}`` to trace the
efficient frontier; at backtest time you pick a single α (typically the
one whose ex-ante volatility matches a target). This module runs the
sweep and returns one row per α with realised return + ex-ante variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .markowitz import MarkowitzConfig, MarkowitzOptimizer


@dataclass
class FrontierPoint:
    risk_aversion: float
    expected_return: float
    expected_variance: float
    weights: np.ndarray


def efficient_frontier(
    mu: np.ndarray,
    returns_history: np.ndarray,
    *,
    risk_aversions: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    cfg: MarkowitzConfig | None = None,
) -> list[FrontierPoint]:
    """Compute one :class:`FrontierPoint` per ``risk_aversion``.

    The optimiser warm-starts between consecutive ``α`` values, so this
    is significantly cheaper than building a fresh problem for each
    point.
    """
    cfg = cfg or MarkowitzConfig()
    points: list[FrontierPoint] = []
    for ra in risk_aversions:
        local = MarkowitzConfig(**{**cfg.__dict__, "risk_aversion": ra})
        opt = MarkowitzOptimizer(local)
        w = opt.solve(mu, returns_history)
        sigma = opt._covariance(returns_history)                       # noqa: SLF001
        ev = float(w @ sigma @ w)
        er = float(mu @ w)
        points.append(FrontierPoint(ra, er, ev, w))
    return points
