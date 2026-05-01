"""Vectorised backtest engine.

Inputs
------
weights : np.ndarray [T, N]
    Position weights. Row ``t`` is the weight vector chosen *with
    information up to date t* and applied for the period ``t -> t+1``.
returns : np.ndarray [T, N]
    Realised simple returns from ``t-1 -> t`` (so ``returns[t]`` is
    earned by ``weights[t-1]``).
costs_bps : float
    Round-trip transaction cost in basis points charged on absolute
    weight changes. The default 5 bps roughly matches A-share retail
    commissions + half-spread on liquid names.

The engine is intentionally tiny — every realistic refinement (slippage
models, borrow fees, T+1 settlement) goes into a sibling module rather
than here. Doing so keeps the core path easy to audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import metrics


@dataclass
class BacktestResult:
    portfolio_returns: np.ndarray          # [T] net of costs
    gross_returns:     np.ndarray          # [T] gross
    cumulative_equity: np.ndarray          # [T]
    weights:           np.ndarray          # [T, N]
    cost_drag:         float               # total cost paid (cumulative)
    metrics:           dict                # summary table

    def summary(self) -> dict:
        return self.metrics

    def __repr__(self) -> str:                      # pragma: no cover
        m = self.metrics
        return (
            "BacktestResult("
            f"ann_return={m.get('ann_return', 0):.2%}, "
            f"sharpe={m.get('sharpe', 0):.2f}, "
            f"max_dd={m.get('max_dd', 0):.2%}, "
            f"turnover={m.get('turnover', 0):.2%})"
        )


def run_backtest(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    costs_bps: float = 5.0,
    benchmark: Optional[np.ndarray] = None,
) -> BacktestResult:
    """Run a long-only / long-short vector backtest."""
    if weights.shape != returns.shape:
        raise ValueError(f"shape mismatch: weights {weights.shape}  returns {returns.shape}")

    T, _ = weights.shape

    # Lagged weights: weights[t-1] is what's earned over t-1 -> t.
    w_lag = np.zeros_like(weights)
    w_lag[1:] = weights[:-1]

    gross = (w_lag * returns).sum(axis=1)
    # Cost drag from rebalancing day t.
    delta = np.zeros_like(weights)
    delta[1:] = np.abs(weights[1:] - weights[:-1])
    cost = delta.sum(axis=1) * (costs_bps * 1e-4)
    net = gross - cost

    equity = np.cumprod(1.0 + net)

    summary = {
        "ann_return": metrics.annualised_return(net),
        "ann_vol":    metrics.annualised_vol(net),
        "sharpe":     metrics.sharpe_ratio(net),
        "sortino":    metrics.sortino_ratio(net),
        "calmar":     metrics.calmar_ratio(net),
        "max_dd":     metrics.max_drawdown(net),
        "turnover":   metrics.turnover(weights),
        "cost_drag":  float(cost.sum()),
        "n_periods":  int(T),
    }

    if benchmark is not None and benchmark.shape == net.shape:
        active = net - benchmark
        summary["info_ratio"] = metrics.sharpe_ratio(active)
        summary["alpha_ann"]  = metrics.annualised_return(active)

    return BacktestResult(
        portfolio_returns=net,
        gross_returns=gross,
        cumulative_equity=equity,
        weights=weights,
        cost_drag=float(cost.sum()),
        metrics=summary,
    )
