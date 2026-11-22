"""Frozen-portfolio capital sweeps with lagged turnover budgets."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from .constrained import ConstrainedResult, _permission, run_constrained_backtest


def lagged_turnover_caps(turnover, *, participation_rate=.01, lookback=20):
    """Cap[t,i] = rate * mean(turnover[t-lookback:t,i]), excluding row t.

    Turnover is monetary value in the account currency, not share volume.
    Require a full window of finite nonnegative observations. NaN marks missing
    history; that asset's cap is zero and ready flag false. Genuine zero turnover
    is an observation. Negative values and infinities are rejected. Initial
    lookback rows have zero caps. Direct window sums avoid subtracting large
    cumulative totals when old outliers leave the window.
    """
    values = np.asarray(turnover, dtype=np.float64)
    if values.ndim != 2 or min(values.shape, default=0) < 1:
        raise ValueError("turnover must be a nonempty [T,N] monetary grid")
    if np.isinf(values).any() or np.any(values < 0):
        raise ValueError("turnover must be nonnegative; only NaN is allowed for missing observations")
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if isinstance(participation_rate, bool) or not math.isfinite(participation_rate) or not 0 <= participation_rate <= 1:
        raise ValueError("participation_rate must be finite and in [0,1]")
    caps, ready = np.zeros_like(values), np.zeros(values.shape, dtype=bool)
    for t in range(lookback, len(values)):
        history = values[t - lookback:t]
        ready[t] = np.isfinite(history).all(axis=0)
        # Multiply before summing: finite large turnover need not overflow a mean.
        budget = np.sum(np.nan_to_num(history, nan=0.0) * (participation_rate / lookback), axis=0)
        caps[t] = np.where(ready[t], budget, 0.0)
    return caps, ready


@dataclass(repr=False)
class CapacityResult:
    capitals: np.ndarray
    trade_caps: np.ndarray           # OOS [T,N], shared dollar caps for every account size
    liquidity_ready: np.ndarray      # OOS [T,N], complete lagged-window coverage
    reference: ConstrainedResult     # Same gates, no notional caps, initial equity one
    scenarios: list[ConstrainedResult]
    rows: list[dict]
    start_index: int
    participation_rate: float
    lookback: int


def run_capacity_backtest(
    weights, returns, turnover, capitals, *, start_index=0,
    participation_rate=.01, lookback=20, buy_allowed=None, sell_allowed=None,
    costs_bps=5.0, benchmark=None,
) -> CapacityResult:
    """Replay frozen targets at each capital with shared historical dollar caps.

    Compute lagged budgets on the FULL input history, then slice start_index.
    Each independent account starts in cash at that row; pre-OOS turnover may
    inform its first cap. No retraining, portfolio reselection or best-capital
    choice occurs. The uncapped reference uses the SAME directional permissions
    with initial equity one. This isolates adding the turnover-budget policy.

    Capitals must be a unique, positive, finite one-dimensional sequence in the
    same currency as turnover. Each result retains its complete execution ledger.
    Budgets are participation in lagged mean turnover, not realized participation
    in today's volume and not a calibrated market-impact or feasible-size bound.
    """
    shape = np.asarray(weights).shape
    if len(shape) != 2 or min(shape) < 1 or np.asarray(returns).shape != shape or np.asarray(turnover).shape != shape:
        raise ValueError("weights, returns and turnover must share a nonempty [T,N] grid")
    if isinstance(start_index, bool) or not isinstance(start_index, Integral) or not 0 <= start_index < shape[0]:
        raise ValueError("start_index must select a row in the full history")
    capital_grid = np.asarray(capitals, dtype=np.float64).copy()
    if capital_grid.ndim != 1 or not capital_grid.size or not np.isfinite(capital_grid).all() or np.any(capital_grid <= 0):
        raise ValueError("capitals must be a nonempty positive finite 1D sequence")
    if np.unique(capital_grid).size != capital_grid.size:
        raise ValueError("capital scenarios must be unique")
    buy, sell = (_permission(value, shape, name) for value, name in (
        (buy_allowed, "buy_allowed"), (sell_allowed, "sell_allowed")))
    caps, ready = lagged_turnover_caps(turnover, participation_rate=participation_rate, lookback=lookback)
    selected = slice(start_index, None)
    options = {"costs_bps": costs_bps, "buy_allowed": buy[selected], "sell_allowed": sell[selected]}
    if benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=np.float64)
        if benchmark.shape != (shape[0],) or not np.isfinite(benchmark).all() or np.any(benchmark < -1):
            raise ValueError("benchmark must be a finite full-history [T] simple-return series >= -1")
        options["benchmark"] = benchmark[selected]
    targets, realized = np.asarray(weights)[selected], np.asarray(returns)[selected]
    reference = run_constrained_backtest(targets, realized, **options)
    scenarios, rows = [], []
    for capital in capital_grid:
        result = run_constrained_backtest(targets, realized, initial_equity=float(capital),
                                          max_trade_values=caps[selected], **options)
        positive_nav = result.cumulative_equity > 0
        cash_weights = result.cash_equity[positive_nav] / result.cumulative_equity[positive_nav]
        rows.append({
            "initial_capital": float(capital),
            "final_nav": float(result.cumulative_equity[-1]),
            "final_equity_multiple": float(result.cumulative_equity[-1] / capital),
            "uncapped_final_equity_multiple": float(reference.cumulative_equity[-1]),
            "equity_multiple_gap": float(result.cumulative_equity[-1] / capital - reference.cumulative_equity[-1]),
            "notional_fill_ratio": result.metrics["notional_fill_ratio"],
            "mean_cash_weight": float(cash_weights.mean()),
            "n_positive_nav_dates": int(positive_nav.sum()),
            "mean_target_weight_l1": float(np.abs(result.weights - result.target_weights).sum(axis=1).mean()),
            "cap_unfilled_fraction_initial": result.metrics["cap_unfilled_notional"] / float(capital),
            "fees_fraction_initial": result.metrics["fees_fraction_initial"],
            "bankrupt_index": result.bankrupt_index,
            "metrics": result.metrics,
        })
        scenarios.append(result)
    return CapacityResult(capitals=capital_grid, trade_caps=caps[selected].copy(),
                          liquidity_ready=ready[selected].copy(), reference=reference,
                          scenarios=scenarios, rows=rows, start_index=int(start_index),
                          participation_rate=float(participation_rate), lookback=int(lookback))
