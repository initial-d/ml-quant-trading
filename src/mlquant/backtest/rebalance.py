"""Closed-loop rebalancing decisions from actual marked account holdings."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import cvxpy as cp
import numpy as np

from ..portfolio.rebalance import RebalanceConfig, RebalanceOptimizer
from .constrained import ConstrainedResult, _permission, _run_constrained_ledger


@dataclass(repr=False)
class RebalancedResult(ConstrainedResult):
    decision_records: list[dict]
    planned_holding_values: np.ndarray
    planned_fees: np.ndarray
    config: RebalanceConfig
    lookback: int
    start_index: int


def run_rebalance_backtest(predictions, returns, *, cfg=None, lookback=60, start_index=0,
                           initial_equity=1., buy_allowed=None, sell_allowed=None,
                           max_trade_values=None, benchmark=None):
    """Re-optimize after marking actual holdings; covariance excludes row t.

    Forecasts are one-period expected simple returns, not ranks/probabilities.
    Inputs cover full history; each account starts in cash at start_index. Prior
    history may inform its first decision. Insufficient covariance history holds
    the account exactly. Infeasible/failed solves also hold and retain the reason.
    Directional/cap constraints are passed to BOTH planning and actual fills.
    No future forecasts, returns, permissions or caps enter an earlier decision.
    """
    forecasts, realized = (np.asarray(value, dtype=np.float64) for value in (predictions, returns))
    if forecasts.ndim != 2 or min(forecasts.shape, default=0) < 1 or realized.shape != forecasts.shape:
        raise ValueError("predictions and returns must share a nonempty [T,N] grid")
    if not np.isfinite(forecasts).all() or not np.isfinite(realized).all() or np.any(realized < -1):
        raise ValueError("finite forecasts and finite simple returns >= -1 are required")
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    if isinstance(start_index, bool) or not isinstance(start_index, Integral) or not 0 <= start_index < len(forecasts):
        raise ValueError("start_index must select a row in full history")
    config = cfg or RebalanceConfig()
    optimizer = RebalanceOptimizer(config)
    shape = forecasts.shape
    buy = _permission(buy_allowed, shape, "buy_allowed")
    sell = _permission(sell_allowed, shape, "sell_allowed")
    caps = None if max_trade_values is None else np.asarray(max_trade_values, dtype=np.float64).copy()
    if caps is not None and (caps.shape != shape or not np.isfinite(caps).all() or np.any(caps < 0)):
        raise ValueError("max_trade_values must be finite nonnegative full-history [T,N]")
    if benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=np.float64)
        if benchmark.shape != (len(forecasts),) or not np.isfinite(benchmark).all() or np.any(benchmark < -1):
            raise ValueError("benchmark must be finite full-history [T] simple returns >= -1")
    selected = slice(start_index, None)
    count = len(forecasts) - start_index
    records = [{"row": start_index + t, "status": "bankrupt_skip"} for t in range(count)]
    planned, planned_fees = np.zeros((count, shape[1])), np.zeros(count)
    lost = np.zeros(shape[1], dtype=bool)
    def decide(t, marked, cash, before):
        row = start_index + t
        lost[:] |= realized[row] == -1
        planned[t] = marked
        if row < lookback:
            records[t] = {"row": row, "status": "warmup_hold", "history_rows": row}
            return None
        try:
            proposal = optimizer.solve(forecasts[row], realized[row - lookback:row], marked / before,
                                       buy_allowed=buy[row] & ~lost, sell_allowed=sell[row],
                                       max_trade_weights=None if caps is None else caps[row] / before)
        except (ValueError, cp.error.SolverError) as exc:
            records[t] = {"row": row, "status": "failed_hold", "reason": str(exc)}
            return None
        planned[t] = proposal.holding_values_fraction * before
        planned_fees[t] = proposal.fees_fraction * before
        records[t] = {"row": row, "status": "optimized", "solver_status": proposal.status,
                      "solver": proposal.solver, "objective_value": proposal.objective_value,
                      "repair_l1": proposal.repair_l1, "selection": proposal.selection}
        # A true no-trade plan holds exact marked values rather than round-tripping NAV fractions.
        if np.array_equal(proposal.holding_values_fraction, marked / before):
            return None
        return proposal.target_weights
    base = _run_constrained_ledger(np.zeros((count, shape[1])), realized[selected],
        buy_allowed=buy[selected], sell_allowed=sell[selected],
        max_trade_values=None if caps is None else caps[selected], costs_bps=config.costs_bps,
        initial_equity=initial_equity, benchmark=None if benchmark is None else benchmark[selected], decision=decide)
    base.metrics.update(
        optimized_dates=sum(record['status'] == 'optimized' for record in records),
        failed_hold_dates=sum(record['status'] == 'failed_hold' for record in records),
        warmup_hold_dates=sum(record['status'] == 'warmup_hold' for record in records),
        max_plan_fill_value_error=float(np.abs(planned - base.weights * base.cumulative_equity[:, None]).max()),
        max_plan_fee_error=float(np.abs(planned_fees - base.fees).max()),
    )
    return RebalancedResult(**vars(base), decision_records=records, planned_holding_values=planned,
                            planned_fees=planned_fees, config=config, lookback=int(lookback), start_index=int(start_index))
