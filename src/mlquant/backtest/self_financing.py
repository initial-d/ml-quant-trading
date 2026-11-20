"""Long-only target rebalancing with marked holdings and cash-paid fees.

This is historical accounting assuming complete fills at each row's mark, not
an execution simulator. Targets describe post-fee NAV fractions; residual cash
earns zero interest. No borrowing, shorting, external deposits or withdrawals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .engine import BacktestResult
from . import metrics


@dataclass(repr=False)
class SelfFinancingResult(BacktestResult):
    target_weights: np.ndarray            # [T,N] after small budget-rounding correction
    pretrade_weights: np.ndarray          # [T,N] marked asset NAV fractions, before fees
    trade_values: np.ndarray              # [T,N] signed purchases in initial_equity units
    pretrade_equity: np.ndarray            # [T] marked NAV, including cash
    fees: np.ndarray                       # [T] fees in initial_equity units
    cash_equity: np.ndarray                # [T] cash after trading, zero interest
    turnover_l1: np.ndarray                # [T] gross traded notional / pretrade NAV
    initial_equity: float
    target_rounding_max_excess: float
    bankrupt_index: int | None


def _post_fee_fraction(target, pretrade, rate):
    """Solve y + rate*sum(abs(target*y-pretrade)) = 1 on [0,1].

    With nonnegative weights summing to <=1 and rate<1, this continuous
    piecewise-linear function is strictly increasing. Bisection also handles
    trade-sign changes at breakpoints without guessing a buy/sell partition.
    """
    if rate == 0 or np.array_equal(target, pretrade):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = .5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if mid + rate * float(np.abs(target * mid - pretrade).sum()) > 1:
            hi = mid
        else:
            lo = mid
    return .5 * (lo + hi)


def run_self_financing_backtest(
    weights, returns, *, costs_bps: float = 5.0,
    initial_equity: float = 1.0, benchmark=None,
) -> SelfFinancingResult:
    """Mark old holdings, pay fees and rebalance to post-fee target fractions.

    Row t's returns are earned by row t-1's holdings; row t's targets then
    apply to t->t+1. The initial account is all cash. costs_bps is a ONE-WAY
    proportional fee per absolute traded notional, including entry/exit.
    Fees and purchases are jointly solved from existing NAV, not paid from
    an external cash source. A wiped-out account is absorbing.

    Inputs must be finite, long-only targets and simple returns >=-1, with
    matching nonempty [T,N] grids. Target sums above one by at most 1e-6 are
    normalized on a copy to accommodate float32 portfolio rounding; larger
    overruns are rejected. The correction is recorded. Targets below full
    investment leave residual cash. This model assumes complete fills and
    does not enforce tradability, settlement, volume or price-impact limits.
    An asset with a recorded -100% return cannot receive positive targets
    again; an account with zero NAV cannot be funded or restarted.
    """
    target = np.asarray(weights, dtype=np.float64).copy()
    realized = np.asarray(returns, dtype=np.float64)
    if target.ndim != 2 or target.shape != realized.shape or min(target.shape, default=0) < 1:
        raise ValueError("weights and returns must share a nonempty [T,N] grid")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("targets must be finite and nonnegative; shorting is unsupported")
    if not np.isfinite(realized).all() or np.any(realized < -1):
        raise ValueError("simple returns must be finite and >= -1")
    if isinstance(costs_bps, bool) or not math.isfinite(costs_bps) or not 0 <= costs_bps < 10000:
        raise ValueError("costs_bps must be finite and in [0,10000)")
    if isinstance(initial_equity, bool) or not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    if np.any(target > 1 + 1e-6):
        raise ValueError("target budget exceeds one; borrowing is unsupported")
    sums = target.sum(axis=1)
    excess = float(np.maximum(sums - 1, 0).max())
    if excess > 1e-6:
        raise ValueError("target budget exceeds one; borrowing is unsupported")
    over = sums > 1
    target[over] /= sums[over, None]
    T, N = target.shape
    if benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=np.float64)
        if benchmark.shape != (T,) or not np.isfinite(benchmark).all() or np.any(benchmark < -1):
            raise ValueError("benchmark must be a finite [T] simple-return series >= -1")
    actual, pre_weights, trades = (np.zeros((T, N), dtype=np.float64) for _ in range(3))
    gross, net, equity, pre_equity, fees, cash_path, l1, cost_returns = (np.zeros(T) for _ in range(8))
    holdings, cash, previous = np.zeros(N), float(initial_equity), float(initial_equity)
    bankrupt = None
    lost_assets = np.zeros(N, dtype=bool)
    rate = costs_bps * 1e-4
    for t in range(T):
        if bankrupt is not None:
            continue  # No deposits or resurrection after zero NAV.
        with np.errstate(over="raise", invalid="raise"):
            try:
                marked = holdings * (1 + realized[t])
                before = float(marked.sum() + cash)
            except FloatingPointError as exc:
                raise ValueError("nonfinite NAV from returns/initial_equity scale") from exc
        if not math.isfinite(before):
            raise ValueError("nonfinite NAV from returns/initial_equity scale")
        pre_equity[t] = before
        gross[t] = before / previous - 1
        if before == 0:
            net[t] = -1.0
            bankrupt = t
            continue
        lost_assets |= realized[t] == -1
        if np.any(target[t, lost_assets] > 0):
            raise ValueError("positive target after an asset's total loss; re-entry is unsupported")
        pre_weights[t] = marked / before
        fraction = _post_fee_fraction(target[t], pre_weights[t], rate)
        holdings = target[t] * (before * fraction)
        trades[t] = holdings - marked
        l1[t] = float(np.abs(trades[t] / before).sum())
        fees[t] = before * (rate * l1[t])
        # Residual cash, with only an ULP-level floor at a fully invested row.
        cash = max(0.0, before * fraction - float(holdings.sum()))
        after = float(holdings.sum() + cash)
        if after == 0:
            raise ValueError("post-fee NAV underflow; choose a representable initial_equity scale")
        equity[t], cash_path[t] = after, cash
        actual[t] = holdings / after
        cost_returns[t] = fees[t] / previous
        net[t] = after / previous - 1
        previous = after
    peak = np.maximum.accumulate(np.r_[float(initial_equity), equity])[1:]
    drawdown = float(np.max((peak - equity) / peak))
    annual = metrics.annualised_return(net)
    summary = {"ann_return": annual, "ann_vol": metrics.annualised_vol(net),
               "sharpe": metrics.sharpe_ratio(net) if T > 1 else 0.0,
               "sortino": metrics.sortino_ratio(net),
               "calmar": annual / drawdown if drawdown > 1e-12 else float("inf"),
               "max_dd": drawdown,
               "turnover": .5 * float(l1[1:].mean()) if T > 1 else 0.0,
               "turnover_including_entry": .5 * float(l1.mean()),
               "cost_drag_cumulative": float(cost_returns.sum()), "cost_drag": float(cost_returns.sum()),
               "fees_total": float(fees.sum()), "fees_fraction_initial": float(fees.sum() / initial_equity),
               "max_self_financing_residual": float(np.max(np.abs(pre_equity - equity - fees))),
               "target_rounding_max_excess": excess, "n_periods": T}
    if benchmark is not None:
        summary["info_ratio"] = metrics.sharpe_ratio(net - benchmark) if T > 1 else 0.0
        summary["alpha_ann"] = metrics.annualised_return(net - benchmark)
    return SelfFinancingResult(portfolio_returns=net, gross_returns=gross, cumulative_equity=equity,
                               weights=actual, cost_drag_cumulative=float(cost_returns.sum()), metrics=summary,
                               target_weights=target, pretrade_weights=pre_weights, trade_values=trades,
                               pretrade_equity=pre_equity, fees=fees, cash_equity=cash_path, turnover_l1=l1,
                               initial_equity=float(initial_equity), target_rounding_max_excess=excess,
                               bankrupt_index=bankrupt)
