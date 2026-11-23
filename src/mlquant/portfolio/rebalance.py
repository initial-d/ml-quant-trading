"""Long-only mean-variance rebalancing with cash-paid transaction fees."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class RebalanceConfig:
    risk_aversion: float = 1.
    costs_bps: float = 5.
    position_value_cap: float = .05  # Fraction of PRE-TRADE NAV; existing drift may remain.
    cash_floor: float = 0.           # Cash floor as fraction of PRE-TRADE NAV, after fees.
    shrinkage: bool = True
    solver: str = "CLARABEL"

    def __post_init__(self):
        for name, lower, upper, closed_upper in (
            ("risk_aversion", 0, math.inf, False), ("costs_bps", 0, 10000, False),
            ("position_value_cap", 0, 1, True), ("cash_floor", 0, 1, True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < lower or value > upper or (not closed_upper and value == upper):
                raise ValueError(f"invalid {name}")
        if not isinstance(self.shrinkage, bool) or not isinstance(self.solver, str) or not self.solver:
            raise ValueError("shrinkage must be boolean and solver a nonempty name")


@dataclass
class RebalanceDecision:
    target_weights: np.ndarray       # Post-fee NAV fractions for the execution ledger.
    holding_values_fraction: np.ndarray  # Planned holdings / pre-trade NAV.
    fees_fraction: float
    cash_fraction: float             # Planned cash / pre-trade NAV.
    objective_value: float            # Predicted net utility, not realized alpha.
    solver: str
    status: str
    repair_l1: float
    selection: str


class RebalanceOptimizer:
    """Maximize mu@h - risk*h'Cov*h - c*|h-current| with funded cash.

    All values use pre-trade NAV as numeraire. Absolute trade bounds encode
    directions and optional monetary caps divided by that NAV. The covariance
    is supplied by a trailing history. Residual cash is free to exceed its floor.
    Position caps are monetary fractions of pre-trade NAV. Existing holdings
    above that cap may remain but cannot increase; this avoids forcing trades
    when a drifted position is locked. No shorting or borrowing is supported.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RebalanceConfig()
        self._problems = {}

    def _problem_for(self, n):
        if n not in self._problems:
            mu, factor, current = cp.Parameter(n), cp.Parameter((n, n)), cp.Parameter(n)
            lower, upper = cp.Parameter(n), cp.Parameter(n)
            holdings, absolute = cp.Variable(n), cp.Variable(n, nonneg=True)
            rate = self.cfg.costs_bps * 1e-4
            problem = cp.Problem(cp.Maximize(mu @ holdings
                - self.cfg.risk_aversion * cp.sum_squares(factor.T @ holdings) - rate * cp.sum(absolute)),
                [holdings >= lower, holdings <= upper,
                 absolute >= holdings - current, absolute >= current - holdings,
                 cp.sum(holdings) + rate * cp.sum(absolute) <= 1 - self.cfg.cash_floor])
            self._problems[n] = (mu, factor, current, lower, upper, holdings, problem)
        return self._problems[n]

    def solve(self, mu, returns_history, current_weights, *, buy_allowed=None,
              sell_allowed=None, max_trade_weights=None):
        mu, history, current = (np.asarray(value, dtype=np.float64) for value in (mu, returns_history, current_weights))
        if mu.ndim != 1 or not mu.size or current.shape != mu.shape or history.ndim != 2 or history.shape[1] != mu.size or len(history) < 2:
            raise ValueError("mu/current must be matching [N], with at least two historical [T,N] rows")
        if not np.isfinite(mu).all() or not np.isfinite(history).all() or not np.isfinite(current).all() or np.any(current < 0) or current.sum() > 1 + 1e-12:
            raise ValueError("finite inputs and nonnegative current holdings within NAV are required")
        n, rate = len(mu), self.cfg.costs_bps * 1e-4
        def permission(value):
            result = np.ones(n, dtype=bool) if value is None else np.asarray(value)
            if result.shape != (n,) or result.dtype != np.bool_:
                raise ValueError("trade permissions must be boolean [N]")
            return result
        buy, sell = permission(buy_allowed), permission(sell_allowed)
        lower = np.where(sell, 0., current)
        upper = np.where(buy, np.maximum(current, self.cfg.position_value_cap), current)
        if max_trade_weights is not None:
            cap = np.asarray(max_trade_weights, dtype=np.float64)
            if cap.shape != (n,) or not np.isfinite(cap).all() or np.any(cap < 0):
                raise ValueError("max_trade_weights must be finite nonnegative [N] pre-trade NAV fractions")
            lower = np.maximum(lower, current - cap)
            upper = np.minimum(upper, current + cap)
        def cash_for(h):
            return 1 - float(h.sum()) - rate * float(np.abs(h - current).sum())
        if cash_for(lower) < self.cfg.cash_floor - 1e-12:
            raise ValueError("infeasible cash floor under locked holdings/trade caps")
        covariance = (LedoitWolf().fit(history).covariance_ if self.cfg.shrinkage
                      else np.atleast_2d(np.cov(history, rowvar=False)))
        eigenvalues, eigenvectors = np.linalg.eigh(.5 * (covariance + covariance.T))
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        parameters = self._problem_for(n)
        for parameter, value in zip(parameters[:5], (mu, factor, current, lower, upper)):
            parameter.value = value
        variable, problem = parameters[-2:]
        solver = self.cfg.solver
        try:
            options = ({"eps": 1e-8, "max_iters": 50000} if solver == "SCS" else
                       {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10} if solver == "CLARABEL" else {})
            problem.solve(solver=solver, warm_start=True, **options)
        except cp.error.SolverError:
            solver = "SCS"
            problem.solve(solver=solver, warm_start=True, eps=1e-8, max_iters=50000)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or variable.value is None:
            raise cp.error.SolverError(f"no rebalancing solution (status {problem.status})")
        raw = np.asarray(variable.value).ravel()
        if not np.isfinite(raw).all() or np.any(raw < lower - 1e-6) or np.any(raw > upper + 1e-6) or cash_for(raw) < self.cfg.cash_floor - 1e-6:
            raise cp.error.SolverError("rebalancing solution violates feasibility tolerance")
        holdings = np.clip(raw, lower, upper)
        # Remove only solver noise around a no-trade boundary, not a fee threshold.
        near = np.abs(holdings - current) <= 1e-9
        holdings[near] = current[near]
        if cash_for(holdings) < self.cfg.cash_floor:
            # Moving towards maximal permitted sales increases cash monotonically.
            lo, hi = 0., 1.
            for _ in range(64):
                mid = (lo + hi) / 2
                if cash_for(lower + mid * (holdings - lower)) >= self.cfg.cash_floor:
                    lo = mid
                else:
                    hi = mid
            holdings = lower + lo * (holdings - lower)
        utility = lambda h: float(mu @ h - self.cfg.risk_aversion * np.square(factor.T @ h).sum()
                                  - rate * np.abs(h - current).sum())  # noqa: E731
        selection = "solver_candidate"
        if cash_for(current) >= self.cfg.cash_floor and utility(current) >= utility(holdings):
            holdings, selection = current.copy(), "hold_baseline"
        if cash_for(lower) >= self.cfg.cash_floor and utility(lower) > utility(holdings):
            holdings, selection = lower.copy(), "max_sale_baseline"
        fees = rate * float(np.abs(holdings - current).sum())
        nav = 1 - fees
        if nav <= 0:
            raise cp.error.SolverError("nonpositive post-fee NAV")
        return RebalanceDecision(target_weights=holdings / nav, holding_values_fraction=holdings,
            fees_fraction=fees, cash_fraction=max(0., cash_for(holdings)),
            objective_value=utility(holdings),
            solver=solver, status=problem.status, repair_l1=float(np.abs(holdings - raw).sum()), selection=selection)
