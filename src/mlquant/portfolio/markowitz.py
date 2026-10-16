"""Cross-sectional Markowitz mean-variance optimisation.

Problem
-------
For a given prediction vector :math:`\\mu \\in \\mathbb{R}^n` and a
covariance matrix :math:`\\Sigma`, find weights :math:`w \\in
\\mathbb{R}^n` solving

.. math::

    \\max_w \\; \\mu^\\top w - \\alpha \\, w^\\top \\Sigma w
    \\quad
    \\text{s.t.} \\;
    \\mathbf{1}^\\top w = 1, \\;
    0 \\le w \\le w_{\\max}.

We solve this via :mod:`cvxpy` so the default install needs **no
commercial solver**: ECOS or SCS handle 1 000-asset problems in
milliseconds. The MOSEK path remains available for users with a
licence by selecting ``solver="MOSEK"``.

Covariance estimation uses Ledoit–Wolf shrinkage by default, which is
both faster and more numerically stable than a sample covariance on
short look-back windows.

The risk term is written as :math:`\\lVert F^\\top w \\rVert_2^2` with
:math:`\\Sigma = F F^\\top`, not as ``quad_form(w, Sigma)``. A covariance
*parameter* inside ``quad_form`` is not DPP, so cvxpy would re-canonicalise
the problem on every solve and would also try to certify the parameter
PSD numerically, which fails intermittently on rank-deficient
look-back windows. The factor form is DPP: the cached problem is reused
across dates and no PSD check runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class MarkowitzConfig:
    risk_aversion: float = 1.0
    weight_cap:    float = 0.05         # max single-name weight (5 %)
    long_only:     bool  = True
    cash_weight:   float = 0.0          # leave x % in cash
    solver:        str   = "SCS"        # "SCS" / "ECOS" / "MOSEK"
    shrinkage:     bool  = True


class MarkowitzOptimizer:
    """Stateful optimiser; build once, call :meth:`solve` per date.

    Parameters
    ----------
    cfg : MarkowitzConfig

    Notes
    -----
    One cvxpy problem is built per universe size ``n`` and cached.
    Later calls with the same ``n`` only update parameter values, so
    cvxpy skips canonicalisation — this matters when you backtest
    3 000 trading days. The tradable universe changes size when names
    halt, so a backtest typically builds a handful of problems.
    """

    def __init__(self, cfg: Optional[MarkowitzConfig] = None) -> None:
        self.cfg = cfg or MarkowitzConfig()
        self._problems: dict[int, tuple[cp.Parameter, cp.Parameter, cp.Variable, cp.Problem]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, mu: np.ndarray, returns_history: np.ndarray) -> np.ndarray:
        """Compute portfolio weights for one date.

        Parameters
        ----------
        mu : np.ndarray [n]
            Predicted next-period returns (alpha signal).
        returns_history : np.ndarray [T, n]
            Historical returns used to estimate covariance.

        Returns
        -------
        w : np.ndarray [n]
            Long-only (or long-short) weights summing to ``1 - cash_weight``.
        """
        n = mu.shape[0]
        if returns_history.shape[1] != n:
            raise ValueError("mu and returns_history have mismatched stock dimensions")
        budget = 1.0 - self.cfg.cash_weight
        if n * self.cfg.weight_cap < budget - 1e-12:
            raise ValueError(
                f"infeasible: {n} names x weight_cap {self.cfg.weight_cap} cannot reach "
                f"the invested weight {budget}"
            )

        mu_param, factor_param, w_var, problem = self._problem_for(n)
        mu_param.value = mu.astype(np.float64)
        factor_param.value = self._factor(self._covariance(returns_history))

        try:
            problem.solve(solver=self.cfg.solver, warm_start=True)
        except cp.error.SolverError:
            # Fall back to SCS if the requested solver is unavailable.
            problem.solve(solver="SCS", warm_start=True)
        if w_var.value is None:
            raise cp.error.SolverError(f"no solution (status {problem.status})")

        w = np.asarray(w_var.value).flatten()
        # Numerical fix-ups
        w[np.abs(w) < 1e-7] = 0.0
        if self.cfg.long_only:
            w = np.clip(w, 0.0, self.cfg.weight_cap)
        # Renormalise to compensate for clipping rounding.
        s = w.sum()
        if s > 0:
            w *= (1.0 - self.cfg.cash_weight) / s
        return w

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _covariance(self, history: np.ndarray) -> np.ndarray:
        if not self.cfg.shrinkage:
            return np.cov(history, rowvar=False)
        # LedoitWolf is dramatically more stable on short windows.
        return LedoitWolf().fit(history).covariance_

    @staticmethod
    def _factor(sigma: np.ndarray) -> np.ndarray:
        """``F`` with ``F @ F.T`` equal to ``sigma``, eigenvalues floored at 1e-10."""
        sigma = 0.5 * (sigma + sigma.T)
        w, V = np.linalg.eigh(sigma)
        return V * np.sqrt(np.clip(w, 1e-10, None))

    def _problem_for(self, n: int) -> tuple[cp.Parameter, cp.Parameter, cp.Variable, cp.Problem]:
        cached = self._problems.get(n)
        if cached is not None:
            return cached
        mu_param = cp.Parameter(n)
        factor_param = cp.Parameter((n, n))
        w = cp.Variable(n)
        objective = cp.Maximize(mu_param @ w
                                - self.cfg.risk_aversion * cp.sum_squares(factor_param.T @ w))
        constraints = [cp.sum(w) == 1.0 - self.cfg.cash_weight]
        if self.cfg.long_only:
            constraints += [w >= 0, w <= self.cfg.weight_cap]
        else:
            constraints += [w >= -self.cfg.weight_cap, w <= self.cfg.weight_cap]
        problem = cp.Problem(objective, constraints)
        self._problems[n] = (mu_param, factor_param, w, problem)
        return self._problems[n]
