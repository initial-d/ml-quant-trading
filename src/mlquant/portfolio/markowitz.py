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
    The first call constructs the cvxpy problem. Subsequent calls reuse
    the parameter objects so ``cvxpy`` skips canonicalisation — this
    matters when you backtest 3 000 trading days.
    """

    def __init__(self, cfg: Optional[MarkowitzConfig] = None) -> None:
        self.cfg = cfg or MarkowitzConfig()
        self._n: int | None = None
        self._mu_param: cp.Parameter | None = None
        self._sigma_param: cp.Parameter | None = None
        self._w_var: cp.Variable | None = None
        self._problem: cp.Problem | None = None

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

        sigma = self._covariance(returns_history)
        self._build(n)
        self._mu_param.value = mu.astype(np.float64)
        self._sigma_param.value = self._psd_project(sigma)

        try:
            self._problem.solve(solver=self.cfg.solver, warm_start=True)
        except cp.error.SolverError:
            # Fall back to SCS if the requested solver is unavailable.
            self._problem.solve(solver="SCS", warm_start=True)

        w = np.asarray(self._w_var.value).flatten()
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
    def _psd_project(sigma: np.ndarray) -> np.ndarray:
        # Symmetric, then nudge eigenvalues to be ≥ 0 to keep cvxpy happy.
        sigma = 0.5 * (sigma + sigma.T)
        w, V = np.linalg.eigh(sigma)
        w = np.clip(w, 1e-10, None)
        return (V * w) @ V.T

    def _build(self, n: int) -> None:
        if self._n == n:
            return
        self._n = n
        self._mu_param    = cp.Parameter(n)
        self._sigma_param = cp.Parameter((n, n), PSD=True)
        w = cp.Variable(n)
        objective = cp.Maximize(self._mu_param @ w
                                - self.cfg.risk_aversion * cp.quad_form(w, self._sigma_param))
        constraints = [cp.sum(w) == 1.0 - self.cfg.cash_weight]
        if self.cfg.long_only:
            constraints += [w >= 0, w <= self.cfg.weight_cap]
        else:
            constraints += [w >= -self.cfg.weight_cap, w <= self.cfg.weight_cap]
        self._w_var = w
        self._problem = cp.Problem(objective, constraints)
