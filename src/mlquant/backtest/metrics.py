"""Performance and information-content metrics.

All functions take 1-D numpy arrays so they can be unit-tested without
torch. The conventions follow Bailey & López de Prado (2014):

    annualised return  := (1 + mean(r))**P - 1
    annualised vol     := std(r) * sqrt(P)
    Sharpe             := annualised return / annualised vol
    Sortino            := annualised return / (downside std * sqrt(P))
    Calmar             := annualised return / max drawdown

with ``P = 252`` for daily series.
"""
from __future__ import annotations

import numpy as np

DAYS_PER_YEAR = 252


def annualised_return(returns: np.ndarray, *, periods: int = DAYS_PER_YEAR) -> float:
    if returns.size == 0:
        return 0.0
    return float((1.0 + returns).prod() ** (periods / returns.size) - 1.0)


def annualised_vol(returns: np.ndarray, *, periods: int = DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods)) if returns.size > 1 else 0.0


def sharpe_ratio(returns: np.ndarray, *, rf: float = 0.0, periods: int = DAYS_PER_YEAR) -> float:
    excess = returns - rf / periods
    vol = excess.std(ddof=1)
    if vol < 1e-12:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, *, rf: float = 0.0, periods: int = DAYS_PER_YEAR) -> float:
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")
    if downside.size < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if not np.isfinite(dd) or dd < 1e-12:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max())


def calmar_ratio(returns: np.ndarray, *, periods: int = DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if mdd < 1e-12:
        return float("inf")
    return annualised_return(returns, periods=periods) / mdd


def information_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson IC. NaNs are dropped pairwise."""
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 2:
        return 0.0
    p = pred[valid]
    t = target[valid]
    pm = p - p.mean()
    tm = t - t.mean()
    denom = np.sqrt((pm ** 2).sum() * (tm ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((pm * tm).sum() / denom)


def rank_information_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman IC."""
    from scipy.stats import rankdata
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 2:
        return 0.0
    return information_coefficient(rankdata(pred[valid]), rankdata(target[valid]))


def turnover(weights: np.ndarray) -> float:
    """Average L1 turnover between consecutive weight vectors."""
    if weights.shape[0] < 2:
        return 0.0
    return float(0.5 * np.abs(weights[1:] - weights[:-1]).sum(axis=1).mean())
