"""Vectorised backtest engine + reporting metrics."""
from .engine import run_backtest, BacktestResult
from .factor_ic import daily_factor_ic, DailyFactorIC
from .ic_inference import ic_mean_inference, adjust_pvalues, infer_factor_ic_report, ICMeanInference
from .factor_walk_forward import walk_forward_factor_ic
from .metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    annualised_return, max_drawdown,
    information_coefficient, rank_information_coefficient,
)

__all__ = [
    "run_backtest", "BacktestResult",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "annualised_return", "max_drawdown",
    "information_coefficient", "rank_information_coefficient",
    "daily_factor_ic", "DailyFactorIC",
    "ic_mean_inference", "adjust_pvalues", "infer_factor_ic_report", "ICMeanInference",
    "walk_forward_factor_ic",
]
