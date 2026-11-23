"""Vectorised backtest engine + reporting metrics."""
from .engine import run_backtest, BacktestResult
from .self_financing import run_self_financing_backtest, SelfFinancingResult
from .constrained import run_constrained_backtest, ConstrainedResult, panel_trade_permissions
from .capacity import run_capacity_backtest, lagged_turnover_caps, CapacityResult
from .rebalance import run_rebalance_backtest, RebalancedResult
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
    "run_self_financing_backtest", "SelfFinancingResult",
    "run_constrained_backtest", "ConstrainedResult", "panel_trade_permissions",
    "run_capacity_backtest", "lagged_turnover_caps", "CapacityResult",
    "run_rebalance_backtest", "RebalancedResult",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "annualised_return", "max_drawdown",
    "information_coefficient", "rank_information_coefficient",
    "daily_factor_ic", "DailyFactorIC",
    "ic_mean_inference", "adjust_pvalues", "infer_factor_ic_report", "ICMeanInference",
    "walk_forward_factor_ic",
]
