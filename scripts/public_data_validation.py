"""Public-data validation benchmark with walk-forward backtests.

This script is intentionally conservative: it is a reproducibility and
engineering validation harness, not a profitability claim. It can download a
larger yfinance universe, compute a compact factor set, compare simple and ML
baselines, and report cost/slippage-aware backtest metrics.

Examples
--------
    python scripts/public_data_validation.py --source synthetic --models equal_weight,momentum_20
    python scripts/public_data_validation.py --source yfinance --preset us-large-100 --max-tickers 100
"""
from __future__ import annotations

import csv
import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlquant.backtest import metrics
from mlquant.backtest.engine import run_backtest
from mlquant.data import make_panel
from mlquant.data.panel import Panel
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_legacy_set
from mlquant.features.legacy_factors import LEGACY_REGISTRY
from mlquant.models import MLPRegressor, TransformerRegressor


US_LARGE_100 = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "XOM", "V", "MA", "COST", "HD", "PG", "WMT",
    "NFLX", "JNJ", "ABBV", "BAC", "KO", "ORCL", "CRM", "AMD", "MRK", "CVX",
    "PEP", "TMO", "ADBE", "LIN", "MCD", "CSCO", "ACN", "WFC", "ABT", "QCOM",
    "IBM", "GE", "INTU", "DHR", "TXN", "AMGN", "PM", "NOW", "ISRG", "VZ",
    "CAT", "NEE", "DIS", "PFE", "RTX", "UBER", "GS", "SPGI", "LOW", "UNP",
    "BKNG", "PGR", "T", "HON", "BLK", "TJX", "SYK", "ELV", "ETN", "LMT",
    "VRTX", "COP", "C", "MDT", "ADP", "MU", "CB", "ADI", "PANW", "REGN",
    "APH", "BSX", "KLAC", "PLD", "AMT", "NOC", "DE", "GILD", "SO", "SCHW",
    "ANET", "ICE", "LRCX", "CME", "MO", "WM", "DUK", "MCO", "HCA", "SHW",
)

ETF_50 = (
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "USO",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
    "SMH", "SOXX", "ARKK", "VNQ", "IYR", "KRE", "XBI", "IBB", "TAN", "ICLN",
    "EWT", "EWJ", "EWG", "EWU", "FXI", "MCHI", "INDA", "EWZ", "EWW", "EWC",
)

CN_LARGE_25 = (
    "sh.600000", "sh.600036", "sh.600519", "sh.600585", "sh.600809",
    "sh.600887", "sh.601012", "sh.601088", "sh.601166", "sh.601318",
    "sh.601398", "sh.601668", "sh.601857", "sh.601888", "sh.601899",
    "sz.000001", "sz.000002", "sz.000333", "sz.000651", "sz.000858",
    "sz.002415", "sz.002594", "sz.300750", "sz.300059", "sz.300124",
)

PRESETS = {
    "us-large-100": US_LARGE_100,
    "etf-50": ETF_50,
    "mixed-150": US_LARGE_100 + ETF_50,
    "cn-large-25": CN_LARGE_25,
}

DEFAULT_MODELS = ("equal_weight", "momentum_20", "alpha101_mean", "mlp_alpha101", "transformer_alpha101")


@dataclass(frozen=True)
class ValidationConfig:
    source: str
    preset: str
    tickers: tuple[str, ...]
    start: str
    end: str
    max_tickers: int
    device: str
    costs_bps: float
    slippage_bps: float
    cost_grid_bps: tuple[float, ...]
    bootstrap_samples: int
    bootstrap_block_size: int
    train_window: int
    test_window: int
    step: int
    top_quantile: float
    seed: int
    epochs: int
    batch_size: int
    hidden: int
    models: tuple[str, ...]
    output_dir: Path
    synthetic_dates: int
    synthetic_stocks: int
    command: tuple[str, ...] = ()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_models(value: str) -> tuple[str, ...]:
    requested = tuple(m.strip() for m in value.split(",") if m.strip())
    unknown = sorted(set(requested) - set(DEFAULT_MODELS))
    if unknown:
        raise click.BadParameter(f"unknown model(s): {', '.join(unknown)}")
    return requested


def _parse_cost_grid(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    costs = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            cost = float(item)
        except ValueError as exc:
            raise click.BadParameter(f"invalid cost value {item!r}") from exc
        if cost < 0:
            raise click.BadParameter("cost grid values must be non-negative")
        costs.append(cost)
    return tuple(dict.fromkeys(costs))


def _select_tickers(preset: str, tickers: str, max_tickers: int, source: str = "yfinance") -> tuple[str, ...]:
    if tickers:
        raw = tuple(t.strip() for t in tickers.split(",") if t.strip())
        if source == "baostock":
            selected = raw  # preserve case: baostock uses lowercase exchange prefixes
        else:
            selected = tuple(t.upper() for t in raw)
    else:
        selected = PRESETS[preset]
    return selected[:max_tickers]


def load_validation_panel(cfg: ValidationConfig) -> Panel:
    """Load a public-data, Baostock A-share, or deterministic synthetic panel."""
    if cfg.source == "synthetic":
        return make_synthetic_panel(
            SyntheticConfig(
                n_dates=cfg.synthetic_dates,
                n_stocks=cfg.synthetic_stocks,
                seed=cfg.seed,
                device=cfg.device,
            )
        )
    if cfg.source == "baostock":
        tickers = tuple(t.lower() for t in cfg.tickers)
        invalid = [t for t in tickers if not t.startswith(("sh.", "sz."))]
        if invalid:
            raise ValueError(
                "Baostock validation requires tickers in baostock format "
                f"such as 'sh.600000' or 'sz.000001'; got {invalid[:5]}"
            )
        return make_panel(
            source="baostock",
            tickers=tickers,
            start=cfg.start,
            end=cfg.end,
            device=cfg.device,
        )
    return make_panel(
        source="yfinance",
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        device=cfg.device,
    )


def _alpha101_names() -> tuple[str, ...]:
    return tuple(name for name in LEGACY_REGISTRY if name.startswith("alpha_"))


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _forward_returns(panel: Panel) -> np.ndarray:
    returns = _to_numpy(panel.returns)
    forward = np.zeros_like(returns)
    forward[:-1] = returns[1:]
    return forward


def _valid_forward_mask(panel: Panel, factor_mask: torch.Tensor | None = None) -> np.ndarray:
    valid = _to_numpy(panel.mask).astype(bool)
    valid[:-1] &= valid[1:]
    valid[-1] = False
    if factor_mask is not None:
        valid &= _to_numpy(factor_mask).astype(bool)
    return valid


def _long_only_top_quantile_weights(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    top_quantile: float,
) -> np.ndarray:
    if scores.shape != valid.shape:
        raise ValueError("scores and valid mask must have the same shape")
    weights = np.zeros_like(scores, dtype=np.float32)
    for t in range(scores.shape[0]):
        ok = valid[t] & np.isfinite(scores[t])
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue
        n_pick = max(1, int(math.ceil(n_ok * top_quantile)))
        idx = np.flatnonzero(ok)
        ranked = idx[np.argsort(scores[t, idx])]
        chosen = ranked[-n_pick:]
        weights[t, chosen] = 1.0 / n_pick
    return weights


def _equal_weight(valid: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(valid, dtype=np.float32)
    counts = valid.sum(axis=1)
    for t, count in enumerate(counts):
        if count > 0:
            weights[t, valid[t]] = 1.0 / float(count)
    return weights


def _momentum_scores(panel: Panel, window: int = 20) -> np.ndarray:
    close = _to_numpy(panel.close)
    out = np.zeros_like(close, dtype=np.float32)
    if close.shape[0] <= window:
        return out
    prev = np.clip(close[:-window], 1e-12, None)
    out[window:] = close[window:] / prev - 1.0
    return out


def _alpha101_features(panel: Panel) -> tuple[np.ndarray, np.ndarray, list[str]]:
    factors, factor_mask, names = compute_legacy_set(panel, names=_alpha101_names())
    return _to_numpy(factors), _to_numpy(factor_mask).astype(bool), names


def _walk_forward_splits(n_dates: int, train_window: int, test_window: int, step: int) -> list[tuple[int, int, int, int]]:
    splits = []
    start = train_window
    while start < n_dates - 1:
        train_start = max(0, start - train_window)
        train_end = start
        test_start = start
        test_end = min(n_dates - 1, test_start + test_window)
        if train_end > train_start and test_end > test_start:
            splits.append((train_start, train_end, test_start, test_end))
        start += step
    return splits


def _fit_predict_model(
    model_name: str,
    features: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    cfg: ValidationConfig,
) -> np.ndarray:
    _seed_everything(cfg.seed)
    preds = np.full(target.shape, np.nan, dtype=np.float32)
    splits = _walk_forward_splits(
        features.shape[0],
        train_window=cfg.train_window,
        test_window=cfg.test_window,
        step=cfg.step,
    )
    in_dim = features.shape[-1]
    for split_id, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_valid = valid[train_start:train_end]
        test_valid = valid[test_start:test_end]
        x_train = features[train_start:train_end][train_valid]
        y_train = target[train_start:train_end][train_valid]
        x_test = features[test_start:test_end][test_valid]
        if x_train.shape[0] < max(32, in_dim * 4) or x_test.shape[0] == 0:
            continue

        mu = x_train.mean(axis=0, keepdims=True)
        sigma = x_train.std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma

        if model_name == "mlp_alpha101":
            model: nn.Module = MLPRegressor(in_dim=in_dim, hidden=cfg.hidden, dropout=0.0)
        elif model_name == "transformer_alpha101":
            d_model = max(16, min(64, cfg.hidden))
            n_heads = 4 if d_model % 4 == 0 else 2
            model = TransformerRegressor(in_dim=in_dim, d_model=d_model, n_heads=n_heads, depth=1, dropout=0.0)
        else:
            raise ValueError(f"unsupported ML model {model_name!r}")

        model.to(cfg.device)
        model.train()
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
        for _ in range(cfg.epochs):
            for xb, yb in loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_test.astype(np.float32)).to(cfg.device)).cpu().numpy()
        split_preds = np.full(test_valid.shape, np.nan, dtype=np.float32)
        split_preds[test_valid] = pred.astype(np.float32)
        preds[test_start:test_end] = split_preds
        click.echo(f"trained {model_name} split {split_id + 1}/{len(splits)}")
    return preds


def _summarise_strategy(
    name: str,
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    effective_costs_bps: float,
    benchmark: np.ndarray | None,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    bootstrap_seed: int,
) -> dict[str, float | int | str]:
    result = run_backtest(weights, returns, costs_bps=effective_costs_bps, benchmark=benchmark)
    row: dict[str, float | int | str] = {"strategy": name}
    row.update(result.summary())
    row["effective_costs_bps"] = effective_costs_bps
    row["gross_ann_return"] = metrics.annualised_return(result.gross_returns)
    row["gross_sharpe"] = metrics.sharpe_ratio(result.gross_returns)
    row["final_equity"] = float(result.cumulative_equity[-1]) if result.cumulative_equity.size else 1.0
    row.update(
        _bootstrap_intervals(
            result.portfolio_returns,
            samples=bootstrap_samples,
            block_size=bootstrap_block_size,
            seed=bootstrap_seed,
        )
    )
    return row


def _bootstrap_intervals(
    returns: np.ndarray,
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> dict[str, float | int]:
    if samples <= 0 or returns.size < 2:
        return {}
    block_size = max(1, min(block_size, returns.size))
    n_blocks = int(math.ceil(returns.size / block_size))
    max_start = max(0, returns.size - block_size)
    rng = np.random.default_rng(seed)
    ann_returns = np.empty(samples, dtype=np.float64)
    sharpes = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate([returns[start : start + block_size] for start in starts])[: returns.size]
        ann_returns[i] = metrics.annualised_return(sampled)
        sharpes[i] = metrics.sharpe_ratio(sampled)
    ann_low, ann_high = np.percentile(ann_returns, [2.5, 97.5])
    sharpe_low, sharpe_high = np.percentile(sharpes, [2.5, 97.5])
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_block_size": int(block_size),
        "ann_return_ci_low": float(ann_low),
        "ann_return_ci_high": float(ann_high),
        "sharpe_ci_low": float(sharpe_low),
        "sharpe_ci_high": float(sharpe_high),
    }


def _panel_coverage(panel: Panel) -> dict[str, object]:
    mask = _to_numpy(panel.mask).astype(bool)
    per_stock_count = mask.sum(axis=0)
    no_data = [str(stock) for stock, count in zip(panel.stocks, per_stock_count) if int(count) == 0]
    partial_data = [
        str(stock)
        for stock, count in zip(panel.stocks, per_stock_count)
        if 0 < int(count) < panel.n_dates
    ]
    return {
        "n_dates": panel.n_dates,
        "n_stocks": panel.n_stocks,
        "date_start": str(panel.dates[0]),
        "date_end": str(panel.dates[-1]),
        "tradable_cells": int(mask.sum()),
        "total_cells": int(mask.size),
        "tradable_ratio": float(mask.mean()) if mask.size else 0.0,
        "stocks_with_any_data": int((per_stock_count > 0).sum()),
        "stocks_with_no_data": no_data,
        "stocks_with_partial_data_count": len(partial_data),
        "stocks_with_partial_data_sample": partial_data[:20],
    }


def _metadata(cfg: ValidationConfig, panel: Panel) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(cfg.command),
        "source": cfg.source,
        "preset": cfg.preset,
        "tickers": list(cfg.tickers),
        "start": cfg.start,
        "end": cfg.end,
        "max_tickers": cfg.max_tickers,
        "models": list(cfg.models),
        "costs_bps": cfg.costs_bps,
        "slippage_bps": cfg.slippage_bps,
        "effective_costs_bps": cfg.costs_bps + cfg.slippage_bps,
        "cost_grid_bps": list(cfg.cost_grid_bps),
        "bootstrap_samples": cfg.bootstrap_samples,
        "bootstrap_block_size": cfg.bootstrap_block_size,
        "train_window": cfg.train_window,
        "test_window": cfg.test_window,
        "step": cfg.step,
        "top_quantile": cfg.top_quantile,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "hidden": cfg.hidden,
        "synthetic_dates": cfg.synthetic_dates,
        "synthetic_stocks": cfg.synthetic_stocks,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "unavailable",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "panel": _panel_coverage(panel),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def run_validation(cfg: ValidationConfig) -> list[dict[str, float | int | str]]:
    _seed_everything(cfg.seed)
    panel = load_validation_panel(cfg)
    panel.assert_consistent()
    returns = _to_numpy(panel.returns)
    target = _forward_returns(panel)
    valid_panel = _valid_forward_mask(panel)

    alpha_features: np.ndarray | None = None
    alpha_valid: np.ndarray | None = None
    if any(m in cfg.models for m in ("alpha101_mean", "mlp_alpha101", "transformer_alpha101")):
        alpha_features, alpha_valid, _ = _alpha101_features(panel)
        valid_alpha = valid_panel & alpha_valid
    else:
        valid_alpha = valid_panel

    effective_costs_bps = cfg.costs_bps + cfg.slippage_bps
    strategies: dict[str, np.ndarray] = {}

    if "equal_weight" in cfg.models:
        strategies["equal_weight"] = _equal_weight(valid_panel)
    if "momentum_20" in cfg.models:
        strategies["momentum_20"] = _long_only_top_quantile_weights(
            _momentum_scores(panel, 20),
            valid_panel,
            top_quantile=cfg.top_quantile,
        )
    if "alpha101_mean" in cfg.models:
        assert alpha_features is not None
        scores = np.nanmean(alpha_features, axis=2)
        strategies["alpha101_mean"] = _long_only_top_quantile_weights(
            scores,
            valid_alpha,
            top_quantile=cfg.top_quantile,
        )
    for model_name in ("mlp_alpha101", "transformer_alpha101"):
        if model_name in cfg.models:
            assert alpha_features is not None
            preds = _fit_predict_model(model_name, alpha_features, target, valid_alpha, cfg)
            strategies[model_name] = _long_only_top_quantile_weights(
                preds,
                valid_alpha & np.isfinite(preds),
                top_quantile=cfg.top_quantile,
            )

    benchmark_returns = None
    if "equal_weight" in strategies:
        benchmark_returns = run_backtest(strategies["equal_weight"], returns, costs_bps=effective_costs_bps).portfolio_returns

    rows = []
    for strategy_id, (name, weights) in enumerate(strategies.items()):
        rows.append(
            _summarise_strategy(
                name,
                weights,
                returns,
                effective_costs_bps=effective_costs_bps,
                benchmark=benchmark_returns if name != "equal_weight" else None,
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_block_size=cfg.bootstrap_block_size,
                bootstrap_seed=cfg.seed + strategy_id,
            )
        )
    cost_sensitivity_rows = _cost_sensitivity_rows(strategies, returns, cfg.cost_grid_bps, cfg)
    _write_outputs(cfg, panel, rows, cost_sensitivity_rows)
    return rows


def _cost_sensitivity_rows(
    strategies: dict[str, np.ndarray],
    returns: np.ndarray,
    cost_grid_bps: Sequence[float],
    cfg: ValidationConfig,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for cost_id, effective_costs_bps in enumerate(cost_grid_bps):
        benchmark_returns = None
        if "equal_weight" in strategies:
            benchmark_returns = run_backtest(
                strategies["equal_weight"],
                returns,
                costs_bps=effective_costs_bps,
            ).portfolio_returns
        for strategy_id, (name, weights) in enumerate(strategies.items()):
            row = _summarise_strategy(
                name,
                weights,
                returns,
                effective_costs_bps=effective_costs_bps,
                benchmark=benchmark_returns if name != "equal_weight" else None,
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_block_size=cfg.bootstrap_block_size,
                bootstrap_seed=cfg.seed + 1000 * (cost_id + 1) + strategy_id,
            )
            row["cost_scenario"] = f"{effective_costs_bps:.2f}_bps"
            rows.append(row)
    return rows


def _format_float(value: object) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def _markdown_cell(value: object) -> str:
    """Format a value for a Markdown table cell."""
    return _format_float(value).replace("|", "\\|")


def _markdown_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    columns = [
        "strategy",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_dd",
        "turnover",
        "cost_drag",
        "gross_ann_return",
        "gross_sharpe",
        "info_ratio",
        "alpha_ann",
        "final_equity",
    ]
    if any("sharpe_ci_low" in row for row in rows):
        columns.extend(("ann_return_ci_low", "ann_return_ci_high", "sharpe_ci_low", "sharpe_ci_high"))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _cost_sensitivity_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    columns = [
        "cost_scenario",
        "strategy",
        "effective_costs_bps",
        "ann_return",
        "sharpe",
        "max_dd",
        "turnover",
        "cost_drag",
        "final_equity",
    ]
    if any("sharpe_ci_low" in row for row in rows):
        columns.extend(("ann_return_ci_low", "ann_return_ci_high", "sharpe_ci_low", "sharpe_ci_high"))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---"] + ["---:"] * (len(columns) - 2)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _submission_body(metadata: dict[str, object], rows: Sequence[dict[str, float | int | str]]) -> str:
    panel = metadata["panel"]
    assert isinstance(panel, dict)
    command = metadata.get("command") or "python scripts/public_data_validation.py ..."
    return "\n".join(
        [
            "## Public-data validation report",
            "",
            "### Command",
            "",
            "```bash",
            str(command),
            "```",
            "",
            "### Environment",
            "",
            f"- Python: {metadata['python']}",
            f"- Platform: {metadata['platform']}",
            f"- PyTorch: {metadata['pytorch']}",
            f"- CUDA available: {metadata['cuda_available']}",
            f"- CUDA version: {metadata['cuda_version']}",
            f"- CUDA device: {metadata['cuda_device']}",
            "",
            "### Data Coverage",
            "",
            f"- Source: {metadata['source']}",
            f"- Preset: {metadata['preset']}",
            f"- Dates x stocks: {panel['n_dates']} x {panel['n_stocks']}",
            f"- Date range: {panel['date_start']} to {panel['date_end']}",
            f"- Tradable ratio: {float(panel['tradable_ratio']):.4f}",
            f"- Stocks with no data: {panel['stocks_with_no_data']}",
            f"- Partial-data stock count: {panel['stocks_with_partial_data_count']}",
            "",
            "### Results",
            "",
            _markdown_table(rows),
            "",
            "### Interpretation Notes",
            "",
            "- This is a validation diagnostic, not a trading recommendation.",
            "- Please mention any data-provider warnings, failed tickers, rate limits, or local changes.",
        ]
    )


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    cfg: ValidationConfig,
    panel: Panel,
    rows: Sequence[dict[str, float | int | str]],
    cost_sensitivity_rows: Sequence[dict[str, float | int | str]],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(cfg, panel)
    _write_csv(cfg.output_dir / "summary.csv", rows)

    with (cfg.output_dir / "metadata.json").open("w") as f:
        json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)

    with (cfg.output_dir / "summary.json").open("w") as f:
        json.dump(
            _json_safe(
                {
                    "metadata": metadata,
                    "results": list(rows),
                    "cost_sensitivity": list(cost_sensitivity_rows),
                }
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    with (cfg.output_dir / "submission.md").open("w") as f:
        f.write(_submission_body(metadata, rows))
        f.write("\n")

    with (cfg.output_dir / "summary.md").open("w") as f:
        f.write("# Public-Data Validation Summary\n\n")
        f.write("This report is a reproducibility and engineering diagnostic, not a trading claim.\n\n")
        f.write("| Field | Value |\n| --- | --- |\n")
        f.write(f"| Source | {cfg.source} |\n")
        f.write(f"| Preset | {cfg.preset} |\n")
        f.write(f"| Dates x stocks | {panel.n_dates} x {panel.n_stocks} |\n")
        f.write(f"| Date range | {panel.dates[0]} to {panel.dates[-1]} |\n")
        f.write(f"| Tradable ratio | {metadata['panel']['tradable_ratio']:.4f} |\n")
        f.write(f"| Stocks with no data | {metadata['panel']['stocks_with_no_data']} |\n")
        f.write(f"| Costs + slippage | {cfg.costs_bps:.2f} + {cfg.slippage_bps:.2f} bps |\n")
        if cfg.bootstrap_samples:
            f.write(f"| Bootstrap samples / block size | {cfg.bootstrap_samples} / {cfg.bootstrap_block_size} days |\n")
        f.write(f"| Walk-forward train/test/step | {cfg.train_window}/{cfg.test_window}/{cfg.step} days |\n")
        f.write(f"| Python | {platform.python_version()} |\n")
        f.write(f"| Platform | {platform.platform()} |\n")
        f.write(f"| PyTorch | {torch.__version__} |\n\n")
        f.write(_markdown_table(rows))
        if cost_sensitivity_rows:
            f.write("\n\n## Cost Sensitivity\n\n")
            f.write("The same strategy weights are re-scored under alternative effective cost assumptions.\n\n")
            f.write(_cost_sensitivity_table(cost_sensitivity_rows))
        f.write("\n")

    if cost_sensitivity_rows:
        _write_csv(cfg.output_dir / "cost_sensitivity.csv", cost_sensitivity_rows)
        with (cfg.output_dir / "cost_sensitivity.json").open("w") as f:
            json.dump(_json_safe(list(cost_sensitivity_rows)), f, indent=2, sort_keys=True)
        with (cfg.output_dir / "cost_sensitivity.md").open("w") as f:
            f.write("# Public-Data Cost Sensitivity\n\n")
            f.write("The same strategy weights are re-scored under alternative effective cost assumptions.\n\n")
            f.write(_cost_sensitivity_table(cost_sensitivity_rows))
            f.write("\n")


@click.command()
@click.option("--source", type=click.Choice(["yfinance", "synthetic", "baostock"]), default="yfinance", show_default=True)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default="us-large-100", show_default=True)
@click.option("--tickers", default="", help="Comma-separated tickers. Overrides --preset.")
@click.option("--start", default="2021-01-01", show_default=True)
@click.option("--end", default="2025-01-01", show_default=True)
@click.option("--max-tickers", default=100, show_default=True, type=click.IntRange(1, 500))
@click.option("--device", default="cpu", show_default=True)
@click.option("--costs-bps", default=5.0, show_default=True, type=float)
@click.option("--slippage-bps", default=2.0, show_default=True, type=float)
@click.option("--cost-grid-bps", default="", help="Comma-separated effective cost scenarios for sensitivity reports.")
@click.option("--bootstrap-samples", default=0, show_default=True, type=click.IntRange(0, 10000), help="Block-bootstrap samples for return and Sharpe confidence intervals.")
@click.option("--bootstrap-block-size", default=20, show_default=True, type=click.IntRange(1, 252), help="Trading-day block size for bootstrap intervals.")
@click.option("--train-window", default=504, show_default=True, type=int)
@click.option("--test-window", default=63, show_default=True, type=int)
@click.option("--step", default=63, show_default=True, type=int)
@click.option("--top-quantile", default=0.2, show_default=True, type=click.FloatRange(0.01, 1.0))
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--epochs", default=2, show_default=True, type=int)
@click.option("--batch-size", default=4096, show_default=True, type=int)
@click.option("--hidden", default=64, show_default=True, type=int)
@click.option("--models", default=",".join(DEFAULT_MODELS), show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("artifacts/public_data_validation"), show_default=True)
@click.option("--synthetic-dates", default=260, show_default=True, type=int)
@click.option("--synthetic-stocks", default=80, show_default=True, type=int)
def main(
    source: str,
    preset: str,
    tickers: str,
    start: str,
    end: str,
    max_tickers: int,
    device: str,
    costs_bps: float,
    slippage_bps: float,
    cost_grid_bps: str,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    train_window: int,
    test_window: int,
    step: int,
    top_quantile: float,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    models: str,
    output_dir: Path,
    synthetic_dates: int,
    synthetic_stocks: int,
) -> None:
    """Run public-data walk-forward validation and print a Markdown table."""
    selected_models = _parse_models(models)
    cfg = ValidationConfig(
        source=source,
        preset=preset,
        tickers=_select_tickers(preset, tickers, max_tickers, source),
        start=start,
        end=end,
        max_tickers=max_tickers,
        device=device,
        costs_bps=costs_bps,
        slippage_bps=slippage_bps,
        cost_grid_bps=_parse_cost_grid(cost_grid_bps),
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_size=bootstrap_block_size,
        train_window=train_window,
        test_window=test_window,
        step=step,
        top_quantile=top_quantile,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden=hidden,
        models=selected_models,
        output_dir=output_dir,
        synthetic_dates=synthetic_dates,
        synthetic_stocks=synthetic_stocks,
        command=tuple(sys.argv),
    )

    rows = run_validation(cfg)
    click.echo("")
    click.echo(_markdown_table(rows))
    click.echo("")
    click.echo(f"Wrote reports to {cfg.output_dir}")
    click.echo("Key files: summary.md, summary.csv, summary.json, metadata.json, submission.md")
    if cfg.cost_grid_bps:
        click.echo("Cost sensitivity files: cost_sensitivity.md, cost_sensitivity.csv, cost_sensitivity.json")
    click.echo("Interpret this as validation evidence, not as a deployable trading result.")


if __name__ == "__main__":  # pragma: no cover
    main()
