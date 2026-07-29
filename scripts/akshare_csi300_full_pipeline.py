"""AkShare CSI 300 full-pipeline public validation.

This script extends the conservative public validation path with a complete
research pipeline:

1. resolve current CSI 300 constituents from AkShare;
2. load public A-share OHLCV data;
3. compute the selected factor set, defaulting to the full registered library;
4. train a walk-forward MLP predictor;
5. neutralize predicted scores cross-sectionally;
6. construct portfolios with top-quantile, buffered, and optimized weights;
7. backtest with transaction costs and write an auditable report.

It is still a public-data diagnostic, not a production strategy or paper claim.
The CSI 300 universe is current membership resolved at runtime, not historical
point-in-time membership.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for local_path in (str(REPO_ROOT), str(SRC_ROOT)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from mlquant.backtest.engine import run_backtest
from mlquant.features.neutralize import neutralize_cs

from scripts.public_data_validation import (
    ValidationConfig,
    _alpha101_features,
    _cost_sensitivity_table,
    _fit_predict_model,
    _forward_returns,
    _json_safe,
    _markdown_table,
    _metadata,
    _parse_cost_grid,
    _seed_everything,
    _select_tickers,
    _summarise_strategy,
    _to_numpy,
    _valid_forward_mask,
    load_validation_panel,
)


@dataclass(frozen=True)
class FullPipelineConfig:
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
    train_window: int
    test_window: int
    step: int
    top_quantile: float
    seed: int
    epochs: int
    batch_size: int
    hidden: int
    factor_set: str
    covariance_window: int
    optimizer_candidates: int
    optimizer_weight_cap: float
    optimizer_risk_aversion: float
    rebalance_step: int
    factor_ic_window: int
    bootstrap_samples: int
    bootstrap_block_size: int
    output_dir: Path
    command: tuple[str, ...] = ()


def _validation_cfg(cfg: FullPipelineConfig) -> ValidationConfig:
    return ValidationConfig(
        source=cfg.source,
        preset=cfg.preset,
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        max_tickers=cfg.max_tickers,
        device=cfg.device,
        costs_bps=cfg.costs_bps,
        slippage_bps=cfg.slippage_bps,
        cost_grid_bps=cfg.cost_grid_bps,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_block_size=cfg.bootstrap_block_size,
        train_window=cfg.train_window,
        test_window=cfg.test_window,
        step=cfg.step,
        top_quantile=cfg.top_quantile,
        seed=cfg.seed,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        hidden=cfg.hidden,
        models=("mlp_alpha101",),
        output_dir=cfg.output_dir,
        synthetic_dates=260,
        synthetic_stocks=80,
        command=cfg.command,
    )


def _neutralize_score_matrix(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score model scores on valid cells only."""
    finite_valid = valid & np.isfinite(scores)
    safe_scores = np.where(finite_valid, scores, 0.0).astype(np.float32)
    with torch.no_grad():
        out = neutralize_cs(torch.from_numpy(safe_scores), torch.from_numpy(finite_valid))
    neutralized = out.numpy().astype(np.float32)
    neutralized[~finite_valid] = np.nan
    return neutralized


def _rebalanced_equal_weight(valid: np.ndarray, *, rebalance_step: int) -> np.ndarray:
    weights = np.zeros_like(valid, dtype=np.float32)
    previous = np.zeros(valid.shape[1], dtype=np.float32)
    rebalance_step = max(1, rebalance_step)
    for t in range(valid.shape[0]):
        if t % rebalance_step == 0:
            previous = np.zeros(valid.shape[1], dtype=np.float32)
            count = int(valid[t].sum())
            if count > 0:
                previous[valid[t]] = 1.0 / float(count)
        weights[t] = previous
    return weights


def _rebalanced_top_quantile_weights(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    top_quantile: float,
    rebalance_step: int,
) -> np.ndarray:
    if scores.shape != valid.shape:
        raise ValueError("scores and valid mask must have the same shape")
    weights = np.zeros_like(scores, dtype=np.float32)
    previous = np.zeros(scores.shape[1], dtype=np.float32)
    rebalance_step = max(1, rebalance_step)
    for t in range(scores.shape[0]):
        if t % rebalance_step == 0:
            previous = np.zeros(scores.shape[1], dtype=np.float32)
            ok = valid[t] & np.isfinite(scores[t])
            n_ok = int(ok.sum())
            if n_ok > 0:
                n_pick = max(1, int(math.ceil(n_ok * top_quantile)))
                idx = np.flatnonzero(ok)
                chosen = idx[np.argsort(scores[t, idx])][-n_pick:]
                previous[chosen] = 1.0 / float(chosen.size)
        weights[t] = previous
    return weights


def _buffered_top_quantile_weights(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    target_quantile: float,
    exit_quantile: float,
) -> np.ndarray:
    """Daily evaluated top-quantile selection with a turnover-control buffer."""
    if scores.shape != valid.shape:
        raise ValueError("scores and valid mask must have the same shape")
    if exit_quantile < target_quantile:
        raise ValueError("exit_quantile must be >= target_quantile")

    weights = np.zeros_like(scores, dtype=np.float32)
    held: set[int] = set()
    for t in range(scores.shape[0]):
        ok = valid[t] & np.isfinite(scores[t])
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            held = set()
            continue

        ranked = idx[np.argsort(scores[t, idx])]
        target_n = max(1, int(math.ceil(idx.size * target_quantile)))
        exit_n = max(target_n, int(math.ceil(idx.size * exit_quantile)))
        keep_zone = set(ranked[-exit_n:].tolist())

        held = {stock for stock in held if stock in keep_zone and ok[stock]}
        for stock in ranked[::-1]:
            if len(held) >= target_n:
                break
            held.add(int(stock))

        if held:
            chosen = np.array(sorted(held), dtype=int)
            weights[t, chosen] = 1.0 / float(chosen.size)
    return weights


def _strategy_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 30:
        return 0.0
    x = x[ok].astype(np.float64)
    y = y[ok].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _rolling_ic_weighted_scores(
    features: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    lookback: int,
    rebalance_step: int,
) -> np.ndarray:
    """Build factor-composite scores from trailing IC estimates only."""
    if features.shape[:2] != target.shape or target.shape != valid.shape:
        raise ValueError("features, target, and valid have mismatched shapes")
    n_dates, n_stocks, n_features = features.shape
    scores = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
    lookback = max(20, lookback)
    rebalance_step = max(1, rebalance_step)

    for t in range(n_dates):
        if t < 20 or t % rebalance_step != 0:
            continue
        start = max(0, t - lookback)
        hist_valid = valid[start:t]
        if int(hist_valid.sum()) < max(100, n_features * 30):
            continue

        weights = np.zeros(n_features, dtype=np.float64)
        y = target[start:t]
        for k in range(n_features):
            weights[k] = _safe_corr(features[start:t, :, k][hist_valid], y[hist_valid])

        scale = float(np.sum(np.abs(weights)))
        if scale <= 1e-12:
            continue
        weights = weights / scale

        current_valid = valid[t] & np.isfinite(features[t]).all(axis=1)
        if not current_valid.any():
            continue
        current = features[t, current_valid].astype(np.float64)
        score = current @ weights
        scores[t, current_valid] = score.astype(np.float32)

    return scores


def _fallback_candidate_weights(
    scores: np.ndarray,
    candidates: np.ndarray,
    n_stocks: int,
) -> np.ndarray:
    weights = np.zeros(n_stocks, dtype=np.float32)
    if candidates.size:
        weights[candidates] = 1.0 / float(candidates.size)
    return weights


def _cap_and_normalize(raw: np.ndarray, cap: float) -> np.ndarray:
    weights = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    cap = float(max(cap, 1.0 / max(1, weights.size)))
    capped = np.zeros_like(weights)
    free = np.ones(weights.size, dtype=bool)
    remaining = 1.0
    for _ in range(weights.size + 1):
        if not free.any():
            break
        candidate = weights[free]
        candidate = candidate / max(float(candidate.sum()), 1e-12) * remaining
        over = candidate > cap
        free_idx = np.flatnonzero(free)
        if not over.any():
            capped[free_idx] = candidate
            break
        capped[free_idx[over]] = cap
        remaining = 1.0 - float(capped.sum())
        free[free_idx[over]] = False
    total = float(capped.sum())
    if total > 0:
        capped /= total
    return capped


def _optimized_candidate_weights(
    scores: np.ndarray,
    returns: np.ndarray,
    valid: np.ndarray,
    *,
    covariance_window: int,
    candidates: int,
    rebalance_step: int,
    risk_aversion: float,
    weight_cap: float,
) -> np.ndarray:
    """Risk-scale top-scored candidates and hold weights between rebalances."""
    if scores.shape != returns.shape or scores.shape != valid.shape:
        raise ValueError("scores, returns, and valid must have the same shape")

    n_dates, n_stocks = scores.shape
    weights = np.zeros_like(scores, dtype=np.float32)
    previous = np.zeros(n_stocks, dtype=np.float32)
    covariance_window = max(5, covariance_window)
    candidates = max(2, candidates)
    rebalance_step = max(1, rebalance_step)

    for t in range(n_dates - 1):
        if t < covariance_window or t % rebalance_step != 0:
            weights[t] = previous
            continue

        ok = valid[t] & np.isfinite(scores[t])
        if not ok.any():
            weights[t] = previous
            continue

        ranked = np.flatnonzero(ok)[np.argsort(scores[t, ok])]
        chosen = ranked[-min(candidates, ranked.size) :]
        hist = returns[t - covariance_window : t, :][:, chosen]
        hist = np.where(np.isfinite(hist), hist, 0.0)

        usable = np.isfinite(hist).all(axis=0) & np.isfinite(scores[t, chosen])
        chosen = chosen[usable]
        hist = hist[:, usable]
        if chosen.size < 2:
            previous = _fallback_candidate_weights(scores[t], chosen, n_stocks)
            weights[t] = previous
            continue

        local_cap = max(weight_cap, 1.0 / float(chosen.size))
        mu = scores[t, chosen].astype(np.float64)
        mu = mu - np.nanmean(mu)
        mu_std = np.nanstd(mu)
        if not np.isfinite(mu_std) or mu_std <= 1e-12:
            previous = _fallback_candidate_weights(scores[t], chosen, n_stocks)
            weights[t] = previous
            continue
        mu = mu / mu_std

        try:
            covariance = LedoitWolf().fit(hist).covariance_
            variance = np.clip(np.diag(covariance), 1e-8, None)
            risk_scaled = mu / np.sqrt(variance * max(risk_aversion, 1e-8))
            raw = risk_scaled - np.nanmin(risk_scaled)
            if not np.isfinite(raw).all() or float(raw.sum()) <= 1e-12:
                raise ValueError("degenerate optimized scores")
            local_weights = _cap_and_normalize(raw, local_cap)
        except Exception:
            previous = _fallback_candidate_weights(scores[t], chosen, n_stocks)
        else:
            previous = np.zeros(n_stocks, dtype=np.float32)
            previous[chosen] = local_weights.astype(np.float32)
        weights[t] = previous

    weights[-1] = previous
    return weights


def _strategy_rows(
    strategies: dict[str, np.ndarray],
    returns: np.ndarray,
    cfg: FullPipelineConfig,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    effective_costs_bps = cfg.costs_bps + cfg.slippage_bps
    benchmark_name = "equal_weight_daily"
    benchmark_returns = None
    if benchmark_name in strategies:
        benchmark_returns = run_backtest(
            strategies[benchmark_name],
            returns,
            costs_bps=effective_costs_bps,
        ).portfolio_returns

    rows = []
    for strategy_id, (name, weights) in enumerate(strategies.items()):
        rows.append(
            _summarise_strategy(
                name,
                weights,
                returns,
                effective_costs_bps=effective_costs_bps,
                benchmark=benchmark_returns if name != benchmark_name else None,
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_block_size=cfg.bootstrap_block_size,
                bootstrap_seed=cfg.seed + strategy_id,
            )
        )

    sensitivity_rows: list[dict[str, float | int | str]] = []
    for cost_id, effective_costs_bps in enumerate(cfg.cost_grid_bps):
        benchmark_returns = None
        if benchmark_name in strategies:
            benchmark_returns = run_backtest(
                strategies[benchmark_name],
                returns,
                costs_bps=effective_costs_bps,
            ).portfolio_returns
        for strategy_id, (name, weights) in enumerate(strategies.items()):
            row = _summarise_strategy(
                name,
                weights,
                returns,
                effective_costs_bps=effective_costs_bps,
                benchmark=benchmark_returns if name != benchmark_name else None,
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_block_size=cfg.bootstrap_block_size,
                bootstrap_seed=cfg.seed + 1000 * (cost_id + 1) + strategy_id,
            )
            row["cost_scenario"] = f"{effective_costs_bps:.2f}_bps"
            sensitivity_rows.append(row)

    return rows, sensitivity_rows


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _full_pipeline_metadata(
    cfg: FullPipelineConfig,
    panel,
    factor_names: Sequence[str],
) -> dict[str, object]:
    metadata = _metadata(_validation_cfg(cfg), panel)
    metadata["pipeline"] = {
        "stages": [
            "akshare_current_csi300_universe",
            "ohlcv_panel",
            "selected_factor_set_features",
            "walk_forward_mlp_training",
            "cross_sectional_score_neutralization",
            "daily_or_configured_top_quantile_selection",
            "daily_or_configured_optimized_candidate_weights",
            "cost_aware_backtest",
        ],
        "model": "mlp_alpha101 implementation trained on selected factor matrix",
        "factor_set": cfg.factor_set,
        "feature_count": len(factor_names),
        "neutralization": "cross_sectional_zscore_on_model_scores",
        "neutralization_limit": "no industry or size neutralization because public AkShare run does not provide those exposures",
        "portfolio_optimizer": "LedoitWolf diagonal risk-scaled long-only mean-variance weights on top scored candidates",
        "factor_ic_window": cfg.factor_ic_window,
        "covariance_window": cfg.covariance_window,
        "optimizer_candidates": cfg.optimizer_candidates,
        "optimizer_weight_cap": cfg.optimizer_weight_cap,
        "optimizer_risk_aversion": cfg.optimizer_risk_aversion,
        "rebalance_step": cfg.rebalance_step,
    }
    metadata["limitations"].append(
        "This full-pipeline run uses cross-sectional score neutralization only; "
        "it does not claim industry or size neutrality."
    )
    return metadata


def _write_outputs(
    cfg: FullPipelineConfig,
    panel,
    factor_names: Sequence[str],
    rows: Sequence[dict[str, float | int | str]],
    sensitivity_rows: Sequence[dict[str, float | int | str]],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _full_pipeline_metadata(cfg, panel, factor_names)
    _write_csv(cfg.output_dir / "summary.csv", rows)
    with (cfg.output_dir / "metadata.json").open("w") as f:
        json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)
    with (cfg.output_dir / "summary.json").open("w") as f:
        json.dump(
            _json_safe(
                {
                    "metadata": metadata,
                    "results": list(rows),
                    "cost_sensitivity": list(sensitivity_rows),
                }
            ),
            f,
            indent=2,
            sort_keys=True,
        )
    with (cfg.output_dir / "summary.md").open("w") as f:
        panel_meta = metadata["panel"]
        pipeline = metadata["pipeline"]
        f.write("# AkShare CSI 300 Full-Pipeline Validation Summary\n\n")
        f.write("This report is a public-data research diagnostic, not a trading claim.\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        f.write(f"| Source | {cfg.source} |\n")
        f.write(f"| Preset | {cfg.preset} |\n")
        f.write(f"| Dates x stocks | {panel.n_dates} x {panel.n_stocks} |\n")
        f.write(f"| Date range | {panel.dates[0]} to {panel.dates[-1]} |\n")
        f.write(f"| Tradable ratio | {panel_meta['tradable_ratio']:.4f} |\n")
        f.write(f"| Model | {pipeline['model']} |\n")
        f.write(f"| Factor set | {pipeline['factor_set']} |\n")
        f.write(f"| Features | {pipeline['feature_count']} factors |\n")
        f.write(f"| Neutralization | {pipeline['neutralization']} |\n")
        f.write(f"| Factor IC window | {cfg.factor_ic_window} trading days |\n")
        f.write(f"| Portfolio optimizer | {pipeline['portfolio_optimizer']} |\n")
        f.write(f"| Optimizer candidates / cap | {cfg.optimizer_candidates} / {cfg.optimizer_weight_cap:.4f} |\n")
        rebalance_label = "daily" if cfg.rebalance_step == 1 else f"every {cfg.rebalance_step} trading days"
        f.write(f"| Rebalance | {rebalance_label} |\n")
        f.write(f"| Costs + slippage | {cfg.costs_bps:.2f} + {cfg.slippage_bps:.2f} bps |\n")
        f.write(f"| Python / PyTorch | {platform.python_version()} / {torch.__version__} |\n\n")
        f.write(_markdown_table(rows))
        if sensitivity_rows:
            f.write("\n\n## Cost Sensitivity\n\n")
            f.write(_cost_sensitivity_table(sensitivity_rows))
        f.write("\n\n## Interpretation Notes\n\n")
        f.write("- The model is trained walk-forward on the selected public-data factor set.\n")
        f.write("- Score neutralization is cross-sectional only, not industry or size neutralization.\n")
        f.write("- Optimized portfolios are constrained to top-scored candidates to keep the public run tractable.\n")
        f.write("- Read net return together with gross return, turnover, and cost drag.\n")
        f.write("- This is not investment advice and not a full paper reproduction.\n")


def run_full_pipeline(cfg: FullPipelineConfig) -> list[dict[str, float | int | str]]:
    _seed_everything(cfg.seed)
    validation_cfg = _validation_cfg(cfg)
    click.echo("stage: loading AkShare panel")
    panel = load_validation_panel(validation_cfg)
    panel.assert_consistent()
    click.echo(f"stage: panel loaded ({panel.n_dates} dates x {panel.n_stocks} stocks)")

    returns = _to_numpy(panel.returns)
    target = _forward_returns(panel)
    valid_panel = _valid_forward_mask(panel)
    click.echo(f"stage: computing factor set {cfg.factor_set}")
    if cfg.factor_set == "alpha101":
        features, factor_valid, factor_names = _alpha101_features(panel)
    elif cfg.factor_set == "all":
        from mlquant.features import compute_legacy_set

        factor_tensor, factor_mask, factor_names = compute_legacy_set(panel)
        features = _to_numpy(factor_tensor)
        factor_valid = _to_numpy(factor_mask).astype(bool)
    else:
        raise ValueError(f"unknown factor set: {cfg.factor_set!r}")
    click.echo(f"stage: computed {len(factor_names)} factors")
    valid_alpha = valid_panel & factor_valid

    click.echo("stage: training walk-forward MLP")
    raw_preds = _fit_predict_model("mlp_alpha101", features, target, valid_alpha, validation_cfg)
    click.echo("stage: building scores")
    neutralized_preds = _neutralize_score_matrix(raw_preds, valid_alpha)
    alpha_scores = np.nanmean(features, axis=2)
    neutralized_alpha = _neutralize_score_matrix(alpha_scores, valid_alpha)
    ic_weighted_scores = _rolling_ic_weighted_scores(
        features,
        target,
        valid_alpha,
        lookback=cfg.factor_ic_window,
        rebalance_step=cfg.rebalance_step,
    )
    neutralized_ic_weighted = _neutralize_score_matrix(ic_weighted_scores, valid_alpha)

    click.echo("stage: building daily portfolios")
    strategies = {
        "equal_weight_daily": _rebalanced_equal_weight(valid_panel, rebalance_step=cfg.rebalance_step),
    }
    strategies.update(
        {
            "factor_mean_daily": _rebalanced_top_quantile_weights(
                neutralized_alpha,
                valid_alpha & np.isfinite(neutralized_alpha),
                top_quantile=cfg.top_quantile,
                rebalance_step=cfg.rebalance_step,
            ),
            "factor_mean_buffered_daily": _buffered_top_quantile_weights(
                neutralized_alpha,
                valid_alpha & np.isfinite(neutralized_alpha),
                target_quantile=cfg.top_quantile,
                exit_quantile=min(1.0, cfg.top_quantile * 2.0),
            ),
            "factor_ic_weighted_daily": _rebalanced_top_quantile_weights(
                neutralized_ic_weighted,
                valid_alpha & np.isfinite(neutralized_ic_weighted),
                top_quantile=cfg.top_quantile,
                rebalance_step=cfg.rebalance_step,
            ),
            "factor_mean_optimized_daily": _optimized_candidate_weights(
                neutralized_alpha,
                returns,
                valid_alpha & np.isfinite(neutralized_alpha),
                covariance_window=cfg.covariance_window,
                candidates=cfg.optimizer_candidates,
                rebalance_step=cfg.rebalance_step,
                risk_aversion=cfg.optimizer_risk_aversion,
                weight_cap=cfg.optimizer_weight_cap,
            ),
            "factor_ic_optimized_daily": _optimized_candidate_weights(
                neutralized_ic_weighted,
                returns,
                valid_alpha & np.isfinite(neutralized_ic_weighted),
                covariance_window=cfg.covariance_window,
                candidates=cfg.optimizer_candidates,
                rebalance_step=cfg.rebalance_step,
                risk_aversion=cfg.optimizer_risk_aversion,
                weight_cap=cfg.optimizer_weight_cap,
            ),
            "mlp_top20_raw_daily": _rebalanced_top_quantile_weights(
                raw_preds,
                valid_alpha & np.isfinite(raw_preds),
                top_quantile=cfg.top_quantile,
                rebalance_step=cfg.rebalance_step,
            ),
            "mlp_top20_neutralized_daily": _rebalanced_top_quantile_weights(
                neutralized_preds,
                valid_alpha & np.isfinite(neutralized_preds),
                top_quantile=cfg.top_quantile,
                rebalance_step=cfg.rebalance_step,
            ),
            "mlp_optimized_daily": _optimized_candidate_weights(
                neutralized_preds,
                returns,
                valid_alpha & np.isfinite(neutralized_preds),
                covariance_window=cfg.covariance_window,
                candidates=cfg.optimizer_candidates,
                rebalance_step=cfg.rebalance_step,
                risk_aversion=cfg.optimizer_risk_aversion,
                weight_cap=cfg.optimizer_weight_cap,
            ),
        }
    )

    rows, sensitivity_rows = _strategy_rows(strategies, returns, cfg)
    click.echo("stage: writing reports")
    _write_outputs(cfg, panel, factor_names, rows, sensitivity_rows)
    return rows


@click.command()
@click.option("--preset", type=click.Choice(["csi-300", "hs300"]), default="csi-300", show_default=True)
@click.option("--tickers", default="", help="Comma-separated A-share tickers. Overrides --preset.")
@click.option("--start", default="2021-01-01", show_default=True)
@click.option("--end", default="2025-01-01", show_default=True)
@click.option("--max-tickers", default=300, show_default=True, type=click.IntRange(2, 500))
@click.option("--device", default="cpu", show_default=True)
@click.option("--costs-bps", default=5.0, show_default=True, type=float)
@click.option("--slippage-bps", default=2.0, show_default=True, type=float)
@click.option("--cost-grid-bps", default="0,7,15,30", show_default=True)
@click.option("--train-window", default=504, show_default=True, type=int)
@click.option("--test-window", default=63, show_default=True, type=int)
@click.option("--step", default=63, show_default=True, type=int)
@click.option("--top-quantile", default=0.2, show_default=True, type=click.FloatRange(0.01, 1.0))
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--epochs", default=1, show_default=True, type=int)
@click.option("--batch-size", default=4096, show_default=True, type=int)
@click.option("--hidden", default=32, show_default=True, type=int)
@click.option("--factor-set", type=click.Choice(["all", "alpha101"]), default="all", show_default=True)
@click.option("--covariance-window", default=126, show_default=True, type=int)
@click.option("--optimizer-candidates", default=60, show_default=True, type=click.IntRange(2, 300))
@click.option("--optimizer-weight-cap", default=0.05, show_default=True, type=click.FloatRange(0.001, 1.0))
@click.option("--optimizer-risk-aversion", default=1.0, show_default=True, type=float)
@click.option("--rebalance-step", default=1, show_default=True, type=click.IntRange(1, 252), help="1 means daily rebalancing; larger values are sensitivity checks.")
@click.option("--factor-ic-window", default=252, show_default=True, type=click.IntRange(20, 756))
@click.option("--bootstrap-samples", default=100, show_default=True, type=click.IntRange(0, 10000))
@click.option("--bootstrap-block-size", default=20, show_default=True, type=click.IntRange(1, 252))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/akshare_csi300_full_pipeline"),
    show_default=True,
)
def main(
    preset: str,
    tickers: str,
    start: str,
    end: str,
    max_tickers: int,
    device: str,
    costs_bps: float,
    slippage_bps: float,
    cost_grid_bps: str,
    train_window: int,
    test_window: int,
    step: int,
    top_quantile: float,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    factor_set: str,
    covariance_window: int,
    optimizer_candidates: int,
    optimizer_weight_cap: float,
    optimizer_risk_aversion: float,
    rebalance_step: int,
    factor_ic_window: int,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    output_dir: Path,
) -> None:
    """Run the AkShare CSI 300 full-pipeline public validation."""
    click.echo("stage: resolving CSI 300 universe")
    selected_tickers = _select_tickers(preset, tickers, max_tickers, source="akshare")
    click.echo(f"stage: resolved {len(selected_tickers)} tickers")
    cfg = FullPipelineConfig(
        source="akshare",
        preset=preset,
        tickers=selected_tickers,
        start=start,
        end=end,
        max_tickers=max_tickers,
        device=device,
        costs_bps=costs_bps,
        slippage_bps=slippage_bps,
        cost_grid_bps=_parse_cost_grid(cost_grid_bps),
        train_window=train_window,
        test_window=test_window,
        step=step,
        top_quantile=top_quantile,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden=hidden,
        factor_set=factor_set,
        covariance_window=covariance_window,
        optimizer_candidates=optimizer_candidates,
        optimizer_weight_cap=optimizer_weight_cap,
        optimizer_risk_aversion=optimizer_risk_aversion,
        rebalance_step=rebalance_step,
        factor_ic_window=factor_ic_window,
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_size=bootstrap_block_size,
        output_dir=output_dir,
        command=tuple(sys.argv),
    )

    rows = run_full_pipeline(cfg)
    click.echo("")
    click.echo(_markdown_table(rows))
    click.echo("")
    click.echo(f"Wrote reports to {cfg.output_dir}")
    click.echo("Key files: summary.md, summary.csv, summary.json, metadata.json")
    click.echo("Interpret this as a full-pipeline public-data diagnostic, not a trading claim.")


if __name__ == "__main__":  # pragma: no cover
    main()
