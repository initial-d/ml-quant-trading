from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts.akshare_csi300_full_pipeline import (
    _buffered_top_quantile_weights,
    _optimized_candidate_weights,
    _neutralize_score_matrix,
    _rebalanced_equal_weight,
    _rebalanced_top_quantile_weights,
    _rolling_ic_weighted_scores,
    _write_equity_curves,
)


def test_neutralize_score_matrix_is_cross_sectional_mean_zero():
    scores = np.array(
        [
            [1.0, 2.0, 3.0, np.nan],
            [2.0, 2.0, 2.0, 5.0],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
        ]
    )

    out = _neutralize_score_matrix(scores, valid)

    assert np.isfinite(out[valid]).all()
    assert np.isnan(out[~valid]).all()
    assert abs(float(out[0, valid[0]].mean())) < 1e-6
    assert abs(float(out[1, valid[1]].mean())) < 1e-6


def test_optimized_candidate_weights_shape_and_budget():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=(40, 8)).astype(np.float32)
    scores = rng.normal(size=(40, 8)).astype(np.float32)
    valid = np.ones((40, 8), dtype=bool)

    weights = _optimized_candidate_weights(
        scores,
        returns,
        valid,
        covariance_window=10,
        candidates=5,
        rebalance_step=5,
        risk_aversion=1.0,
        weight_cap=0.25,
    )

    assert weights.shape == returns.shape
    assert np.all(weights >= -1e-7)
    invested = weights[10:].sum(axis=1)
    assert np.nanmax(invested) <= 1.0001
    assert np.nanmax(invested) > 0.99


def test_rebalanced_equal_weight_holds_between_rebalances():
    valid = np.ones((8, 4), dtype=bool)
    valid[5:, 0] = False

    weights = _rebalanced_equal_weight(valid, rebalance_step=5)

    assert np.allclose(weights[0], [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(weights[1], weights[0])
    assert np.allclose(weights[4], weights[0])
    assert np.allclose(weights[5], [0.0, 1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(weights[7], weights[5])


def test_rebalanced_top_quantile_holds_between_rebalances():
    scores = np.array(
        [
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(scores, dtype=bool)

    weights = _rebalanced_top_quantile_weights(
        scores,
        valid,
        top_quantile=0.25,
        rebalance_step=5,
    )

    assert np.allclose(weights[0], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(weights[4], weights[0])
    assert np.allclose(weights[5], [1.0, 0.0, 0.0, 0.0])


def test_buffered_top_quantile_reduces_churn_inside_exit_band():
    scores = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 2.0, 1.0],
            [2.0, 4.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(scores, dtype=bool)

    weights = _buffered_top_quantile_weights(
        scores,
        valid,
        target_quantile=0.25,
        exit_quantile=0.5,
    )

    assert np.allclose(weights[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(weights[1], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(weights[2], [0.0, 1.0, 0.0, 0.0])


def test_rolling_ic_weighted_scores_do_not_use_future_targets():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(80, 6, 3)).astype(np.float32)
    target = (0.1 * features[:, :, 0] - 0.2 * features[:, :, 1]).astype(np.float32)
    valid = np.ones((80, 6), dtype=bool)

    scores = _rolling_ic_weighted_scores(
        features,
        target,
        valid,
        lookback=30,
        rebalance_step=5,
    )
    changed_future = target.copy()
    changed_future[61:] *= -100.0
    changed_scores = _rolling_ic_weighted_scores(
        features,
        changed_future,
        valid,
        lookback=30,
        rebalance_step=5,
    )

    assert np.allclose(scores[:60], changed_scores[:60], equal_nan=True)


def test_write_equity_curves_exports_daily_series(tmp_path):
    returns = np.array(
        [
            [0.00, 0.00],
            [0.10, 0.00],
            [0.00, 0.10],
        ],
        dtype=np.float32,
    )
    strategies = {
        "equal_weight_daily": np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
    }
    path = tmp_path / "equity_curves.csv"

    _write_equity_curves(
        path,
        ["2021-01-01", "2021-01-04", "2021-01-05"],
        strategies,
        returns,
        effective_costs_bps=0.0,
    )

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["date"] == "2021-01-01"
    assert rows[1]["date"] == "2021-01-04"
    assert float(rows[-1]["equal_weight_daily_equity"]) > 1.10
    assert "equal_weight_daily_return" in rows[0]


def test_full_pipeline_separates_signal_and_forward_label_masks(tmp_path, monkeypatch):
    from dataclasses import replace
    import scripts.akshare_csi300_full_pipeline as full
    from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel

    panel = make_synthetic_panel(SyntheticConfig(n_dates=60, n_stocks=8, halt_prob=0, seed=3))
    mask = panel.mask.clone()
    mask[31, 0] = False
    panel = replace(panel, mask=mask)
    cfg = full.FullPipelineConfig(
        source="synthetic", preset="custom", tickers=(), start="2021-01-01",
        end="2022-01-01", max_tickers=8, device="cpu", costs_bps=5, slippage_bps=0,
        cost_grid_bps=(), train_window=30, test_window=10, step=10, top_quantile=0.25,
        seed=3, epochs=1, batch_size=64, hidden=8, factor_set="alpha101",
        covariance_window=10, optimizer_candidates=5, optimizer_weight_cap=0.25,
        optimizer_risk_aversion=1, rebalance_step=1, factor_ic_window=20,
        min_tradable_ratio=0.9, bootstrap_samples=0, bootstrap_block_size=10,
        output_dir=tmp_path,
    )
    seen = {}

    def predict(model, features, target, labels, validation_cfg, *, prediction_valid=None, diagnostics=None):
        if prediction_valid is None:
            prediction_valid = labels
        seen["labels"] = labels.copy()
        seen["signals"] = prediction_valid.copy()
        return np.where(prediction_valid, features[:, :, 0], np.nan)

    def write(config, data, names, rows, sensitivity, strategies, returns, *, diagnostics=None):
        seen["equal_weight"] = strategies["equal_weight_daily"]

    monkeypatch.setattr(full, "load_validation_panel", lambda cfg: panel)
    monkeypatch.setattr(full, "_alpha101_features", lambda panel: (
        np.ones((60, 8, 3), dtype=np.float32), mask.numpy().copy(), ["a", "b", "c"],
    ))
    monkeypatch.setattr(full, "_fit_predict_model", predict)
    monkeypatch.setattr(full, "_write_outputs", write)
    full.run_full_pipeline(cfg)

    assert seen["equal_weight"][30, 0] > 0
    assert seen["equal_weight"][31, 0] == 0
    assert seen["signals"][30, 0]
    assert not seen["labels"][30, 0]


def test_rolling_ic_ignores_missing_labels_but_scores_current_universe():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(80, 8, 2)).astype(np.float32)
    target = features[:, :, 0] * 0.01
    signal_valid = np.ones((80, 8), dtype=bool)
    label_valid = signal_valid.copy()
    label_valid[20:30, 0] = False
    changed = target.copy()
    changed[~label_valid] = 1e8
    kwargs = dict(lookback=30, rebalance_step=1, label_valid=label_valid)
    clean = _rolling_ic_weighted_scores(features, target, signal_valid, **kwargs)
    poisoned = _rolling_ic_weighted_scores(features, changed, signal_valid, **kwargs)
    assert np.allclose(clean, poisoned, equal_nan=True)
    assert np.isfinite(clean[25, 0])


@pytest.mark.parametrize("factor_available", [False, True])
def test_full_pipeline_reports_skipped_training_and_empty_portfolios(tmp_path, monkeypatch, factor_available):
    """A no-label model must be reported as skipped, not as zero-return skill."""
    import json
    import scripts.akshare_csi300_full_pipeline as full
    from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel

    panel = make_synthetic_panel(SyntheticConfig(n_dates=60, n_stocks=8, halt_prob=0, seed=3))
    cfg = full.FullPipelineConfig(
        source="synthetic", preset="custom", tickers=(), start="2021-01-01",
        end="2022-01-01", max_tickers=8, device="cpu", costs_bps=5, slippage_bps=2,
        cost_grid_bps=(), train_window=30, test_window=10, step=10, top_quantile=0.25,
        seed=3, epochs=1, batch_size=64, hidden=8, factor_set="alpha101",
        covariance_window=10, optimizer_candidates=5, optimizer_weight_cap=0.25,
        optimizer_risk_aversion=1, rebalance_step=1, factor_ic_window=20,
        min_tradable_ratio=0.9, bootstrap_samples=0, bootstrap_block_size=10,
        output_dir=tmp_path,
    )
    monkeypatch.setattr(full, "load_validation_panel", lambda cfg: panel)
    monkeypatch.setattr(full, "_alpha101_features", lambda panel: (
        np.ones((60, 8, 3), dtype=np.float32), np.full((60, 8), factor_available, dtype=bool), ["a", "b", "c"],
    ))
    full.run_full_pipeline(cfg)
    report = json.loads((tmp_path / "summary.json").read_text())
    diagnostics = report["diagnostics"]
    model = diagnostics["model"]
    portfolio = diagnostics["strategies"]["mlp_top20_raw_daily"]
    if factor_available:
        assert diagnostics["factor_signal_cells"] == 472  # synthetic row 0 is unavailable
        assert model["trained_splits"] == 3
        assert model["finite_prediction_cells"] == 232
        assert portfolio["invested_dates"] == portfolio["pnl_exposure_dates"] == 29
        assert portfolio["first_pnl_exposure_date"] == str(panel.dates[31])
        assert any("not matched-window" in warning for warning in diagnostics["warnings"])
    else:
        assert diagnostics["factor_signal_cells"] == 0
        assert model["trained_splits"] == 0
        assert model["skipped_splits"] == 3
        assert all(split["status"] == "insufficient_training_samples" for split in model["splits"])
        assert portfolio["invested_dates"] == 0
        assert any("No walk-forward split" in warning for warning in diagnostics["warnings"])
        assert "not evidence of model performance" in (tmp_path / "summary.md").read_text()
