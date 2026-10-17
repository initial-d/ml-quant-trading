import json
import unittest.mock as mock
from pathlib import Path

import click
import pandas as pd
import pytest

from scripts.public_data_validation import (
    ValidationConfig,
    _codes_from_frame,
    _cost_sensitivity_table,
    _limit_masked_panel,
    _long_only_top_quantile_weights,
    _markdown_table,
    _normalize_akshare_tickers,
    _normalize_baostock_tickers,
    _select_tickers,
    load_validation_panel,
    run_validation,
)


def test_public_data_validation_runs_on_synthetic(tmp_path: Path):
    cfg = ValidationConfig(
        source="synthetic",
        preset="us-large-100",
        tickers=(),
        start="2021-01-01",
        end="2022-01-01",
        max_tickers=12,
        device="cpu",
        costs_bps=5.0,
        slippage_bps=2.0,
        cost_grid_bps=(0.0, 7.0, 15.0),
        bootstrap_samples=25,
        bootstrap_block_size=10,
        train_window=40,
        test_window=20,
        step=20,
        top_quantile=0.25,
        seed=7,
        epochs=1,
        batch_size=128,
        hidden=16,
        models=("equal_weight", "momentum_20", "alpha101_mean"),
        output_dir=tmp_path,
        synthetic_dates=90,
        synthetic_stocks=12,
    )

    rows = run_validation(cfg)

    assert [row["strategy"] for row in rows] == ["equal_weight", "momentum_20", "alpha101_mean"]
    assert all("sharpe" in row for row in rows)
    assert all("sharpe_ci_low" in row for row in rows)
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "submission.md").exists()
    assert (tmp_path / "cost_sensitivity.md").exists()
    assert (tmp_path / "cost_sensitivity.csv").exists()
    assert (tmp_path / "cost_sensitivity.json").exists()

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["panel"]["n_stocks"] == 12
    assert metadata["cost_grid_bps"] == [0.0, 7.0, 15.0]
    assert metadata["bootstrap_samples"] == 25
    assert metadata["bootstrap_block_size"] == 10
    assert metadata["bootstrap_method"] == "moving-block bootstrap over portfolio returns"
    assert metadata["bootstrap_confidence_level"] == 0.95
    assert metadata["bootstrap_seed_base"] == 7
    assert "tradable_ratio" in metadata["panel"]

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["cost_sensitivity"]) == 9
    assert summary["cost_sensitivity"][0]["effective_costs_bps"] == 0.0
    assert summary["results"][0]["bootstrap_samples"] == 25
    assert summary["results"][0]["bootstrap_method"] == "moving-block bootstrap over portfolio returns"
    assert summary["results"][0]["bootstrap_confidence_level"] == 0.95
    assert summary["results"][0]["bootstrap_seed"] == 7

    submission = (tmp_path / "submission.md").read_text()
    assert "Public-data validation report" in submission
    assert "Data Coverage" in submission
    assert "Bootstrap intervals use a moving-block bootstrap" in submission


def test_public_data_validation_markdown_escapes_pipes():
    table = _markdown_table(
        [
            {
                "strategy": "alpha | beta",
                "ann_return": 0.1,
                "ann_vol": 0.2,
                "sharpe": 0.3,
                "max_dd": 0.4,
                "turnover": 0.5,
                "cost_drag": 0.6,
                "gross_ann_return": 0.7,
                "gross_sharpe": 0.8,
                "info_ratio": 0.9,
                "alpha_ann": 1.0,
                "final_equity": 1.1,
            }
        ]
    )

    assert "alpha \\| beta" in table


def test_top_quantile_weights_reject_invalid_quantiles():
    scores = pd.DataFrame([[1.0, 2.0, 3.0]]).to_numpy()
    valid = pd.DataFrame([[True, True, True]]).to_numpy()

    with pytest.raises(ValueError, match="top_quantile"):
        _long_only_top_quantile_weights(scores, valid, top_quantile=0.0)

    with pytest.raises(ValueError, match="top_quantile"):
        _long_only_top_quantile_weights(scores, valid, top_quantile=1.5)


def test_normalize_baostock_tickers_lowercases_valid_codes():
    tickers = _normalize_baostock_tickers(("SH.600000", "sz.000001"))

    assert tickers == ("sh.600000", "sz.000001")


def test_normalize_baostock_tickers_rejects_invalid_codes():
    with pytest.raises(ValueError, match="baostock format"):
        _normalize_baostock_tickers(("600000.SS", "sh.bad"))


def test_normalize_akshare_tickers_accepts_common_a_share_formats():
    tickers = _normalize_akshare_tickers(("600519", "sh.600000", "000001.SZ", "SZ300750"))

    assert tickers == ("600519", "600000", "000001", "300750")


def test_normalize_akshare_tickers_rejects_invalid_codes():
    with pytest.raises(ValueError, match="six-digit A-share codes"):
        _normalize_akshare_tickers(("AAPL", "sh.bad"))


def test_codes_from_frame_selects_six_digit_code_column():
    frame = pd.DataFrame(
        {
            "date": ["2026-07-28", "2026-07-28"],
            "constituent_code": ["000001", "600519"],
            "name": ["Ping An Bank", "Kweichow Moutai"],
        }
    )

    assert _codes_from_frame(frame) == ("000001", "600519")


def test_select_tickers_resolves_csi_300_for_akshare():
    with mock.patch(
        "scripts.public_data_validation._load_akshare_csi300_tickers",
        return_value=("000001", "600519", "300750"),
    ):
        tickers = _select_tickers("csi-300", "", 2, source="akshare")

    assert tickers == ("000001", "600519")


def test_select_tickers_supports_cross_market_smoke_preset():
    tickers = _select_tickers("cross-market-5", "", 10, source="yfinance")

    assert tickers == ("SPY", "QQQ", "TLT", "GLD", "AGG")


def test_select_tickers_rejects_csi_300_for_non_akshare_source():
    with pytest.raises(click.BadParameter, match="requires --source akshare"):
        _select_tickers("csi-300", "", 300, source="yfinance")


def test_load_validation_panel_dispatches_to_akshare_with_normalized_tickers(tmp_path: Path):
    cfg = ValidationConfig(
        source="akshare",
        preset="csi-300",
        tickers=("SH.600519", "000001.SZ"),
        start="2021-01-01",
        end="2022-01-01",
        max_tickers=2,
        device="cpu",
        costs_bps=5.0,
        slippage_bps=2.0,
        cost_grid_bps=(),
        bootstrap_samples=0,
        bootstrap_block_size=10,
        train_window=40,
        test_window=20,
        step=20,
        top_quantile=0.25,
        seed=7,
        epochs=1,
        batch_size=128,
        hidden=16,
        models=("equal_weight",),
        output_dir=tmp_path,
        synthetic_dates=90,
        synthetic_stocks=12,
    )

    with mock.patch("scripts.public_data_validation.make_panel", return_value=mock.sentinel.panel) as loader:
        panel = load_validation_panel(cfg)

    assert panel is mock.sentinel.panel
    loader.assert_called_once_with(
        source="akshare",
        tickers=("600519", "000001"),
        start="2021-01-01",
        end="2022-01-01",
        device="cpu",
    )


def test_public_data_validation_cost_table_escapes_pipes():
    table = _cost_sensitivity_table(
        [
            {
                "cost_scenario": "0 | 7 bps",
                "strategy": "equal | weight",
                "effective_costs_bps": 7.0,
                "ann_return": 0.1,
                "sharpe": 0.2,
                "max_dd": 0.3,
                "turnover": 0.4,
                "cost_drag": 0.5,
                "final_equity": 1.6,
            }
        ]
    )

    assert "0 \\| 7 bps" in table
    assert "equal \\| weight" in table


def test_limit_masked_panel_masks_a_share_limit_days_only():
    """Validation factors are computed mask-first on A-share boards; non-A-share
    tickers keep their mask because they have no daily price limit."""
    from dataclasses import replace

    import torch

    from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel

    p = make_synthetic_panel(SyntheticConfig(n_stocks=3, n_dates=30, halt_prob=0.0, seed=5))
    p = replace(
        p,
        stocks=pd.Series(["600000", "300750", "AAPL"]).to_numpy(),
        dates=pd.bdate_range("2023-01-02", periods=30).to_numpy(),
        limit_up=None, limit_down=None, last_close=None,
    )
    p.close[10:, :] *= 1.15                   # +15 % on day 10 for all three
    masked = _limit_masked_panel(p)
    assert not bool(masked.mask[10, 0])       # main board: 10 % limit
    assert bool(masked.mask[10, 1])           # ChiNext: 20 % limit
    assert bool(masked.mask[10, 2])           # US ticker: no limit
    assert torch.equal(masked.close, p.close)


def test_run_validation_computes_factors_on_limit_masked_panel(tmp_path: Path):
    import scripts.public_data_validation as pdv

    seen = {}
    real = pdv._alpha101_features

    def spy(panel):
        seen["mask"] = panel.mask.clone()
        return real(panel)

    cfg = ValidationConfig(
        source="synthetic", preset="us-large-100", tickers=(), start="2021-01-01",
        end="2022-01-01", max_tickers=8, device="cpu", costs_bps=5.0, slippage_bps=0.0,
        cost_grid_bps=(), bootstrap_samples=0, bootstrap_block_size=10, train_window=30,
        test_window=10, step=10, top_quantile=0.25, seed=3, epochs=1, batch_size=64,
        hidden=8, models=("alpha101_mean",), output_dir=tmp_path, synthetic_dates=60,
        synthetic_stocks=8,
    )
    with mock.patch.object(pdv, "_alpha101_features", side_effect=spy):
        run_validation(cfg)

    panel = load_validation_panel(cfg)
    assert (seen["mask"] == _limit_masked_panel(panel).mask).all()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["limit_mask"].startswith("mask-first")


def test_signal_weights_do_not_use_tomorrows_availability(tmp_path):
    """A halt discovered tomorrow must not change today's equal-weight book."""
    from dataclasses import replace
    import scripts.public_data_validation as pdv
    from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel

    panel = make_synthetic_panel(SyntheticConfig(n_dates=60, n_stocks=8, halt_prob=0, seed=3))
    mask = panel.mask.clone()
    mask[31, 0] = False
    halted = replace(panel, mask=mask)
    cfg = ValidationConfig(
        source="synthetic", preset="us-large-100", tickers=(), start="2021-01-01",
        end="2022-01-01", max_tickers=8, device="cpu", costs_bps=5, slippage_bps=0,
        cost_grid_bps=(), bootstrap_samples=0, bootstrap_block_size=10, train_window=30,
        test_window=10, step=10, top_quantile=0.25, seed=3, epochs=1, batch_size=64,
        hidden=8, models=("equal_weight",), output_dir=tmp_path, synthetic_dates=60,
        synthetic_stocks=8,
    )
    books = []
    real = pdv._equal_weight

    def capture(valid):
        weights = real(valid)
        books.append(weights.copy())
        return weights

    for data in (panel, halted):
        with mock.patch.object(pdv, "load_validation_panel", return_value=data), \
             mock.patch.object(pdv, "_equal_weight", side_effect=capture):
            run_validation(cfg)
    import numpy as np
    assert np.allclose(books[0][:31], books[1][:31])
    assert books[1][30, 0] > 0
    assert books[1][31, 0] == 0
    assert bool(panel.mask[31, 0])
    assert not bool(halted.mask[31, 0])


def test_model_predicts_before_halt_without_training_on_missing_label(tmp_path):
    import numpy as np
    import scripts.public_data_validation as pdv

    cfg = ValidationConfig(
        source="synthetic", preset="us-large-100", tickers=(), start="2021-01-01",
        end="2022-01-01", max_tickers=8, device="cpu", costs_bps=5, slippage_bps=0,
        cost_grid_bps=(), bootstrap_samples=0, bootstrap_block_size=10, train_window=30,
        test_window=10, step=10, top_quantile=0.25, seed=3, epochs=1, batch_size=64,
        hidden=8, models=("mlp_alpha101",), output_dir=tmp_path, synthetic_dates=60,
        synthetic_stocks=8,
    )
    features = np.random.default_rng(3).normal(size=(60, 8, 3)).astype(np.float32)
    target = features[:, :, 0] * 0.01
    signal_valid = np.ones((60, 8), dtype=bool)
    signal_valid[31, 0] = False
    label_valid = signal_valid.copy()
    label_valid[:-1] &= signal_valid[1:]
    label_valid[-1] = False
    target[~label_valid] = np.nan
    preds = pdv._fit_predict_model(
        "mlp_alpha101", features, target, label_valid, cfg,
        prediction_valid=signal_valid,
    )
    assert np.isfinite(preds[30, 0])
    assert np.isnan(preds[31, 0])
    assert np.isfinite(preds[40, 0])  # no NaN label poisoned later training
