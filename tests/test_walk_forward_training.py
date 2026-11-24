"""Sequential forecasts, independent gradient oracle and leakage boundaries."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from mlquant.cli.main import cli
from mlquant.models.losses import ICLoss, RankICLoss
from mlquant.models.nets import MLPRegressor
from mlquant.training import TrainConfig, Trainer, WalkForwardConfig, train_walk_forward


def inputs(total=30, assets=5, features=2):
    generator = torch.Generator().manual_seed(51)
    factors = torch.randn(total, assets, features, generator=generator)
    labels = .03 * factors[..., 0] + .01 * torch.randn(total, assets, generator=generator)
    mask = torch.ones((total, assets), dtype=torch.bool)
    return factors, mask, labels, mask.clone()


def run(tmp_path, data=None, cfg=None, **kwargs):
    return train_walk_forward(
        *(inputs() if data is None else data),
        cfg=cfg or WalkForwardConfig(train_dates=12, test_dates=7),
        train_cfg=TrainConfig(epochs=2, batch_size=64, val_split=.25, save_dir=str(tmp_path)),
        model_factory=lambda: MLPRegressor(2, hidden=4), loss_factory=torch.nn.MSELoss, **kwargs)


@pytest.mark.parametrize("window", ["rolling", "expanding"])
@pytest.mark.parametrize("horizon", [1, 3])
def test_original_row_boundaries_partial_tail_and_checkpoint_provenance(tmp_path, window, horizon):
    data = inputs()
    data[1][3:5] = False
    data[1][25] = False
    result = run(tmp_path, data, WalkForwardConfig(train_dates=12, test_dates=7,
                                                  label_horizon=horizon, window=window))
    assert result.fold_id[:12 + horizon].tolist() == [-1] * (12 + horizon)
    assert not result.prediction_mask[:12 + horizon].any()
    assert not result.predictions[:12 + horizon].any()
    assert not result.prediction_mask[25].any()
    for fold in result.report["folds"]:
        start, stop = fold["test_start"], fold["test_end_exclusive"]
        split = fold["split"]["partitions"]
        assert split["train"]["date_range"][1] + horizon < split["validation"]["date_range"][0]
        assert fold["last_validation_label_row"] < start
        assert fold["history_end_exclusive"] == start - horizon
        assert fold["train_start"] == (start - horizon - 12 if window == "rolling" else 0)
        assert result.fold_id[start:stop].tolist() == [fold["fold"]] * (stop - start)
        assert hashlib.sha256(Path(fold["checkpoint"]).read_bytes()).hexdigest() == fold["checkpoint_sha256"]
        assert fold["prediction_cells"] == int(result.prediction_mask[start:stop].sum())
    assert result.report["folds"][-1]["test_end_exclusive"] == 30
    assert result.report["folds"][-1]["partial_test"]
    assert result.report["summary"]["prediction_cells"] == int(result.prediction_mask.sum())


def test_first_fold_matches_independent_single_step_adam_gradient(tmp_path):
    # History signals 0..7, validation 6..7, purged 5, training 0..4.
    x = torch.arange(1, 4).float()[None, :, None].expand(16, -1, -1).clone()
    y = .02 * x[..., 0]
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0] = False
    label_mask = torch.ones_like(mask)
    label_mask[1, 1] = False
    y[6:] = -100  # Must not change the training gradient or initialization.

    def model():
        layer = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(layer.weight)
        return torch.nn.Sequential(layer, torch.nn.Flatten(0))

    result = train_walk_forward(x, mask, y, label_mask, model_factory=model,
        loss_factory=torch.nn.MSELoss, cfg=WalkForwardConfig(train_dates=8, test_dates=8),
        train_cfg=TrainConfig(epochs=1, batch_size=1000, lr=.01, weight_decay=0.,
                              val_split=.25, save_dir=str(tmp_path)))
    pairs = mask[:5] & label_mask[:5]
    gradient = (-2 * x[:5, :, 0][pairs].double() * y[:5][pairs].double()).mean().item()
    expected_weight = -.01 * gradient / (abs(gradient) + 1e-8)
    torch.testing.assert_close(result.predictions[9:], x[9:, :, 0] * expected_weight, rtol=2e-6, atol=1e-9)
    assert result.report["folds"][0]["split"]["partitions"]["train"]["samples"] == 13


def test_future_labels_masks_and_features_do_not_change_earlier_forecasts(tmp_path):
    original = inputs()
    baseline = run(tmp_path / "original", original)
    changed = [value.clone() for value in original]
    changed[0][20:] *= -100
    changed[1][20:, 0] = False
    changed[2][13:] = 1000
    changed[3][13:, 1] = False
    altered = run(tmp_path / "changed", changed)
    # All first-fold predictions (13..19) precede any feature mutation; no
    # labels at or after test start were used by that model.
    assert torch.equal(baseline.predictions[:20], altered.predictions[:20])
    assert torch.equal(baseline.prediction_mask[:20], altered.prediction_mask[:20])
    assert baseline.report["folds"][0]["checkpoint_sha256"] == altered.report["folds"][0]["checkpoint_sha256"]


def test_outer_purge_labels_and_padding_factors_are_not_consumed(tmp_path):
    data = inputs()
    baseline = run(tmp_path / "before", data, WalkForwardConfig(train_dates=12, test_dates=50, purge_dates=3))
    changed = [value.clone() for value in data]
    changed[0][12:15] = float("nan")
    changed[1][12:15] = False
    changed[2][12:] = float("nan")
    changed[3][12:] = False
    altered = run(tmp_path / "after", changed, WalkForwardConfig(train_dates=12, test_dates=50, purge_dates=3))
    assert torch.equal(baseline.predictions, altered.predictions)
    assert torch.equal(baseline.prediction_mask, altered.prediction_mask)


def test_future_label_availability_never_gates_current_predictions(tmp_path):
    data = inputs()
    data[3][13:] = False
    data[2][13:] = float("nan")
    data[0][14, 1] = float("inf")
    result = run(tmp_path, data, WalkForwardConfig(train_dates=12, test_dates=50))
    assert result.prediction_mask[13].all()
    assert not result.prediction_mask[14, 1]
    assert result.predictions[14, 1] == 0
    assert torch.isfinite(result.predictions).all()


def test_best_validation_checkpoint_is_loaded_instead_of_final_epoch(tmp_path, monkeypatch):
    counters = {}

    def epoch(trainer, loader, *, train):
        number = counters.get(id(trainer), 0)
        if train:
            counters[id(trainer)] = number + 1
            with torch.no_grad():
                trainer.model[0].weight.fill_(number + 1)
            return 1.
        return [3., 1., 2.][number - 1]

    monkeypatch.setattr(Trainer, "_run_epoch", epoch)
    data = inputs(total=20, features=1)
    result = train_walk_forward(*data, cfg=WalkForwardConfig(train_dates=12, test_dates=50),
        train_cfg=TrainConfig(epochs=3, val_split=.25, save_dir=str(tmp_path)),
        model_factory=lambda: torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Flatten(0)),
        loss_factory=torch.nn.MSELoss)
    torch.testing.assert_close(result.predictions[13:], 2 * data[0][13:, :, 0])
    assert result.report["folds"][0]["best_epoch"] == 1


def test_fresh_models_determinism_and_rng_restoration(tmp_path):
    torch.manual_seed(311)
    before = torch.random.get_rng_state().clone()
    baseline = run(tmp_path / "one")
    assert torch.equal(torch.random.get_rng_state(), before)
    changed = run(tmp_path / "two")
    assert torch.equal(baseline.predictions, changed.predictions)
    assert [fold["seed"] for fold in baseline.report["folds"]] == [42, 43, 44]
    assert [fold["checkpoint_sha256"] for fold in baseline.report["folds"]] == [
        fold["checkpoint_sha256"] for fold in changed.report["folds"]]


@pytest.mark.parametrize("criterion", [ICLoss, RankICLoss])
def test_cross_sectional_losses_keep_complete_dates(tmp_path, criterion):
    result = train_walk_forward(*inputs(), cfg=WalkForwardConfig(train_dates=12, test_dates=50),
        train_cfg=TrainConfig(epochs=1, val_split=.25, save_dir=str(tmp_path)),
        model_factory=lambda: MLPRegressor(2, hidden=4), loss_factory=criterion)
    assert result.report["folds"][0]["split"]["batch_mode"] == "date"
    assert result.prediction_mask[13:].all()


@pytest.mark.parametrize("kwargs,match", [
    ({"train_dates": True}, "positive integer"), ({"test_dates": 0}, "positive integer"),
    ({"label_horizon": 0}, "positive integer"), ({"purge_dates": 0}, "label_horizon"),
    ({"purge_dates": 2, "label_horizon": 3}, "label_horizon"),
    ({"window": "random"}, "window"), ({"seed": -1}, "seed"),
])
def test_invalid_schedule_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WalkForwardConfig(**kwargs)


def test_no_test_rows_random_validation_and_insufficient_history_fail(tmp_path):
    with pytest.raises(ValueError, match="no test rows"):
        run(tmp_path, cfg=WalkForwardConfig(train_dates=30))
    with pytest.raises(ValueError, match="fold 0.*no training dates"):
        run(tmp_path, cfg=WalkForwardConfig(train_dates=3, label_horizon=3))
    with pytest.raises(ValueError, match="val_split_mode"):
        train_walk_forward(*inputs(), model_factory=lambda: MLPRegressor(2), loss_factory=torch.nn.MSELoss,
                           train_cfg=TrainConfig(val_split_mode="random"), cfg=WalkForwardConfig(train_dates=12))


def test_nonfinite_losses_fail_without_reusing_existing_best(tmp_path, monkeypatch):
    old = tmp_path / "best.pt"
    old.write_bytes(b"old checkpoint must not be loaded")
    monkeypatch.setattr(Trainer, "_run_epoch", lambda *args, **kwargs: float("nan"))
    with pytest.raises(ValueError, match="nonfinite training/validation"):
        run(tmp_path)
    assert old.read_bytes() == b"old checkpoint must not be loaded"


def test_cli_rolling_forecasts_feed_portfolio_backtest_and_rebalancing(tmp_path):
    config = tmp_path / "tiny.yaml"
    art = tmp_path / "artifacts"
    config.write_text(f"""seed: 51
artifacts_dir: {art}
train_frac: 0.6
hidden: 4
cov_lookback: 8
costs_bps: 5
synthetic:
  n_dates: 40
  n_stocks: 12
train:
  epochs: 1
  batch_size: 128
  val_split: 0.25
portfolio:
  weight_cap: 0.2
  solver: SCS
""")
    runner = CliRunner()
    assert runner.invoke(cli, ["gen-data", "--config", str(config)]).exit_code == 0
    data = inputs(total=40, assets=12)
    torch.save({"factors": data[0], "mask": data[1], "names": ["synthetic_x", "synthetic_y"]}, art / "features.pt")
    result = runner.invoke(cli, ["train", "--config", str(config), "--walk-forward", "--test-dates", "7"])
    assert result.exit_code == 0, result.output
    blob = torch.load(art / "predictions.pt", weights_only=False)
    assert blob["oos_start"] == 24
    assert blob["protocol"] == "model-walk-forward-v1"
    report = json.loads((art / "walk-forward.json").read_text())
    assert len(report["folds"]) == 3
    assert report["summary"]["prediction_cells"] == 16 * 12
    assert report["input_sha256"]["features.pt"] == hashlib.sha256((art / "features.pt").read_bytes()).hexdigest()
    for command in (["portfolio"], ["backtest", "--accounting", "constrained"],
                    ["rebalance", "--currency", "CNY", "--data-category", "synthetic", "--data-source", "test"]):
        result = runner.invoke(cli, [*command, "--config", str(config)])
        assert result.exit_code == 0, result.output
    summary = json.loads((art / "summary.json").read_text())
    assert summary["evaluation"]["refit_folds"] == 3
    dates = torch.load(art / "panel.pt", weights_only=False)["dates"]
    assert summary["evaluation"]["train_end"] == str(dates[22].astype("datetime64[D]"))
    assert summary["metrics"]["n_periods"] == 16
    assert "earlier test observations" in (art / "summary.md").read_text()
    # New forecasts must not be silently evaluated against old portfolio targets.
    blob["pred"][25] += 1
    torch.save(blob, art / "predictions.pt")
    failed = runner.invoke(cli, ["backtest", "--config", str(config)])
    assert failed.exit_code != 0 and "rerun mlquant portfolio" in failed.output


def test_cli_invalid_or_failed_refit_preserves_published_predictions(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(f"artifacts_dir: {tmp_path}\n")
    runner = CliRunner()
    invalid = runner.invoke(cli, ["train", "--config", str(config), "--test-dates", "7"])
    assert invalid.exit_code == 2 and "require --walk-forward" in invalid.output
    assert not (tmp_path / "walk-forward.json").exists()
    art = tmp_path / "failed"
    config.write_text(f"artifacts_dir: {art}\ntrain_frac: 0.6\ntrain:\n  val_split_mode: random\n")
    assert runner.invoke(cli, ["gen-data", "--config", str(config)]).exit_code == 0
    data = inputs(total=500, assets=200)
    torch.save({"factors": data[0], "mask": data[1]}, art / "features.pt")
    published = art / "predictions.pt"
    published.write_bytes(b"previous successful forecast artifact")
    failed = runner.invoke(cli, ["train", "--config", str(config), "--walk-forward"])
    assert failed.exit_code != 0 and "val_split_mode" in failed.output
    assert published.read_bytes() == b"previous successful forecast artifact"
    assert not (art / "walk-forward.json").exists()
