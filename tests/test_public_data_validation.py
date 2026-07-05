import json
from pathlib import Path

from scripts.public_data_validation import ValidationConfig, run_validation


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
    assert "tradable_ratio" in metadata["panel"]

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["cost_sensitivity"]) == 9
    assert summary["cost_sensitivity"][0]["effective_costs_bps"] == 0.0
    assert summary["results"][0]["bootstrap_samples"] == 25

    submission = (tmp_path / "submission.md").read_text()
    assert "Public-data validation report" in submission
    assert "Data Coverage" in submission
