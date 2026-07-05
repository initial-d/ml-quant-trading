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
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.csv").exists()
