import json
from pathlib import Path

from click.testing import CliRunner

from mlquant.cli.main import (
    _write_backtest_summary,
    cli,
    cmd_backtest,
    cmd_features,
    cmd_gen_data,
    cmd_portfolio,
    cmd_train,
)


def test_demo_runs_pipeline_stages_in_order(monkeypatch, tmp_path):
    config = tmp_path / "small.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    calls = []

    for command in (cmd_gen_data, cmd_features, cmd_train, cmd_portfolio, cmd_backtest):
        name = command.name

        def callback(config_path, stage=name):
            calls.append((stage, config_path))

        monkeypatch.setattr(command, "callback", callback)

    result = CliRunner().invoke(cli, ["demo", "--config", str(config)])

    assert result.exit_code == 0
    assert [stage for stage, _ in calls] == [
        "gen-data",
        "features",
        "train",
        "portfolio",
        "backtest",
    ]
    assert all(path == str(config) for _, path in calls)
    assert "Demo complete" in result.output
    assert "https://github.com/initial-d/ml-quant-trading" in result.output
    assert "reproduction_report.yml" in result.output
    assert "consider starring" in result.output


def test_demo_is_visible_in_cli_help():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "demo" in result.output


def test_demo_uses_packaged_config_outside_repository(monkeypatch, tmp_path):
    calls = []

    for command in (cmd_gen_data, cmd_features, cmd_train, cmd_portfolio, cmd_backtest):
        name = command.name

        def callback(config_path, stage=name):
            calls.append((stage, config_path, Path(config_path).read_text(encoding="utf-8")))

        monkeypatch.setattr(command, "callback", callback)

    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        result = CliRunner().invoke(cli, ["demo"])

    assert result.exit_code == 0
    assert len(calls) == 5
    assert all(Path(path).name == "small.yaml" for _, path, _ in calls)
    assert all("n_stocks:     200" in content for _, _, content in calls)
    assert "Using packaged default config" in result.output


def test_packaged_demo_config_matches_repository_config():
    packaged = Path(__file__).parents[1] / "src" / "mlquant" / "configs" / "small.yaml"
    repository = Path(__file__).parents[1] / "configs" / "small.yaml"

    assert packaged.read_text(encoding="utf-8") == repository.read_text(encoding="utf-8")


def test_backtest_summary_is_shareable_and_strict_json(tmp_path):
    markdown_path, json_path = _write_backtest_summary(
        tmp_path,
        {"sharpe": 1.25, "sortino": float("inf"), "n_periods": 500},
        config_path="configs/small.yaml",
        costs_bps=5.0,
    )

    payload = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: 1 / 0,
    )
    assert payload["metrics"]["sharpe"] == 1.25
    assert payload["metrics"]["sortino"] is None
    assert payload["costs_bps"] == 5.0
    assert payload["repository"] == "https://github.com/initial-d/ml-quant-trading"
    assert payload["reproduction_report"].endswith("reproduction_report.yml")

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| `sharpe` | 1.250000 |" in markdown
    assert "| `sortino` | n/a |" in markdown
    assert "not investment advice" in markdown
    assert "https://github.com/initial-d/ml-quant-trading" in markdown
    assert "open the report form" in markdown


TINY_CONFIG = """\
seed: 7
artifacts_dir: artifacts/tiny
limit_pct: 0.098
costs_bps: 5.0
cov_lookback: 20
train_frac: 0.6
hidden: 8
synthetic:
  n_stocks: 20
  n_dates: 120
  seed: 7
train:
  epochs: 1
  batch_size: 256
  save_dir: artifacts/tiny/checkpoints
portfolio:
  risk_aversion: 5.0
  weight_cap: 0.2
  long_only: true
  solver: SCS
"""


def test_pipeline_scores_only_the_out_of_sample_window(tmp_path):
    import torch

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("tiny.yaml").write_text(TINY_CONFIG, encoding="utf-8")
        result = runner.invoke(cli, ["demo", "--config", "tiny.yaml"])
        assert result.exit_code == 0, result.output

        art = Path("artifacts/tiny")
        pred = torch.load(art / "predictions.pt", weights_only=False)
        oos = pred["oos_start"]
        assert oos == 72
        assert not pred["pred"][:oos].any()

        weights = torch.load(art / "weights.pt", weights_only=False)["weights"]
        assert not weights[:oos].any()
        assert weights[oos:].abs().sum() > 0

        summary = json.loads((art / "summary.json").read_text(encoding="utf-8"))
        evaluation = summary["evaluation"]
        assert evaluation["n_train_dates"] == oos
        assert evaluation["n_oos_dates"] == 120 - oos
        assert evaluation["train_end"] < evaluation["oos_start"]
        assert summary["metrics"]["n_periods"] == 120 - oos
        assert "out-of-sample window" in (art / "summary.md").read_text(encoding="utf-8")


def test_invalid_train_frac_is_rejected(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("bad.yaml").write_text(TINY_CONFIG.replace("train_frac: 0.6", "train_frac: 1.5"),
                                    encoding="utf-8")
        result = runner.invoke(cli, ["demo", "--config", "bad.yaml"])
        assert result.exit_code != 0
        assert "train_frac" in result.output
