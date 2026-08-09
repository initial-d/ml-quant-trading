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
    config.write_text("seed: 42\n")
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
            calls.append((stage, config_path, Path(config_path).read_text()))

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

    assert packaged.read_text() == repository.read_text()


def test_backtest_summary_is_shareable_and_strict_json(tmp_path):
    markdown_path, json_path = _write_backtest_summary(
        tmp_path,
        {"sharpe": 1.25, "sortino": float("inf"), "n_periods": 500},
        config_path="configs/small.yaml",
        costs_bps=5.0,
    )

    payload = json.loads(json_path.read_text(), parse_constant=lambda value: 1 / 0)
    assert payload["metrics"]["sharpe"] == 1.25
    assert payload["metrics"]["sortino"] is None
    assert payload["costs_bps"] == 5.0
    assert payload["repository"] == "https://github.com/initial-d/ml-quant-trading"
    assert payload["reproduction_report"].endswith("reproduction_report.yml")

    markdown = markdown_path.read_text()
    assert "| `sharpe` | 1.250000 |" in markdown
    assert "| `sortino` | n/a |" in markdown
    assert "not investment advice" in markdown
    assert "https://github.com/initial-d/ml-quant-trading" in markdown
    assert "open the report form" in markdown
