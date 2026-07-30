from click.testing import CliRunner

from mlquant.cli.main import (
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


def test_demo_is_visible_in_cli_help():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "demo" in result.output
