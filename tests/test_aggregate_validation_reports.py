import json
from pathlib import Path

from scripts.aggregate_validation_reports import load_reports, markdown_table, write_csv, write_markdown


def _write_report(path: Path, *, preset: str, strategy: str, sharpe: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "source": "synthetic",
                    "preset": preset,
                    "python": "3.11.0",
                    "pytorch": "2.8.0",
                    "platform": "Linux | test",
                    "panel": {
                        "date_start": "2021-01-01",
                        "date_end": "2021-12-31",
                        "n_dates": 252,
                        "n_stocks": 50,
                        "tradable_ratio": 0.99,
                    },
                },
                "results": [
                    {
                        "strategy": strategy,
                        "ann_return": 0.1,
                        "sharpe": sharpe,
                        "max_dd": 0.2,
                        "turnover": 0.3,
                        "cost_drag": 0.04,
                        "final_equity": 1.1,
                    }
                ],
            }
        )
    )


def test_aggregate_validation_reports(tmp_path: Path):
    _write_report(tmp_path / "run-a" / "summary.json", preset="a", strategy="slow", sharpe=0.2)
    _write_report(tmp_path / "run-b" / "summary.json", preset="a", strategy="fast", sharpe=1.5)

    rows = load_reports([tmp_path])

    assert [row["strategy"] for row in rows] == ["fast", "slow"]
    table = markdown_table(rows)
    assert "fast" in table
    assert "1.5000" in table
    assert "Linux \\| test" in table

    write_markdown(tmp_path / "leaderboard.md", rows)
    write_csv(tmp_path / "leaderboard.csv", rows)
    assert (tmp_path / "leaderboard.md").exists()
    assert (tmp_path / "leaderboard.csv").exists()
