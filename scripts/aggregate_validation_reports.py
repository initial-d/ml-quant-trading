"""Aggregate public-data validation `summary.json` files.

The public validation script writes machine-readable reports. This helper turns
one or many reports into a compact leaderboard-style Markdown/CSV summary for
maintainers, issues, or docs.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import click


LEADERBOARD_COLUMNS = (
    "source",
    "preset",
    "date_range",
    "panel",
    "strategy",
    "ann_return",
    "sharpe",
    "sharpe_ci_low",
    "sharpe_ci_high",
    "max_dd",
    "turnover",
    "cost_drag",
    "effective_costs_bps",
    "final_equity",
    "tradable_ratio",
    "python",
    "pytorch",
    "platform",
    "report",
)


def _discover_summary_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.name == "summary.json":
                files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("summary.json")))
    return sorted(dict.fromkeys(files))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _fmt(value: Any) -> str:
    number = _as_float(value)
    if number is not None:
        return f"{number:.4f}"
    return "" if value is None else str(value)


def _md_cell(value: Any) -> str:
    """Format a value for a Markdown table cell."""
    return _fmt(value).replace("|", "\\|")


def _flatten_report(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text())
    metadata = report.get("metadata", {})
    panel = metadata.get("panel", {})
    rows = []
    for result in report.get("results", []):
        row = {
            "source": metadata.get("source", ""),
            "preset": metadata.get("preset", ""),
            "date_range": f"{panel.get('date_start', '')} to {panel.get('date_end', '')}",
            "panel": f"{panel.get('n_dates', '')} x {panel.get('n_stocks', '')}",
            "strategy": result.get("strategy", ""),
            "ann_return": result.get("ann_return"),
            "sharpe": result.get("sharpe"),
            "sharpe_ci_low": result.get("sharpe_ci_low"),
            "sharpe_ci_high": result.get("sharpe_ci_high"),
            "max_dd": result.get("max_dd"),
            "turnover": result.get("turnover"),
            "cost_drag": result.get("cost_drag"),
            "effective_costs_bps": result.get("effective_costs_bps", metadata.get("effective_costs_bps")),
            "final_equity": result.get("final_equity"),
            "tradable_ratio": panel.get("tradable_ratio"),
            "python": metadata.get("python", ""),
            "pytorch": metadata.get("pytorch", ""),
            "platform": metadata.get("platform", ""),
            "report": str(path),
        }
        rows.append(row)
    return rows


def load_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and flatten all discovered validation reports."""
    files = _discover_summary_files(paths)
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(_flatten_report(file))
    rows.sort(
        key=lambda row: (
            str(row.get("source", "")),
            str(row.get("preset", "")),
            -(_as_float(row.get("sharpe")) or -999.0),
            str(row.get("strategy", "")),
        )
    )
    return rows


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render flattened rows as a Markdown table."""
    lines = [
        "| " + " | ".join(LEADERBOARD_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(LEADERBOARD_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row.get(column)) for column in LEADERBOARD_COLUMNS) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in LEADERBOARD_COLUMNS} for row in rows)


def write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Public-Data Validation Leaderboard\n\n")
        f.write("This aggregate is a comparison aid, not a trading-performance claim.\n\n")
        if rows:
            f.write(markdown_table(rows))
            f.write("\n")
        else:
            f.write("No validation reports found.\n")


@click.command()
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--output-md", type=click.Path(path_type=Path), default=Path("artifacts/public_data_validation/leaderboard.md"), show_default=True)
@click.option("--output-csv", type=click.Path(path_type=Path), default=Path("artifacts/public_data_validation/leaderboard.csv"), show_default=True)
def main(paths: tuple[Path, ...], output_md: Path, output_csv: Path) -> None:
    """Aggregate one or more validation report directories/files."""
    selected_paths = paths or (Path("artifacts/public_data_validation"),)
    rows = load_reports(selected_paths)
    write_markdown(output_md, rows)
    write_csv(output_csv, rows)
    click.echo(f"Found {len(rows)} strategy rows.")
    click.echo(f"Wrote {output_md} and {output_csv}.")


if __name__ == "__main__":  # pragma: no cover
    main()
