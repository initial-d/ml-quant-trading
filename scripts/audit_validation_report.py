"""Audit public-data validation reports for reproducibility quality.

The audit is deliberately about evidence hygiene, not profitability. It checks
whether a generated `summary.json` has enough metadata, data coverage, and sane
metrics to be useful as a community validation report.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    check: str
    message: str


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


def _finding(severity: str, check: str, message: str) -> AuditFinding:
    return AuditFinding(severity=severity, check=check, message=message)


def audit_report(
    report: dict[str, Any],
    *,
    min_tradable_ratio: float = 0.80,
    require_equal_weight: bool = True,
) -> list[AuditFinding]:
    """Return audit findings for one validation report."""
    findings: list[AuditFinding] = []
    metadata = report.get("metadata")
    results = report.get("results")

    if not isinstance(metadata, dict):
        return [_finding("error", "metadata", "missing metadata object")]
    if not isinstance(results, list) or not results:
        findings.append(_finding("error", "results", "missing non-empty results list"))
        results = []

    command = str(metadata.get("command", "")).strip()
    if not command:
        findings.append(_finding("warning", "command", "missing exact command string"))

    for field in ("source", "preset", "python", "platform", "pytorch"):
        if not metadata.get(field):
            findings.append(_finding("warning", field, f"missing metadata field: {field}"))

    for field in ("costs_bps", "slippage_bps", "effective_costs_bps"):
        value = _as_float(metadata.get(field))
        if value is None:
            findings.append(_finding("warning", field, f"missing or non-numeric {field}"))
        elif value < 0:
            findings.append(_finding("error", field, f"{field} must be non-negative"))

    panel = metadata.get("panel")
    if not isinstance(panel, dict):
        findings.append(_finding("error", "panel", "missing panel coverage metadata"))
        panel = {}
    n_dates = _as_float(panel.get("n_dates"))
    n_stocks = _as_float(panel.get("n_stocks"))
    tradable_ratio = _as_float(panel.get("tradable_ratio"))
    if n_dates is None or n_dates <= 1:
        findings.append(_finding("error", "panel.n_dates", "panel must contain more than one date"))
    if n_stocks is None or n_stocks <= 0:
        findings.append(_finding("error", "panel.n_stocks", "panel must contain at least one stock"))
    if tradable_ratio is None:
        findings.append(_finding("warning", "panel.tradable_ratio", "missing tradable ratio"))
    elif tradable_ratio < min_tradable_ratio:
        findings.append(
            _finding(
                "warning",
                "panel.tradable_ratio",
                f"tradable ratio {tradable_ratio:.4f} is below {min_tradable_ratio:.4f}",
            )
        )

    no_data = panel.get("stocks_with_no_data", [])
    if isinstance(no_data, list) and no_data:
        findings.append(_finding("warning", "panel.stocks_with_no_data", f"tickers with no data: {no_data}"))

    strategy_names = {str(row.get("strategy", "")) for row in results if isinstance(row, dict)}
    if require_equal_weight and "equal_weight" not in strategy_names:
        findings.append(_finding("warning", "baseline", "missing equal_weight baseline"))

    for index, row in enumerate(results):
        if not isinstance(row, dict):
            findings.append(_finding("error", f"results[{index}]", "result row is not an object"))
            continue
        strategy = row.get("strategy") or f"results[{index}]"
        for field in ("ann_return", "ann_vol", "sharpe", "max_dd", "turnover", "cost_drag", "final_equity"):
            value = _as_float(row.get(field))
            if value is None:
                findings.append(_finding("warning", f"{strategy}.{field}", f"missing or non-finite {field}"))
        max_dd = _as_float(row.get("max_dd"))
        turnover = _as_float(row.get("turnover"))
        final_equity = _as_float(row.get("final_equity"))
        if max_dd is not None and not 0 <= max_dd <= 1.5:
            findings.append(_finding("warning", f"{strategy}.max_dd", f"unusual max drawdown: {max_dd:.4f}"))
        if turnover is not None and turnover > 2.0:
            findings.append(_finding("warning", f"{strategy}.turnover", f"unusual average turnover: {turnover:.4f}"))
        if final_equity is not None and final_equity <= 0:
            findings.append(_finding("error", f"{strategy}.final_equity", "final equity must be positive"))

    if not findings:
        findings.append(_finding("pass", "report", "report passed all audit checks"))
    return findings


def findings_to_dicts(findings: Sequence[AuditFinding]) -> list[dict[str, str]]:
    return [{"severity": item.severity, "check": item.check, "message": item.message} for item in findings]


def has_errors(findings: Sequence[AuditFinding]) -> bool:
    return any(item.severity == "error" for item in findings)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def markdown_report(findings: Sequence[AuditFinding], *, source: Path) -> str:
    lines = [
        "# Validation Report Audit",
        "",
        f"Source: `{source}`",
        "",
        "| Severity | Check | Message |",
        "|---|---|---|",
    ]
    for item in findings:
        lines.append(
            f"| {_markdown_cell(item.severity)} | `{_markdown_cell(item.check)}` | {_markdown_cell(item.message)} |"
        )
    return "\n".join(lines) + "\n"


@click.command()
@click.argument("summary_json", type=click.Path(path_type=Path))
@click.option("--output-md", type=click.Path(path_type=Path), default=None)
@click.option("--output-json", type=click.Path(path_type=Path), default=None)
@click.option("--min-tradable-ratio", default=0.80, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option("--no-require-equal-weight", is_flag=True, help="Do not warn when equal_weight is missing.")
def main(
    summary_json: Path,
    output_md: Path | None,
    output_json: Path | None,
    min_tradable_ratio: float,
    no_require_equal_weight: bool,
) -> None:
    """Audit one public-data validation summary JSON file."""
    report = json.loads(summary_json.read_text())
    findings = audit_report(
        report,
        min_tradable_ratio=min_tradable_ratio,
        require_equal_weight=not no_require_equal_weight,
    )
    text = markdown_report(findings, source=summary_json)
    click.echo(text, nl=False)

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(text)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(findings_to_dicts(findings), indent=2, sort_keys=True))

    if has_errors(findings):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
