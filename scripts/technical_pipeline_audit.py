"""Run a deterministic engineering audit of the public research pipeline.

The checks deliberately avoid market data and investment-performance claims.
They exercise invariants that should hold before a factor backtest is treated as
technically inspectable: catalogue shape, deterministic generation, mask
isolation, forward-label boundaries, lagged execution, and cost arithmetic.

Usage:
    python scripts/technical_pipeline_audit.py
    python scripts/technical_pipeline_audit.py --output-dir artifacts/technical_audit
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import torch

from mlquant.backtest.engine import run_backtest
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_legacy_set, cs_rank, ewma, ts_corr, ts_mean, ts_rank
from mlquant.features.target import target_01


EXPECTED_FACTOR_COUNT = 213
PROJECT_URL = "https://github.com/initial-d/ml-quant-trading"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def audit_factor_catalog(panel) -> dict[str, Any]:
    started = time.perf_counter()
    factors, factor_mask, names = compute_legacy_set(panel)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    expected_shape = (panel.n_dates, panel.n_stocks, EXPECTED_FACTOR_COUNT)
    finite = bool(torch.isfinite(factors).all())
    passed = tuple(factors.shape) == expected_shape and len(names) == EXPECTED_FACTOR_COUNT and finite
    return _check(
        "factor_catalog",
        passed,
        tensor_shape=list(factors.shape),
        factor_count=len(names),
        all_values_finite=finite,
        joint_valid_cells=int(factor_mask.sum()),
        elapsed_ms=round(elapsed_ms, 3),
    )


def audit_deterministic_generation(cfg: SyntheticConfig) -> dict[str, Any]:
    first = make_synthetic_panel(cfg)
    second = make_synthetic_panel(cfg)
    same = bool(
        torch.equal(first.close, second.close)
        and torch.equal(first.volume, second.volume)
        and torch.equal(first.mask, second.mask)
        and np.array_equal(first.dates, second.dates)
        and np.array_equal(first.stocks, second.stocks)
    )
    return _check(
        "deterministic_generation",
        same,
        seed=cfg.seed,
        close_sha_proxy=round(float(first.close.double().sum()), 6),
        tradable_cells=int(first.mask.sum()),
    )


def audit_mask_isolation(*, n_dates: int, n_stocks: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_dates, n_stocks, generator=generator)
    y = torch.randn(n_dates, n_stocks, generator=generator)
    mask = torch.rand(n_dates, n_stocks, generator=generator) > 0.10
    poisoned_x = torch.where(mask, x, torch.full_like(x, 1e9))
    poisoned_y = torch.where(mask, y, torch.full_like(y, -1e9))

    Operation = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    operations: dict[str, Operation] = {
        "cs_rank": lambda a, _b: cs_rank(a, mask),
        "ts_mean_20": lambda a, _b: ts_mean(a, mask, 20),
        "ts_rank_20": lambda a, _b: ts_rank(a, mask, 20),
        "ts_corr_20": lambda a, b: ts_corr(a, b, mask, 20),
        "ewma_0.05": lambda a, _b: ewma(a, mask, 0.05),
    }

    rows = []
    for name, operation in operations.items():
        clean_values, clean_mask = operation(x, y)
        poisoned_values, poisoned_mask = operation(poisoned_x, poisoned_y)
        comparable = clean_mask & poisoned_mask
        max_abs_drift = (
            float((clean_values - poisoned_values).abs()[comparable].max())
            if bool(comparable.any())
            else 0.0
        )
        rows.append(
            {
                "operation": name,
                "comparable_cells": int(comparable.sum()),
                "max_abs_drift": max_abs_drift,
                "passed": max_abs_drift <= 1e-6,
            }
        )

    return _check(
        "mask_isolation",
        all(row["passed"] for row in rows),
        poison_magnitude=1e9,
        operations=rows,
    )


def audit_forward_labels(panel) -> dict[str, Any]:
    labels, label_mask = target_01(panel)
    expected_mask = torch.zeros_like(panel.mask)
    expected_mask[:-1] = panel.mask[:-1] & panel.mask[1:]
    boundary_zero = bool(torch.equal(labels[-1], torch.zeros_like(labels[-1])))
    mask_exact = bool(torch.equal(label_mask, expected_mask))
    finite = bool(torch.isfinite(labels).all())
    return _check(
        "forward_label_boundary",
        boundary_zero and mask_exact and finite,
        horizon=1,
        last_row_masked=bool(~label_mask[-1].any()),
        last_row_zero=boundary_zero,
        endpoint_mask_exact=mask_exact,
        all_values_finite=finite,
    )


def audit_backtest_alignment() -> dict[str, Any]:
    weights = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    returns = np.asarray(
        [[0.0, 0.0], [0.10, -0.10], [0.20, -0.20], [0.30, -0.30]],
        dtype=np.float64,
    )
    expected_gross = np.asarray([0.0, 0.10, -0.20, 0.30], dtype=np.float64)
    result = run_backtest(weights, returns, costs_bps=0.0)
    passed = bool(np.allclose(result.gross_returns, expected_gross, atol=1e-12))
    return _check(
        "lagged_execution_alignment",
        passed,
        convention="weights[t-1] earn returns[t]",
        expected_gross=expected_gross.tolist(),
        observed_gross=result.gross_returns.tolist(),
    )


def audit_cost_linearity() -> dict[str, Any]:
    weights = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    returns = np.zeros_like(weights)
    one_bps = run_backtest(weights, returns, costs_bps=1.0).cost_drag_cumulative
    ten_bps = run_backtest(weights, returns, costs_bps=10.0).cost_drag_cumulative
    ratio = ten_bps / one_bps if one_bps else float("nan")
    passed = bool(np.isclose(ratio, 10.0, atol=1e-12))
    return _check(
        "transaction_cost_linearity",
        passed,
        one_bps_cost_drag=one_bps,
        ten_bps_cost_drag=ten_bps,
        observed_ratio=ratio,
        expected_ratio=10.0,
    )


def run_audit(*, n_dates: int, n_stocks: int, seed: int) -> dict[str, Any]:
    cfg = SyntheticConfig(n_dates=n_dates, n_stocks=n_stocks, seed=seed)
    panel = make_synthetic_panel(cfg)
    checks = [
        audit_factor_catalog(panel),
        audit_deterministic_generation(cfg),
        audit_mask_isolation(n_dates=n_dates, n_stocks=n_stocks, seed=seed + 1),
        audit_forward_labels(panel),
        audit_backtest_alignment(),
        audit_cost_linearity(),
    ]
    return {
        "project": "ml-quant-trading",
        "repository": PROJECT_URL,
        "audit": "technical_pipeline_invariants",
        "commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
        },
        "config": {"n_dates": n_dates, "n_stocks": n_stocks, "seed": seed},
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "scope": (
            "Engineering invariants on deterministic synthetic data; not evidence of market alpha, "
            "out-of-sample performance, or production readiness."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# Technical Pipeline Audit",
        "",
        f"**Overall status: {status}**",
        "",
        f"- Commit: `{report['commit']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- PyTorch: `{report['environment']['pytorch']}`",
        (
            "- Synthetic panel: "
            f"`{report['config']['n_dates']} dates × {report['config']['n_stocks']} stocks`"
        ),
        f"- Seed: `{report['config']['seed']}`",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        details = check["details"]
        if check["name"] == "factor_catalog":
            evidence = (
                f"shape `{details['tensor_shape']}`; {details['factor_count']} factors; "
                f"finite={details['all_values_finite']}; {details['elapsed_ms']:.3f} ms"
            )
        elif check["name"] == "deterministic_generation":
            evidence = f"seed `{details['seed']}` reproduced identical panel tensors"
        elif check["name"] == "mask_isolation":
            worst = max(row["max_abs_drift"] for row in details["operations"])
            evidence = (
                f"5 primitives; poisoned masked cells by `1e9`; max valid-cell drift `{worst:.3g}`"
            )
        elif check["name"] == "forward_label_boundary":
            evidence = "last row masked and zero; endpoint tradability mask matched exactly"
        elif check["name"] == "lagged_execution_alignment":
            evidence = "controlled path matched `weights[t-1] × returns[t]` exactly"
        else:
            evidence = f"10 bps / 1 bps cumulative cost ratio `{details['observed_ratio']:.6g}`"
        lines.append(f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {evidence} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The audit checks implementation invariants that are easy to state and easy to get wrong:",
            "masked values must not affect valid outputs in core tensor primitives, labels must not",
            "cross the end of the sample, positions must be lagged before earning returns, and cost",
            "drag must scale with the configured fee.",
            "",
            "It does **not** establish profitable alpha, real-market fidelity, or production readiness.",
            "Those questions require public or licensed market data, walk-forward evaluation, realistic",
            "execution assumptions, and independent reproduction.",
            "",
            f"Source: [{PROJECT_URL}]({PROJECT_URL})",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "technical_audit.json"
    markdown_path = output_dir / "technical_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(report))
    return markdown_path, json_path


@click.command()
@click.option("--n-dates", default=180, show_default=True, type=click.IntRange(min=60))
@click.option("--n-stocks", default=40, show_default=True, type=click.IntRange(min=10))
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--output-dir",
    default="artifacts/technical_audit",
    show_default=True,
    type=click.Path(path_type=Path),
)
def main(n_dates: int, n_stocks: int, seed: int, output_dir: Path) -> None:
    """Audit implementation invariants on deterministic synthetic data."""
    report = run_audit(n_dates=n_dates, n_stocks=n_stocks, seed=seed)
    markdown_path, json_path = write_report(report, output_dir)
    click.echo(_markdown(report))
    click.echo(f"JSON: {json_path}")
    click.echo(f"Markdown: {markdown_path}")
    if not report["passed"]:
        raise click.ClickException("one or more technical audit checks failed")


if __name__ == "__main__":  # pragma: no cover
    main()
