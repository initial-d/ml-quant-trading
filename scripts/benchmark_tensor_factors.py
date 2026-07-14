"""Benchmark tensor factor primitives on synthetic panel data.

Usage:
    python scripts/benchmark_tensor_factors.py --device auto
    python scripts/benchmark_tensor_factors.py --device cpu --n-dates 750 --n-stocks 1000
"""
from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import click
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import cs_rank, ewma, ts_corr, ts_mean, ts_rank
from mlquant.features.legacy_factors import compute_legacy_set


@dataclass(frozen=True)
class BenchCase:
    name: str
    fn: Callable[[], Any]


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_case(case: BenchCase, *, device: str, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        case.fn()
    _sync(device)

    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        case.fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)

    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def _devices(requested: str) -> list[str]:
    if requested == "auto":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices
    if requested == "cuda" and not torch.cuda.is_available():
        click.echo("CUDA requested but unavailable; skipping benchmark.")
        return []
    return [requested]


def _format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.3f} s"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _environment_row(field: str, value: object) -> str:
    return f"| {_markdown_cell(field)} | {_markdown_cell(value)} |"


def _benchmark_row(device: str, case_name: str, mean_s: float, std_s: float, peak_mem: str) -> str:
    return (
        f"| {_markdown_cell(device)} | `{_markdown_cell(case_name)}` | "
        f"{_markdown_cell(_format_seconds(mean_s))} | {_markdown_cell(_format_seconds(std_s))} | "
        f"{_markdown_cell(peak_mem)} |"
    )


def _print_environment(*, n_dates: int, n_stocks: int, repeat: int, warmup: int) -> None:
    click.echo("# Tensor Factor Benchmark")
    click.echo("")
    click.echo("| Field | Value |")
    click.echo("| --- | --- |")
    click.echo(_environment_row("Python", platform.python_version()))
    click.echo(_environment_row("Platform", platform.platform()))
    click.echo(_environment_row("PyTorch", torch.__version__))
    click.echo(_environment_row("CUDA available", torch.cuda.is_available()))
    if torch.cuda.is_available():
        click.echo(_environment_row("CUDA device", torch.cuda.get_device_name(0)))
    click.echo(_environment_row("Synthetic panel", f"{n_dates} dates x {n_stocks} stocks"))
    click.echo(_environment_row("Warmup / repeat", f"{warmup} / {repeat}"))
    click.echo("")


@click.command()
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--n-dates", default=750, show_default=True, type=int)
@click.option("--n-stocks", default=1000, show_default=True, type=int)
@click.option("--window", default=20, show_default=True, type=int)
@click.option("--repeat", default=5, show_default=True, type=int)
@click.option("--warmup", default=2, show_default=True, type=int)
def main(device: str, n_dates: int, n_stocks: int, window: int, repeat: int, warmup: int) -> None:
    """Run a compact benchmark for core tensor factor operations."""
    if n_dates <= window:
        raise click.BadParameter("--n-dates must be greater than --window")
    if repeat <= 0 or warmup < 0:
        raise click.BadParameter("--repeat must be positive and --warmup cannot be negative")

    selected_devices = _devices(device)
    if not selected_devices:
        return

    _print_environment(n_dates=n_dates, n_stocks=n_stocks, repeat=repeat, warmup=warmup)

    click.echo("| Device | Case | Mean | Std | Peak CUDA memory |")
    click.echo("| --- | --- | ---: | ---: | ---: |")

    for dev in selected_devices:
        panel = make_synthetic_panel(
            SyntheticConfig(n_dates=n_dates, n_stocks=n_stocks, device=dev, seed=42)
        )
        returns = panel.returns
        factor_subset = ("best_001", "best_002", "original_001", "stock_001", "add_015", "old_042")

        cases = [
            BenchCase("cs_rank(close)", lambda panel=panel: cs_rank(panel.close, panel.mask)),
            BenchCase(f"ts_mean(close,{window})", lambda panel=panel: ts_mean(panel.close, panel.mask, window)),
            BenchCase(f"ts_rank(close,{window})", lambda panel=panel: ts_rank(panel.close, panel.mask, window)),
            BenchCase(
                f"ts_corr(close,returns,{window})",
                lambda panel=panel, returns=returns: ts_corr(panel.close, returns, panel.mask, window),
            ),
            BenchCase("ewma(close,0.05)", lambda panel=panel: ewma(panel.close, panel.mask, 0.05)),
            BenchCase(
                "compute_legacy_set(6 factors)",
                lambda panel=panel, factor_subset=factor_subset: compute_legacy_set(panel, names=factor_subset),
            ),
        ]

        for case in cases:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            mean_s, std_s = _time_case(case, device=dev, warmup=warmup, repeat=repeat)
            peak_mem = "-"
            if dev == "cuda":
                peak_mem = f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
            click.echo(_benchmark_row(dev, case.name, mean_s, std_s, peak_mem))


if __name__ == "__main__":  # pragma: no cover
    main()
