"""Benchmark tensor factor primitives on synthetic panel data.

Usage:
    python scripts/benchmark_tensor_factors.py --device cpu
    python scripts/benchmark_tensor_factors.py --device cuda
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import cs_rank, ewma, ts_corr, ts_mean, ts_rank
from mlquant.features.legacy_factors import compute_legacy_set

PROTOCOL_VERSION = "v1"


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


def _configure_threads(threads: int, interop_threads: int) -> None:
    """Pin PyTorch thread pools before any benchmark tensor work starts."""
    if threads <= 0 or interop_threads <= 0:
        raise ValueError("thread counts must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)


def _cpu_name() -> str:
    """Return a portable best-effort CPU identifier."""
    return platform.processor() or platform.machine() or "unknown"


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


def _environment_payload(
    *,
    n_dates: int,
    n_stocks: int,
    window: int,
    repeat: int,
    warmup: int,
    seed: int,
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "pytorch": torch.__version__,
        "pytorch_threads": torch.get_num_threads(),
        "pytorch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_dates": n_dates,
        "n_stocks": n_stocks,
        "window": window,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
    }


def _print_environment(environment: dict[str, object]) -> None:
    click.echo("# Tensor Factor Benchmark")
    click.echo("")
    click.echo("| Field | Value |")
    click.echo("| --- | --- |")
    display_rows = (
        ("Protocol", environment["protocol"]),
        ("Python", environment["python"]),
        ("Platform", environment["platform"]),
        ("CPU", environment["cpu"]),
        ("Logical CPUs", environment["logical_cpus"] or "unknown"),
        ("PyTorch", environment["pytorch"]),
        ("PyTorch threads", environment["pytorch_threads"]),
        ("PyTorch interop threads", environment["pytorch_interop_threads"]),
        ("CUDA available", environment["cuda_available"]),
        ("CUDA device", environment["cuda_device"] or "-"),
        ("Synthetic panel", f"{environment['n_dates']} dates x {environment['n_stocks']} stocks"),
        ("Window", environment["window"]),
        ("Warmup / repeat", f"{environment['warmup']} / {environment['repeat']}"),
        ("Seed", environment["seed"]),
    )
    for field, value in display_rows:
        click.echo(_environment_row(field, value))
    click.echo("")


def _write_json_report(path: Path, environment: dict[str, object], results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "environment": environment, "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@click.command()
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--n-dates", default=750, show_default=True, type=int)
@click.option("--n-stocks", default=1000, show_default=True, type=int)
@click.option("--window", default=20, show_default=True, type=int)
@click.option("--repeat", default=10, show_default=True, type=int)
@click.option("--warmup", default=3, show_default=True, type=int)
@click.option("--threads", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--interop-threads", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    device: str,
    n_dates: int,
    n_stocks: int,
    window: int,
    repeat: int,
    warmup: int,
    threads: int,
    interop_threads: int,
    seed: int,
    json_out: Path | None,
) -> None:
    """Run a compact benchmark for core tensor factor operations."""
    if n_dates <= window:
        raise click.BadParameter("--n-dates must be greater than --window")
    if repeat <= 0 or warmup < 0:
        raise click.BadParameter("--repeat must be positive and --warmup cannot be negative")

    _configure_threads(threads, interop_threads)
    selected_devices = _devices(device)
    if not selected_devices:
        return

    environment = _environment_payload(
        n_dates=n_dates,
        n_stocks=n_stocks,
        window=window,
        repeat=repeat,
        warmup=warmup,
        seed=seed,
    )
    _print_environment(environment)

    click.echo("| Device | Case | Mean | Std | Peak CUDA memory |")
    click.echo("| --- | --- | ---: | ---: | ---: |")
    results: list[dict[str, object]] = []

    for dev in selected_devices:
        panel = make_synthetic_panel(
            SyntheticConfig(n_dates=n_dates, n_stocks=n_stocks, device=dev, seed=seed)
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
            results.append(
                {
                    "device": dev,
                    "case": case.name,
                    "mean_seconds": mean_s,
                    "std_seconds": std_s,
                    "peak_cuda_memory": peak_mem,
                }
            )

    if json_out is not None:
        _write_json_report(json_out, environment, results)
        click.echo(f"JSON report: {json_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
