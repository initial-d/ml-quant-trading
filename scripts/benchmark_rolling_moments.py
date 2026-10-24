#!/usr/bin/env python3
"""Controlled CPU window sweep; fresh process per kernel and window.

Synthetic systems evidence only. Peak RSS includes interpreter, tensors and
allocator caching; it is not an exact allocation count or trading evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from mlquant.features import ts_corr, ts_corr_linear, ts_mean, ts_mean_linear


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def inputs(args):
    gen = torch.Generator().manual_seed(args.seed)
    x = 100 + torch.randn(args.n_dates, args.n_stocks, generator=gen)
    y = 0.6 * (x - 100) + torch.randn(x.shape, generator=gen)
    mask = torch.rand(x.shape, generator=gen) >= 0.1
    return x, y, mask


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    x, y, mask = inputs(args)
    input_rss = peak_rss()
    functions = {"mean-unfold": ts_mean, "mean-linear": ts_mean_linear,
                 "corr-unfold": ts_corr, "corr-linear": ts_corr_linear}
    function = functions[args.worker]
    data = (x, y, mask, args.window) if args.worker.startswith("corr") else (x, mask, args.window)
    elapsed = []
    with torch.no_grad():
        for iteration in range(args.warmup + args.repeat):
            start = time.perf_counter()
            out, valid = function(*data)
            duration = time.perf_counter() - start
            if iteration >= args.warmup:
                elapsed.append(duration)
            # Avoid overlap with the next invocation's result tensors.
            del out, valid
        kernel_rss = peak_rss()
        out, valid = function(*data)
    np.savez(args.worker_output, values=out.numpy(), mask=valid.numpy())
    print(json.dumps({"kernel": args.worker, "window": args.window,
                      "median_seconds": statistics.median(elapsed), "seconds": elapsed,
                      "peak_process_rss_bytes": kernel_rss, "input_peak_rss_bytes": input_rss,
                      "input_sha256": hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()
                                                     + mask.numpy().tobytes()).hexdigest()}))


def accuracy(args, paths, kind, window):
    a, b = [np.load(path) for path in paths]
    equal_mask = np.array_equal(a["mask"], b["mask"])
    error = np.abs(a["values"][a["mask"]] - b["values"][a["mask"]])
    x, y, mask = [v.numpy() for v in inputs(args)]
    oracle_errors = {"unfold": [], "linear": []}
    # Independent double precision window oracle at deterministic sample cells.
    for t in np.unique(np.linspace(window - 1, args.n_dates - 1, 20, dtype=int)):
        for col in np.unique(np.linspace(0, args.n_stocks - 1, 12, dtype=int)):
            sl = slice(t - window + 1, t + 1)
            good = mask[sl, col]
            values = x[sl, col][good].astype(np.float64)
            if len(values) < (2 if kind == "corr" else 1):
                continue
            if kind == "mean":
                expected = values.mean()
            else:
                other = y[sl, col][good].astype(np.float64)
                ca, cb = values - values.mean(), other - other.mean()
                expected = np.dot(ca, cb) / max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-12)
            for label, actual in zip(("unfold", "linear"), (a, b)):
                oracle_errors[label].append(abs(float(actual["values"][t, col]) - expected))
    max_error = float(error.max()) if error.size else 0.0
    return {"masks_equal": equal_mask, "max_valid_difference": max_error,
            "sample_oracle_max_abs_error": {k: max(v, default=0) for k, v in oracle_errors.items()},
            "passed": equal_mask and max_error < (1e-4 if kind == "mean" else 1e-5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dates", type=int, default=750)
    parser.add_argument("--n-stocks", type=int, default=1000)
    parser.add_argument("--windows", default="20,60,252")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/rolling-moments.json"))
    parser.add_argument("--worker", choices=["mean-unfold", "mean-linear", "corr-unfold", "corr-linear"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    windows = [int(w) for w in args.windows.split(",")]
    if min(args.n_dates, args.n_stocks, args.repeat) < 1 or args.warmup < 0 or any(
            w < 1 or w > args.n_dates for w in windows):
        parser.error("positive panel/repeat/windows required; windows cannot exceed dates")
    if args.worker:
        worker(args)
        return
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"category": "synthetic CPU systems benchmark; no alpha or investment evidence",
              "protocol": "rolling-moments-v1", "commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "numpy": np.__version__, "os": platform.platform(),
              "cpu": cpu, "logical_cpus": os.cpu_count(), "gpu": "not used", "device": "cpu",
              "threads": 1, "interop_threads": 1,
              "data_source": "seeded normal synthetic panel, x level 100, correlated y, 10% missing",
              "cost_assumptions": "no market data, trading or transaction costs in systems benchmark",
              "settings": {k: v for k, v in vars(args).items()
                           if k not in ("json_out", "worker_output", "worker", "window")},
              "limitations": ["One CPU/software environment; no GPU throughput claim.",
                              "Process peak RSS includes interpreter and cached allocations.",
                              "Separate subprocesses; warm timings exclude input creation and correctness checks.",
                              "FP64 prefix subtraction can lose precision on very long, strongly drifting series.",
                              "Different finite-input policy: comparison panel contains only finite values."],
              "cases": [], "comparisons": [], "failures": []}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for window in windows:
            for kind in ("mean", "corr"):
                paths, cases = [], []
                for method in ("unfold", "linear"):
                    path = Path(tmp) / f"{kind}-{method}-{window}.npz"
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", f"{kind}-{method}",
                           "--window", str(window), "--worker-output", str(path),
                           "--n-dates", str(args.n_dates), "--n-stocks", str(args.n_stocks),
                           "--windows", str(window), "--seed", str(args.seed),
                           "--repeat", str(args.repeat), "--warmup", str(args.warmup)]
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                        if result.returncode:
                            raise RuntimeError(f"exit {result.returncode}: {result.stderr}")
                        case = json.loads(result.stdout)
                        case["command"] = shlex.join(cmd)
                        report["cases"].append(case)
                        paths.append(path)
                        cases.append(case)
                        print(json.dumps(case), flush=True)
                    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                        report["failures"].append({"window": window, "kernel": f"{kind}-{method}",
                                                   "error": str(error), "command": shlex.join(cmd)})
                if len(paths) == 2:
                    check = accuracy(args, paths, kind, window)
                    if cases[0]["input_sha256"] != cases[1]["input_sha256"]:
                        check["passed"] = False
                    check.update({"kind": kind, "window": window,
                                  "speedup": cases[0]["median_seconds"] / cases[1]["median_seconds"]})
                    report["comparisons"].append(check)
                args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if report["failures"] or not all(c["passed"] for c in report["comparisons"]):
        raise SystemExit("benchmark/correctness failures preserved in JSON")


if __name__ == "__main__":
    main()
