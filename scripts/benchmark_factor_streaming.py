#!/usr/bin/env python3
"""Full-factor reduction benchmark: stack versus consume/discard one factor.

Synthetic engineering evidence. Streaming serves per-factor screening or
incremental storage, not a replacement for a trainer needing all dimensions.
Both workloads compute all 213 neutralized factors and reduce each to mean/RMS.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_legacy_set, iter_legacy_factors
from mlquant.features.bias import limit_move_mask
from benchmark_factor_backends import input_hash, peak_rss


def reduce_values(values):
    double = values.double()
    return [float(double.mean()), float(double.square().mean().sqrt())]


def workload(panel, method, backend):
    if method == "stack":
        values, joint, names = compute_legacy_set(panel, rolling_backend=backend)
        result = [reduce_values(values[:, :, i]) for i in range(len(names))]
    else:
        result, names, joint = [], [], None
        for name, values, mask in iter_legacy_factors(panel, rolling_backend=backend):
            names.append(name)
            result.append(reduce_values(values))
            joint = mask if joint is None else joint & mask
            del values, mask
    return result, names, joint


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    panel = make_synthetic_panel(SyntheticConfig(n_dates=args.n_dates, n_stocks=args.n_stocks, seed=args.seed))
    panel = replace(panel, mask=limit_move_mask(panel, limit_pct=.098))
    digest = input_hash(panel)
    input_rss, times = peak_rss(), []
    with torch.no_grad():
        for i in range(args.warmup + args.repeat):
            start = time.perf_counter()
            summary, names, joint = workload(panel, args.worker, args.backend)
            if i >= args.warmup:
                times.append(time.perf_counter() - start)
        engine_peak = peak_rss()
        # A separate untimed pass checks exact bytes after the RSS snapshot.
        if args.worker == "stack":
            values, _, _ = compute_legacy_set(panel, rolling_backend=args.backend)
            hashes = [hashlib.sha256(values[:, :, i].contiguous().numpy().tobytes()).hexdigest()
                      for i in range(len(names))]
        else:
            hashes = [hashlib.sha256(value.numpy().tobytes()).hexdigest()
                      for _, value, _ in iter_legacy_factors(panel, rolling_backend=args.backend)]
    print(json.dumps({"method": args.worker, "backend": args.backend,
                      "seconds": times, "median_seconds": statistics.median(times),
                      "input_peak_rss_bytes": input_rss, "workload_peak_rss_bytes": engine_peak,
                      "input_sha256": digest, "factor_names": names,
                      "factor_value_sha256": hashes, "factor_mean_rms": summary,
                      "joint_mask_sha256": hashlib.sha256(joint.numpy().tobytes()).hexdigest()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dates", type=int, default=750)
    parser.add_argument("--n-stocks", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["unfold", "linear"], default="unfold")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--worker", choices=["stack", "stream"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.n_dates < 100 or min(args.n_stocks, args.repeat, args.timeout) < 1 or args.warmup < 0:
        parser.error("100+ dates, positive stocks/repeat/timeout and nonnegative warmup required")
    if args.worker:
        worker(args)
        return
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"category": "synthetic CPU engineering; no predictive-quality or trading evidence",
              "protocol": "factor-streaming-v1", "commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "pytorch": torch.__version__, "os": platform.platform(), "cpu": cpu,
              "gpu": "not used", "threads": 1, "interop_threads": 1,
              "data_source": "seeded SyntheticConfig GBM, default halt_prob .005, limit_move_mask .098 before factors",
              "cost_assumptions": "Local CPU systems workload; no trading, paid data or transaction costs.",
              "settings": {k: v for k, v in vars(args).items() if k not in ("json_out", "worker")},
              "limitations": ["One CPU; warm filesystem/allocator, fixed stack-then-stream order.",
                              "RSS includes interpreter, tensors and allocator caching; diagnostics follow its snapshot.",
                              "Per-factor mean/RMS screening workload; consumers retaining every output lose the streaming benefit.",
                              "Both methods compute the same 213 factors; this does not measure model training or predictive quality.",
                              "No CUDA performance claim; autograd graphs may accumulate unless consumers use no_grad."],
              "cases": [], "failures": [], "comparison": None}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    for method in ("stack", "stream"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
               "--n-dates", str(args.n_dates), "--n-stocks", str(args.n_stocks),
               "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--seed", str(args.seed),
               "--backend", args.backend, "--json-out", str(args.json_out)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            if result.returncode:
                raise RuntimeError(f"exit {result.returncode}: {result.stderr}")
            case = json.loads(result.stdout)
            case["command"] = shlex.join(cmd)
            report["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k in (
                "method", "backend", "median_seconds", "workload_peak_rss_bytes")}), flush=True)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            report["failures"].append({"method": method, "error": str(error), "command": shlex.join(cmd)})
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if len(report["cases"]) == 2:
        a, b = report["cases"]
        same = all(a[key] == b[key] for key in ("input_sha256", "factor_names", "factor_value_sha256", "joint_mask_sha256"))
        report["comparison"] = {"all_factor_bytes_and_joint_mask_equal": same,
                                "speedup": a["median_seconds"] / b["median_seconds"]}
        if not same:
            report["failures"].append({"error": "stream and stack values/masks/inputs differ"})
    args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("failures preserved in JSON")


if __name__ == "__main__":
    main()
