#!/usr/bin/env python3
"""Compare the complete 213-factor engine in fresh CPU subprocesses.

Synthetic engineering evidence only. Timings include factor construction,
cross-sectional z-scoring and stacking, but exclude panel generation, saving
and diagnostics. Numerical/rank differences are reported, never thresholded
away to imply interchangeable research signals.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
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
import scipy
from scipy.stats import rankdata
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_legacy_set, using_rolling_backend
from mlquant.features.bias import limit_move_mask
from mlquant.features.legacy_factors import LEGACY_REGISTRY
from mlquant.features.tensor_factors import cs_rank, ts_corr


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def input_hash(panel):
    digest = hashlib.sha256()
    for field in fields(panel):
        value = getattr(panel, field.name)
        digest.update(field.name.encode())
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def rank_correlation_oracle(panel, backend):
    """Untimed direct FP64 oracle for the discrete-rank correlation in old_063.

    The rank inputs are identical across backends. Include constant windows
    because tiny FP32 residuals there can change subsequent cross-section ranks.
    This audits the moment, not the entire nonlinear factor as an oracle.
    """
    x, xm = cs_rank(panel.vwap, panel.mask)
    y, ym = cs_rank(panel.volume, panel.mask)
    mask = xm & ym
    with using_rolling_backend(backend):
        actual, valid = ts_corr(x, y, mask, 4)
    errors, constants, examples = [], [], []
    for t in np.unique(np.linspace(3, panel.n_dates - 1, 25, dtype=int)):
        for col in np.unique(np.linspace(0, panel.n_stocks - 1, 12, dtype=int)):
            sl = slice(t - 3, t + 1)
            good = mask[sl, col].numpy()
            a, b = x[sl, col].numpy()[good].astype(np.float64), y[sl, col].numpy()[good].astype(np.float64)
            if len(a) < 2:
                continue
            ca, cb = a - a.mean(), b - b.mean()
            expected = float(np.dot(ca, cb) / max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-12))
            observed = float(actual[t, col])
            error = abs(observed - expected)
            errors.append(error)
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                constants.append(error)
                if len(examples) < 3:
                    examples.append({"date_index": int(t), "stock_index": int(col),
                                     "x": a.tolist(), "y": b.tolist(),
                                     "expected": expected, "actual": observed})
            if not valid[t, col]:
                raise RuntimeError("Rank-correlation oracle mask differs")
    return {"formula": "corr(cs_rank(vwap), cs_rank(volume), 4); moment used in old_063",
            "sample_cells": len(errors), "max_abs_error": max(errors, default=None),
            "constant_sample_cells": len(constants), "constant_max_abs_error": max(constants, default=None),
            "constant_examples": examples}


def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    panel = make_synthetic_panel(SyntheticConfig(
        n_dates=args.n_dates, n_stocks=args.n_stocks, seed=args.seed,
    ))
    panel = replace(panel, mask=limit_move_mask(panel, limit_pct=0.098))
    digest, input_rss = input_hash(panel), peak_rss()
    seconds = []
    with torch.no_grad():
        for iteration in range(args.warmup + args.repeat):
            start = time.perf_counter()
            values, joint_mask, names = compute_legacy_set(panel, rolling_backend=args.worker)
            elapsed = time.perf_counter() - start
            if iteration >= args.warmup:
                seconds.append(elapsed)
            if iteration < args.warmup + args.repeat - 1:
                del values, joint_mask
        # Snapshot before serialization or per-factor diagnostic allocations.
        engine_peak = peak_rss()
        np.save(args.worker_output / "values.npy", values.numpy())
        np.save(args.worker_output / "joint-mask.npy", joint_mask.numpy())
        per_factor_masks = []
        with using_rolling_backend(args.worker):
            for name in names:
                _, valid = LEGACY_REGISTRY[name](panel)
                per_factor_masks.append(valid.numpy())
        masks = np.stack(per_factor_masks, axis=-1)
        np.save(args.worker_output / "factor-masks.npy", masks)
    result = {"backend": args.worker, "seconds": seconds,
              "median_seconds": statistics.median(seconds),
              "input_peak_rss_bytes": input_rss, "engine_peak_rss_bytes": engine_peak,
              "shape": list(values.shape), "output_tensor_bytes": values.numel() * values.element_size(),
              "input_sha256": digest, "names": names,
              "joint_mask_matches_factor_intersection": bool(np.array_equal(
                  masks.all(axis=-1), joint_mask.numpy())),
              "valid_joint_cells": int(joint_mask.sum()),
              "nonfinite_output_cells": int((~torch.isfinite(values)).sum()),
              "rank_correlation_oracle": rank_correlation_oracle(panel, args.worker)}
    print(json.dumps(result), flush=True)


def rank_diagnostics(a, b, valid):
    shifts, correlations = [], []
    for t in np.unique(np.linspace(0, len(a) - 1, 25, dtype=int)):
        good = valid[t]
        if good.sum() < 2:
            continue
        ra = rankdata(a[t, good], method="average")
        rb = rankdata(b[t, good], method="average")
        shifts.extend((np.abs(ra - rb) / good.sum()).tolist())
        if ra.std() > 0 and rb.std() > 0:
            correlations.append(float(np.corrcoef(ra, rb)[0, 1]))
    return {"sample_dates": 25, "sample_rank_cells": len(shifts),
            "mean_abs_percentile_rank_shift": float(np.mean(shifts)) if shifts else None,
            "max_abs_percentile_rank_shift": max(shifts, default=None),
            "mean_spearman": float(np.mean(correlations)) if correlations else None,
            "spearman_nonconstant_dates": len(correlations)}


def compare_outputs(paths, names):
    a, b = [np.load(path / "values.npy", mmap_mode="r") for path in paths]
    ma, mb = [np.load(path / "factor-masks.npy", mmap_mode="r") for path in paths]
    ja, jb = [np.load(path / "joint-mask.npy", mmap_mode="r") for path in paths]
    rows = []
    for index, name in enumerate(names):
        av, bv = a[:, :, index], b[:, :, index]
        am, bm = ma[:, :, index], mb[:, :, index]
        common = am & bm
        finite = common & np.isfinite(av) & np.isfinite(bv)
        error = np.abs(av[finite].astype(np.float64) - bv[finite].astype(np.float64))
        row = {"name": name, "unfold_valid_cells": int(am.sum()), "linear_valid_cells": int(bm.sum()),
               "mask_different_cells": int((am != bm).sum()), "common_valid_cells": int(common.sum()),
               "nonfinite_common_cells": int((common & ~finite).sum()),
               "mean_abs_difference": float(error.mean()) if error.size else None,
               "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
               "p99_abs_difference": float(np.quantile(error, 0.99)) if error.size else None,
               "max_abs_difference": float(error.max()) if error.size else None,
               "fraction_abs_difference_gt_1e-5": float(np.mean(error > 1e-5)) if error.size else None,
               **rank_diagnostics(av, bv, finite)}
        rows.append(row)
    return {"joint_masks_equal": bool(np.array_equal(ja, jb)),
            "joint_mask_different_cells": int((ja != jb).sum()),
            "all_factor_masks_equal": bool(np.array_equal(ma, mb)),
            "factors_with_max_difference_gt_1e-5": sum(
                row["max_abs_difference"] is not None and row["max_abs_difference"] > 1e-5
                for row in rows), "per_factor": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dates", type=int, default=500)
    parser.add_argument("--n-stocks", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/factor-backends.json"))
    parser.add_argument("--worker", choices=["unfold", "linear"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.n_dates < 100 or min(args.n_stocks, args.repeat, args.timeout) < 1 or args.warmup < 0:
        parser.error("at least 100 dates, positive stocks/repeat/timeout and nonnegative warmup required")
    if args.worker:
        worker(args)
        return
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {
        "category": "synthetic CPU systems benchmark; no alpha or investment evidence",
        "protocol": "full-factor-backends-v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
        "pytorch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "os": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "gpu": "not used", "device": "cpu", "threads": 1, "interop_threads": 1,
        "data_source": "SyntheticConfig seeded GBM; default halt_prob .005; limit_move_mask .098 before factors",
        "cost_assumptions": "No market data, trading or transaction costs in this systems benchmark.",
        "settings": {k: v for k, v in vars(args).items()
                     if k not in ("json_out", "worker", "worker_output")},
        "neutralize": True, "factor_count": len(LEGACY_REGISTRY),
        "limitations": [
            "One CPU environment; no CUDA performance measured. Linear requires FP64, unsupported on MPS.",
            "Peak process RSS includes interpreter, output tensors and allocator caching, not exact allocation counts.",
            "All 213 factors include unchanged ranks, extrema, sums and EWMA; kernel gains need not transfer to the full engine.",
            "Backend order is unfold then linear; warm median is reported without a hardware-ranking claim.",
            "Linear filters nonfinite inputs and clamps correlations; precision changes can alter downstream ranks.",
            "Prefix subtraction can lose precision on very long, strongly drifting series.",
            "Rank diagnostics sample 25 evenly spaced dates; value and mask diagnostics cover every factor's own valid cells.",
            "Numerical differences are observations, not signal-equivalence or trading-quality validation.",
        ], "cases": [], "comparison": None, "failures": [],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for backend in ("unfold", "linear"):
            path = Path(tmp) / backend
            path.mkdir()
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                   "--worker-output", str(path), "--n-dates", str(args.n_dates),
                   "--n-stocks", str(args.n_stocks), "--seed", str(args.seed),
                   "--repeat", str(args.repeat), "--warmup", str(args.warmup)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                if result.returncode:
                    raise RuntimeError(f"exit {result.returncode}: {result.stderr}")
                case = json.loads(result.stdout)
                case["command"] = shlex.join(cmd)
                report["cases"].append(case)
                paths.append(path)
                print(json.dumps({k: v for k, v in case.items() if k != "names"}), flush=True)
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                report["failures"].append({"backend": backend, "error": str(error), "command": shlex.join(cmd)})
            args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if len(paths) == 2:
            ca, cb = report["cases"]
            if ca["input_sha256"] != cb["input_sha256"] or ca["names"] != cb["names"]:
                report["failures"].append({"error": "Backend inputs or factor names differ"})
            else:
                report["comparison"] = compare_outputs(paths, ca["names"])
                report["comparison"]["speedup"] = ca["median_seconds"] / cb["median_seconds"]
            for case in report["cases"]:
                if case["nonfinite_output_cells"] or not case["joint_mask_matches_factor_intersection"]:
                    report["failures"].append({"backend": case["backend"], "error": "nonfinite output or inconsistent joint mask"})
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit("benchmark failures preserved in JSON")


if __name__ == "__main__":
    main()
