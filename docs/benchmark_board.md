# Community Benchmark Board

This page tracks benchmark reports shared by users. Submit results with the
`Benchmark result` issue template.

For trading-workflow validation rather than tensor throughput, use
[`public_data_validation.md`](public_data_validation.md). That path reports
walk-forward baselines, transaction costs, slippage, turnover, and drawdown.
It also writes a copy-ready `submission.md` and machine-readable JSON files for
community reports.

## How to Submit

```bash
python -m pip install -e '.[dev]'
python scripts/benchmark_tensor_factors.py \
  --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
  --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

Then open a benchmark issue, paste the printed Markdown table, and attach the
generated `artifacts/benchmark-v1.json`. `make benchmark` is an equivalent
shortcut on systems with GNU Make.

The command above is the canonical protocol v1 CPU run. It fixes the synthetic
panel at 750 dates x 1000 stocks, the 20-day window, seed 42, 3 warmups, 10
measured repetitions, and both PyTorch thread pools at one thread. Optional GPU,
larger-panel, and multi-thread runs should be reported separately.

For larger panels:

```bash
python scripts/benchmark_tensor_factors.py \
  --device auto \
  --n-dates 1500 \
  --n-stocks 3000 \
  --window 20 \
  --repeat 10 \
  --warmup 3 \
  --threads 1 \
  --interop-threads 1 \
  --seed 42
```

## Comparable Protocol v1 Results

| Contributor | Commit | OS | Python | PyTorch | Threads | CPU | GPU | Command | Notes |
|---|---|---|---|---|---:|---|---|---|---|
| Maintainer | `2765d19` | Windows 11 10.0.26200 | 3.14.4 | 2.11.0+cpu | 1 / 1 | Intel Core i7-1255U, 10 cores / 12 threads | none | `make benchmark` | CPU-only protocol v1 baseline |
| [@sergio12S](https://github.com/sergio12S) | `2a91c6b` | macOS 26.5.2 arm64 | 3.10.16 | 2.13.0 | 1 / 1 | Apple M4, 10 logical CPUs | none | canonical protocol v1 command in [issue #59](https://github.com/initial-d/ml-quant-trading/issues/59) | Community CPU report; `ts_corr` unstable across repeats |

### Maintainer Protocol v1: Intel Core i7-1255U

Environment:

- Commit: `2765d19`
- Protocol: `v1`
- Machine: Intel Core i7-1255U, 10 cores / 12 threads
- OS: Windows 11 10.0.26200
- Python: 3.14.4
- PyTorch: 2.11.0+cpu
- PyTorch threads / interop threads: 1 / 1
- CUDA available: false
- Synthetic panel: 750 dates x 1000 stocks
- Window: 20
- Warmup / repeat: 3 / 10
- Seed: 42

| Device | Case | Mean | Std | Peak CUDA memory |
| --- | --- | ---: | ---: | ---: |
| cpu | `cs_rank(close)` | 77.4 ms | 2.8 ms | - |
| cpu | `ts_mean(close,20)` | 23.8 ms | 2.3 ms | - |
| cpu | `ts_rank(close,20)` | 76.1 ms | 11.1 ms | - |
| cpu | `ts_corr(close,returns,20)` | 152.3 ms | 9.2 ms | - |
| cpu | `ewma(close,0.05)` | 16.2 ms | 1.1 ms | - |
| cpu | `compute_legacy_set(6 factors)` | 551.0 ms | 45.9 ms | - |

This is the first comparable v1 report. Run the same command on another machine
and submit the complete output through the benchmark issue template.

### Community Protocol v1: Apple M4

Environment and raw results are preserved in [issue #59](https://github.com/initial-d/ml-quant-trading/issues/59).
This is a community-submitted, CPU-only protocol v1 report from a fresh checkout.
It is included for reproducibility and hardware context, not as a controlled
ranking against the maintainer result.

Environment:

- Contributor: [@sergio12S](https://github.com/sergio12S)
- Commit: `2a91c6b199d3f3608bbebacb36addb37949200b9`
- Protocol: `v1`
- Machine: Apple M4, 10 logical CPUs
- OS: macOS 26.5.2 arm64
- Python: 3.10.16
- PyTorch: 2.13.0
- PyTorch threads / interop threads: 1 / 1
- CUDA available: false
- Synthetic panel: 750 dates x 1000 stocks
- Window: 20
- Warmup / repeat: 3 / 10
- Seed: 42
- Exact command: the canonical protocol v1 command shown above

| Device | Case | Mean | Std | Peak CUDA memory |
| --- | --- | ---: | ---: | ---: |
| cpu | `cs_rank(close)` | 24.4 ms | 218.0 us | - |
| cpu | `ts_mean(close,20)` | 7.1 ms | 499.6 us | - |
| cpu | `ts_rank(close,20)` | 15.9 ms | 489.3 us | - |
| cpu | `ts_corr(close,returns,20)` | 36.9 ms | 4.7 ms | - |
| cpu | `ewma(close,0.05)` | 3.6 ms | 158.5 us | - |
| cpu | `compute_legacy_set(6 factors)` | 136.3 ms | 6.6 ms | - |

The contributor repeated the command on the same machine: `ts_corr` moved from
36.9 ms to 43.4 ms and its standard deviation rose from 4.7 ms to 12.1 ms,
while the other cases stayed within roughly 10%. The report also notes that a
historical `ts_rank` difference across commits is confounded by benchmark-script
and runtime changes. Keep both caveats attached to the numbers; do not infer a
PyTorch or hardware speedup from them.

## Pre-Protocol Machine Snapshots

The following results predate protocol v1. They did not pin PyTorch thread pools
or software versions and therefore must not be used as a controlled CPU ranking.

![Pre-protocol CPU benchmark snapshots](assets/tensor-benchmark-cpu.svg)

| Contributor | Commit | OS | Python | PyTorch | CUDA | CPU | GPU | Command | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Maintainer | `44c6777` | Windows 11 10.0.26200 | 3.14.4 | 2.11.0+cpu | unavailable | Intel Core i7-1255U, 10 cores / 12 threads | none | `python scripts/benchmark_tensor_factors.py --device auto --n-dates 750 --n-stocks 1000 --window 20 --repeat 5 --warmup 2` | CPU-only Windows report |
| Maintainer | `d3a99b6` | macOS 26.5.1 arm64 | 3.9.6 | 2.8.0 | unavailable | Apple M5, 10 cores, 32 GB RAM | none | `python scripts/benchmark_tensor_factors.py --device auto` | CPU-only report on MacBook Air |

### Maintainer CPU Baseline: Intel Core i7-1255U

Environment:

- Commit: `44c6777`
- Machine: Intel Core i7-1255U, 10 cores / 12 threads
- OS: Windows 11 10.0.26200
- Python: 3.14.4
- PyTorch: 2.11.0+cpu
- CUDA available: false
- CUDA version: unavailable
- Synthetic panel: 750 dates x 1000 stocks
- Warmup / repeat: 2 / 5

| Device | Case | Mean | Std | Peak CUDA memory |
| --- | --- | ---: | ---: | ---: |
| cpu | `cs_rank(close)` | 23.4 ms | 2.5 ms | - |
| cpu | `ts_mean(close,20)` | 21.1 ms | 1.7 ms | - |
| cpu | `ts_rank(close,20)` | 71.3 ms | 6.3 ms | - |
| cpu | `ts_corr(close,returns,20)` | 151.2 ms | 10.2 ms | - |
| cpu | `ewma(close,0.05)` | 50.4 ms | 15.6 ms | - |
| cpu | `compute_legacy_set(6 factors)` | 287.0 ms | 15.8 ms | - |

The Windows and macOS baselines use different Python and PyTorch versions and
unrecorded default thread settings. They are retained for provenance only.

### Maintainer CPU Baseline: Apple M5 MacBook Air

Environment:

- Commit: `d3a99b6`
- Machine: MacBook Air, Apple M5, 10 cores, 32 GB RAM
- OS: macOS 26.5.1 arm64
- Python: 3.9.6
- PyTorch: 2.8.0
- CUDA available: false
- CUDA version: unavailable
- Synthetic panel: 750 dates x 1000 stocks
- Warmup / repeat: 2 / 5

| Device | Case | Mean | Std | Peak CUDA memory |
| --- | --- | ---: | ---: | ---: |
| cpu | `cs_rank(close)` | 7.4 ms | 151.8 us | - |
| cpu | `ts_mean(close,20)` | 3.6 ms | 356.1 us | - |
| cpu | `ts_rank(close,20)` | 11.7 ms | 504.5 us | - |
| cpu | `ts_corr(close,returns,20)` | 20.8 ms | 618.6 us | - |
| cpu | `ewma(close,0.05)` | 3.2 ms | 87.9 us | - |
| cpu | `compute_legacy_set(6 factors)` | 60.0 ms | 807.6 us | - |

## What Makes a Good Benchmark Report

- The command is copy-pasted exactly.
- Protocol v1 is present in the environment table.
- Both PyTorch thread counts are recorded.
- The commit SHA is included.
- CPU and GPU names are included.
- CUDA availability is stated.
- The result table is pasted without editing numbers.
- Any unusual conditions are documented, such as shared GPU, thermal throttling, or low memory.

## Why This Helps

Benchmark reports help users answer practical questions:

- Does GPU help for my panel size?
- Which operations dominate runtime?
- Is performance sensitive to PyTorch or CUDA version?
- What hardware is enough for a student or research workflow?
