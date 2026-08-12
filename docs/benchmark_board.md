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
make benchmark
```

Then open a benchmark issue and paste the printed Markdown table.

For larger panels:

```bash
python scripts/benchmark_tensor_factors.py \
  --device auto \
  --n-dates 1500 \
  --n-stocks 3000 \
  --window 20 \
  --repeat 5 \
  --warmup 2
```

## Results

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

The Windows and macOS baselines use different Python and PyTorch versions, so
they are reproducible machine snapshots rather than a controlled CPU ranking.
Use the same command and software environment for strict hardware comparisons.

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
