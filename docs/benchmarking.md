# Benchmarking Tensor Factors

This page explains how to benchmark the tensor factor engine on synthetic data. The benchmark is designed to be reproducible without proprietary market data.

## Quick Start

```bash
python -m pip install -e .[dev]
python scripts/benchmark_tensor_factors.py --device auto
```

`--device auto` always runs CPU and also runs CUDA when PyTorch can see a GPU.

## Larger Panel

```bash
python scripts/benchmark_tensor_factors.py \
  --device auto \
  --n-dates 1500 \
  --n-stocks 3000 \
  --window 20 \
  --repeat 5 \
  --warmup 2
```

The script prints a Markdown table with environment details, mean runtime, runtime standard deviation, and peak CUDA memory when available.

## What It Measures

The benchmark covers representative operations:

- `cs_rank(close)`
- rolling `ts_mean`
- rolling `ts_rank`
- rolling `ts_corr`
- `ewma`
- `compute_legacy_set` on a six-factor subset

The goal is to compare factor-engine behavior across machines and devices, not to claim universal throughput. Results depend on PyTorch version, CPU, GPU, CUDA runtime, memory bandwidth, panel shape, factor subset, and mask density.

## Reporting Results

When sharing benchmark results, include:

- commit SHA
- command
- Python version
- PyTorch version
- CPU and GPU model
- CUDA availability and CUDA device name
- `n_dates`, `n_stocks`, `window`, `repeat`, and `warmup`
- whether the run used CPU, CUDA, or both

## Interpreting Results

Small panels may not benefit from GPU execution because transfer overhead and kernel launch overhead dominate runtime. Larger panels and wider universes are more likely to show GPU advantages, especially for rolling operations and multi-factor computation.

Use this benchmark as an engineering diagnostic. It is separate from backtest quality, factor IC, or trading performance.
