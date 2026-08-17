# Benchmarking Tensor Factors

This page explains how to benchmark the tensor factor engine on synthetic data. The benchmark is designed to be reproducible without proprietary market data.

## Quick Start

```bash
python -m pip install -e '.[dev]'
make benchmark
```

`make benchmark` runs the canonical **protocol v1** CPU command:

```bash
python scripts/benchmark_tensor_factors.py \
  --device cpu \
  --n-dates 750 \
  --n-stocks 1000 \
  --window 20 \
  --repeat 10 \
  --warmup 3 \
  --threads 1 \
  --interop-threads 1 \
  --seed 42
```

Protocol v1 fixes the panel, seed, rolling window, repetitions, and PyTorch
thread pools. Submit this CPU result before optional variants so reports remain
comparable across machines.

## Optional GPU Run

Run the same fixed workload on CUDA separately:

```bash
python scripts/benchmark_tensor_factors.py --device cuda
```

GPU results belong in their own comparison group. Do not interpret a CPU/GPU or
multi-thread/single-thread difference as a controlled hardware ranking.

## Larger Panel

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
- CUDA availability, CUDA version, and CUDA device name
- protocol version, thread counts, seed, panel size, window, repeat, and warmup
- whether the run used CPU, CUDA, or both

## Interpreting Results

Small panels may not benefit from GPU execution because transfer overhead and kernel launch overhead dominate runtime. Larger panels and wider universes are more likely to show GPU advantages, especially for rolling operations and multi-factor computation.

Use this benchmark as an engineering diagnostic. It is separate from backtest quality, factor IC, or trading performance.
