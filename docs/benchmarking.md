# Benchmarking Tensor Factors

This page explains how to benchmark the tensor factor engine on synthetic data. The benchmark is designed to be reproducible without proprietary market data.

## Quick Start

```bash
python -m pip install -e '.[dev]'
python scripts/benchmark_tensor_factors.py \
  --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
  --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

This is the canonical cross-platform **protocol v1** CPU command. It writes a
machine-readable report to `artifacts/benchmark-v1.json`. On systems with GNU
Make, `make benchmark` is an equivalent shortcut.

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
  --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

Protocol v1 fixes the panel, seed, rolling window, repetitions, and PyTorch
thread pools. Submit this CPU result and the generated JSON before optional
variants so reports remain comparable across machines.

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

## Opt-in linear-memory moments

The factor APIs and feature CLI can select centred FP64 prefix-sum mean/std/
cov/corr on CPU or CUDA. Other primitives retain their existing implementations.

```python
from mlquant.features import compute_legacy_set
factors, mask, names = compute_legacy_set(panel, rolling_backend="linear")
```

```bash
mlquant features --config configs/small.yaml --rolling-backend linear
python scripts/benchmark_factor_backends.py \
  --n-dates 500 --n-stocks 200 --warmup 2 --repeat 5 --seed 42 \
  --json-out artifacts/factor-backends.json
```

The config key `rolling_backend: linear` also applies to `mlquant demo`; an
explicit feature CLI option overrides it. `features.pt` records the selection.
Both factor-set APIs default to `unfold` even inside an outer scope. For direct
primitive or registry calls, use `with using_rolling_backend("linear"):` from
`mlquant.features`; scopes restore on exceptions and remain local to threads/
async contexts.

**This is a numerical choice, not a bit-identical acceleration.** Linear moments
exclude nonfinite observations, clamp correlations and use FP64 accumulation.
Constant-window correlations and near-ties can therefore change downstream
ranks materially. Very long, strongly drifting series can lose prefix precision;
MPS lacks FP64 and is rejected. Compare factor values, masks and ranks before
changing a research experiment's backend. The full-engine benchmark reports
all 213 factors' own masks and value errors, sampled rank differences, and a
sampled independent FP64 oracle for correlations of discrete rank inputs. Its
fresh-process timings and RSS snapshots precede saving and diagnostics; neither
kernel speedups nor this synthetic benchmark establish predictive quality.
