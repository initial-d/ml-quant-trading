# Docker Setup

The Docker image provides a reproducible CPU development environment for tests,
benchmark scripts, and public-data validation runs. It is intended for local
reproduction and contributor onboarding, not for live trading.

## Build

```bash
docker build -t ml-quant-trading .
```

## Run The Test Suite

```bash
docker run --rm ml-quant-trading make test
```

## Run Lint

```bash
docker run --rm ml-quant-trading make lint
```

## Run The Synthetic Pipeline

```bash
docker run --rm ml-quant-trading make paper CONFIG=configs/small.yaml
```

## Run Benchmarks

```bash
docker run --rm ml-quant-trading make benchmark
```

The default image uses CPU PyTorch from the Python package resolver. For GPU
benchmarks, use a host environment with the correct NVIDIA driver, CUDA runtime,
and PyTorch build, or extend this Dockerfile from an NVIDIA CUDA base image.

## Run Public-Data Validation

```bash
docker run --rm ml-quant-trading \
  python scripts/public_data_validation.py \
    --source synthetic \
    --models equal_weight,momentum_20,alpha101_mean
```

For yfinance runs, the container needs network access:

```bash
docker run --rm ml-quant-trading \
  python scripts/public_data_validation.py \
    --source yfinance \
    --preset us-large-100 \
    --max-tickers 100
```

The generated reports stay inside the container unless you mount a host
directory:

```bash
mkdir -p artifacts/public_data_validation
docker run --rm \
  -v "$PWD/artifacts:/workspace/ml-quant-trading/artifacts" \
  ml-quant-trading \
  python scripts/public_data_validation.py --source synthetic
```

## Dev Container

The VS Code / Codespaces Dev Container reuses the root `Dockerfile` and then
reinstalls the mounted workspace in editable mode. This keeps local Docker and
Dev Container setup aligned.
