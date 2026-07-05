# Contributing

Thanks for your interest in improving `ml-quant-trading`. The project is intended to be research-friendly, fork-friendly, and reproducible.

## Good First Contributions

- Improve or translate documentation.
- Add a small synthetic-data example for an existing module.
- Add tests for edge cases in factor computation, neutralization, bias correction, or backtesting.
- Add benchmark notes for CPU/GPU factor computation.
- Report reproducibility issues with a minimal config and environment details.

## Development Setup

Local Python setup:

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
pip install -e .[dev]
pytest
ruff check .
```

Docker setup:

```bash
docker build -t ml-quant-trading .
docker run --rm ml-quant-trading make test
docker run --rm ml-quant-trading make lint
```

Optional extras:

```bash
pip install -e .[gpu]      # CUDA-enabled PyTorch environment required
pip install -e .[mosek]    # MOSEK license may be required
```

## Pull Request Checklist

- Keep the change focused and explain the research or engineering motivation.
- Add or update tests when behavior changes.
- Update docs when public APIs, configs, or reproducibility steps change.
- Run `pytest` and `ruff check .` before opening the PR.
- Avoid committing private market data, credentials, broker configuration, or large generated artifacts.

## Research Reproducibility

When reporting results, include the config, random seed, data source, date range, benchmark universe, transaction-cost assumptions, and evaluation metrics. Results that depend on proprietary data are welcome, but please provide a synthetic or public-data reproduction path when possible.

## Financial Disclaimer

This project is for research and engineering experimentation. It is not financial advice, investment advice, or a trading recommendation.
