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

## Contributor Workflow

Use issues as the source of truth for upcoming work. The roadmap in
`docs/roadmap.md` is intentionally lightweight so contributors can see the next
useful tasks without needing access to a private project board.

Recommended flow:

1. Open or claim an issue before doing a larger change.
2. Keep pull requests small enough to review in one sitting.
3. Link the PR to the issue it addresses.
4. Include the command output or environment details for benchmark and
   reproducibility changes.
5. Prefer human review and CI signals before introducing automated review tools.

Automated review assistance may be used later for repetitive checks, but it
should not replace clear tests, reproducible examples, or maintainer judgment.

## Paired Contributions

Paired contributions are welcome when the work can be split clearly. Good
pairing candidates include public-data validation reports, benchmark runs on
different hardware, focused docs improvements, and small tests for edge cases.

Use the pairing request issue template when you want to pair on a task. A good
request should include the proposed task, the split of work, reproduction
commands, and the expected PR output.

Only add `Co-authored-by:` trailers when each named person materially
contributed to the merged PR. Useful contributions include writing code or docs,
running and interpreting a reproducible benchmark, debugging a failing test, or
reviewing generated reports closely enough to change the final result.

## Research Reproducibility

When reporting results, include the config, random seed, data source, date range, benchmark universe, transaction-cost assumptions, and evaluation metrics. Results that depend on proprietary data are welcome, but please provide a synthetic or public-data reproduction path when possible.

## Financial Disclaimer

This project is for research and engineering experimentation. It is not financial advice, investment advice, or a trading recommendation.
