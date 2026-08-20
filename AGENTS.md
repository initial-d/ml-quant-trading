# Agent Instructions

This repository is a reproducible quant research baseline, not a live
trading system. Agent work should preserve that boundary.

## Read First

Before making a research, benchmark, data, or reporting change, read:

- `README.md`
- `docs/start_here.md`
- `docs/research_card.md`
- `docs/reality_check.md`
- `docs/benchmarking.md`

For public-data or validation work, also read:

- `docs/public_data_mini_reproduction.md`
- `docs/public_data_validation.md`
- `docs/validation_dashboard.md`

## Evidence Rules

- Do not describe synthetic demos, CPU/GPU benchmarks, or public-data smoke
  tests as deployable alpha or investment evidence.
- Keep synthetic, public-data, provider-backed, and proprietary-data evidence
  separate in code, docs, reports, and issue comments.
- Record the commit SHA, exact command, Python version, PyTorch version, OS,
  CPU/GPU details, data source, cost assumptions, and known limitations for
  any benchmark or validation result.
- Negative, slow, failed, or data-blocked runs are useful evidence when the
  environment and failure mode are documented.
- Do not add live trading, broker execution, account access, credentials, or
  background network services unless a maintainer explicitly asks for that
  scope.
- Do not fetch external market data during routine edits unless the task is
  explicitly a data-validation run.

## Preferred Commands

Install for development:

```bash
python -m pip install -e '.[dev]'
```

Run the deterministic demo:

```bash
mlquant demo
```

Run the protocol v1 CPU benchmark:

```bash
python scripts/benchmark_tensor_factors.py \
  --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
  --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

On systems with GNU Make:

```bash
make benchmark
```

Run the test suite:

```bash
python -m pytest
```

## Reporting Results

When adding or summarizing a new result:

- update the relevant report or documentation page, such as
  `docs/benchmark_board.md` for benchmark results;
- link the issue, pull request, artifact, or discussion where the raw evidence
  can be inspected;
- state whether the result is synthetic, public-data, provider-backed, or
  proprietary-data gated;
- keep caveats close to headline numbers.

If a result cannot be reproduced, preserve the failure details instead of
rewriting the run as a success.
