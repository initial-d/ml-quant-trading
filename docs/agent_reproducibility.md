# Agent Reproducibility Guide

This repository is designed to be auditable by humans and coding agents. The
goal is not to build a trading agent; the goal is to make benchmark and
validation workflows easier to rerun, inspect, and report without blurring the
line between engineering evidence and investment claims.

The same workflow can be used from Codex, DeepSeek Harness, Claude Code, or any
agent runtime that can read files, run shell commands, and preserve artifacts.
No agent runtime is required by the package.

## Agent Scope

Good agent tasks:

- run the deterministic demo and report whether it completed;
- run the protocol v1 tensor benchmark on a new machine;
- check that benchmark reports include the required environment fields;
- compare a generated validation artifact with the documented expectation;
- improve docs, caveats, and reproduction instructions;
- turn a failed public-data run into a useful failure report.

Out of scope unless explicitly requested:

- live trading, order placement, broker integration, or credential handling;
- automatic strategy promotion;
- hyperparameter search presented as validation;
- claims that a synthetic or public-data smoke test proves deployable alpha.

## Minimal Agent Workflow

1. Read `AGENTS.md`, `README.md`, `docs/reality_check.md`, and the page for the
   specific workflow being run.
2. Install the repository in development mode:

   ```bash
   python -m pip install -e '.[dev]'
   ```

3. Run the smallest relevant command first:

   ```bash
   mlquant demo
   ```

4. For a benchmark contribution, run the protocol v1 CPU command:

   ```bash
   python scripts/benchmark_tensor_factors.py \
     --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
     --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
     --json-out artifacts/benchmark-v1.json
   ```

5. Preserve the raw output, generated JSON, command, commit SHA, and environment
   details before summarizing the result.

## Evidence Bundle

A useful agent-produced benchmark or validation report should include:

- repository commit SHA;
- exact command;
- OS, CPU, GPU, Python, PyTorch, CUDA availability, and thread settings;
- data source and whether it is synthetic, public, provider-backed, or
  proprietary-data gated;
- generated artifact paths and checksums when available;
- runtime warnings, failures, skipped rows, or data-quality limitations;
- a short interpretation that does not exceed what the evidence supports.

For benchmark results, update or reference `docs/benchmark_board.md`. For
public-data validation, keep the caveats near the result and link the generated
artifact or issue report.

## Prompt Patterns

Benchmark audit:

```text
Read AGENTS.md and docs/benchmarking.md. Run the protocol v1 CPU benchmark.
Summarize the environment, command, raw table, JSON path, and any instability.
Do not compare this result as a controlled hardware ranking.
```

Validation audit:

```text
Read AGENTS.md, docs/reality_check.md, and docs/public_data_validation.md.
Run only the requested validation workflow. Report the data source, costs,
turnover, failures, and whether the output is synthetic fallback, public data,
or provider-backed evidence.
```

Documentation audit:

```text
Read AGENTS.md and docs/reality_check.md. Check whether a report separates
synthetic, public-data, benchmark, and paper-style claims. Move caveats closer
to headline numbers when needed.
```

## DeepSeek Harness Notes

DeepSeek Harness users can treat this repository as an ordinary workspace:
open the checkout, read `AGENTS.md`, run the commands above, and attach the
generated artifacts to an issue or pull request. A dedicated plugin is not
needed for the current workflows. If a plugin is added later, it should be a
thin wrapper around existing commands such as `mlquant demo` and
`scripts/benchmark_tensor_factors.py`, not a separate research path.

## Boundary

Agent-readiness is an engineering property. It means the repository exposes
clear commands, artifacts, and reporting rules. It does not mean an agent has
validated profitability, execution quality, market-data rights, or production
readiness.
