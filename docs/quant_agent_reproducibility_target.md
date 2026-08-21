# Quant Agent Reproducibility Target

`ml-quant-trading` can be used as a small, evidence-bounded target for
LLM-powered quant agents and agent harnesses.

The goal is not to turn this repository into a trading agent. The goal is to
test whether an agent can read a quant research repository, run a fixed
benchmark or validation workflow, preserve the generated artifacts, and avoid
turning runtime or backtest numbers into unsupported trading claims.

## Why Agent Projects Need A Target

LLM trading agents often focus on workflows such as research planning, market
analysis, debate, portfolio decisions, or report writing. Those are useful, but
they still need boring evidence checks:

- Can the agent follow the repository's setup instructions?
- Can it run an exact benchmark command without changing the protocol?
- Can it preserve JSON, Markdown, logs, and environment details?
- Can it report failures or unstable rows instead of smoothing them away?
- Can it separate engineering throughput, public-data validation, and trading
  performance claims?

This repository provides that kind of target surface without requiring broker
credentials, proprietary market data, or a live-trading loop.

## What Agents Can Run

| Workflow | Entry Point | Evidence Produced |
|---|---|---|
| Synthetic smoke test | `mlquant demo` | Markdown and JSON demo reports |
| Protocol v1 CPU benchmark | `make benchmark` or `scripts/benchmark_tensor_factors.py` | `artifacts/benchmark-v1.json` and a result table |
| Public-data mini reproduction | `docs/public_data_mini_reproduction.md` | Factor-IC note with public-data caveats |
| Public validation benchmark | `docs/public_data_validation.md` | Walk-forward metrics, costs, turnover, and caveats |
| DSH-assisted benchmark | `docs/deepseek_harness_recipe.md` | DeepSeek Harness report issue with prompt, transcript, and artifact |

The protocol v1 benchmark is the most compact first target because it uses
deterministic synthetic input and a fixed command:

```bash
python scripts/benchmark_tensor_factors.py \
  --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
  --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

## DeepSeek Harness Plugin

For DeepSeek Harness users, the optional plugin provides DSH-native tools:

```bash
dsh plugin --profile web add github:initial-d/dsh-plugin-mlquant-benchmark
```

The plugin can:

- run the fixed protocol v1 CPU benchmark;
- read the generated benchmark JSON;
- validate required protocol fields and expected benchmark cases;
- draft an issue-ready DeepSeek Harness benchmark report.

Plugin repository:
[`initial-d/dsh-plugin-mlquant-benchmark`](https://github.com/initial-d/dsh-plugin-mlquant-benchmark).

First release:
[`v0.1.0`](https://github.com/initial-d/dsh-plugin-mlquant-benchmark/releases/tag/v0.1.0).

## Suggested Agent Prompt

```text
Read AGENTS.md, docs/benchmarking.md, docs/deepseek_harness_recipe.md,
and docs/reality_check.md. Run the protocol v1 CPU benchmark exactly as
documented. Preserve the command, commit SHA, environment details, raw table,
and artifacts/benchmark-v1.json. If using DSH, validate the benchmark artifact
and draft a DeepSeek Harness benchmark report. Do not describe runtime numbers
or public-data outputs as trading alpha.
```

## Useful Report

A good agent-produced report should include:

- repository commit SHA;
- exact command and prompt;
- agent runtime or harness version;
- OS, CPU, GPU, Python, PyTorch, CUDA availability, and thread settings;
- raw benchmark table and generated JSON artifact;
- warnings, unstable rows, retries, or failures;
- a short interpretation that stays within the evidence boundary.

Submit DSH runs through the
[DeepSeek Harness benchmark report template](https://github.com/initial-d/ml-quant-trading/issues/new?template=deepseek_harness_benchmark.yml).

Seed report:
[`#61`](https://github.com/initial-d/ml-quant-trading/issues/61).

Announcement:
[`DeepSeek Harness benchmark path: run, validate, report`](https://github.com/initial-d/ml-quant-trading/discussions/62).

## Non-Goals

This target is not:

- an autonomous trading desk;
- investment advice;
- a broker or order-placement layer;
- evidence that a synthetic benchmark predicts alpha;
- a leaderboard for uncontrolled hardware comparisons;
- a substitute for point-in-time market data and independent validation.

The valuable question is narrower and easier to audit: can a quant agent
reproduce a documented workflow, preserve the evidence, and state the result
without overstating it?
