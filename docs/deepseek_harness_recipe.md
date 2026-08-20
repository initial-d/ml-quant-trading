# DeepSeek Harness Recipe

This page shows how to use DeepSeek Harness as a reproducibility assistant for
`ml-quant-trading`. The goal is narrow: run the existing benchmark and
validation workflows, preserve the evidence bundle, and avoid overstating what
the result proves.

DeepSeek Harness is not required by this repository, and this repository does
not ship a DeepSeek plugin. Treat the checkout as an ordinary workspace that an
agent can read, run, and audit.

## Why This Fits

`ml-quant-trading` already has agent-friendly properties:

- deterministic synthetic quick start;
- protocol v1 CPU benchmark with fixed panel, seed, repetitions, and thread
  counts;
- machine-readable benchmark JSON;
- public-data validation docs;
- explicit Reality Check boundaries;
- structured issue templates for reproduction reports.

That makes it a useful workspace for testing whether an agent can reproduce a
quant research workflow without turning benchmark output into trading claims.

## Setup

Open the repository as a DeepSeek Harness workspace, then ask the agent to read:

```text
AGENTS.md
README.md
docs/reality_check.md
docs/benchmarking.md
docs/agent_reproducibility.md
```

Install the development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Run The Benchmark Protocol

Ask the agent to run the fixed CPU benchmark:

```bash
python scripts/benchmark_tensor_factors.py \
  --device cpu --n-dates 750 --n-stocks 1000 --window 20 \
  --repeat 10 --warmup 3 --threads 1 --interop-threads 1 --seed 42 \
  --json-out artifacts/benchmark-v1.json
```

The agent should preserve:

- exact command;
- commit SHA;
- OS, CPU, Python, and PyTorch versions;
- thread settings;
- raw Markdown table;
- `artifacts/benchmark-v1.json`;
- warnings, failures, or unusually unstable cases.

The benchmark is an engineering throughput diagnostic. It is not a trading
performance result and should not be compared as a controlled hardware ranking
unless the environments are controlled.

## Suggested DeepSeek Harness Prompt

```text
Read AGENTS.md, docs/benchmarking.md, and docs/reality_check.md.
Run the protocol v1 CPU benchmark exactly as documented.
Return an evidence bundle with commit SHA, command, environment, raw table,
JSON path, and any unstable cases. Do not describe the result as trading alpha
or as a controlled hardware ranking.
```

## Optional Validation Prompt

```text
Read AGENTS.md, docs/reality_check.md, and docs/public_data_validation.md.
Run only the requested public-data validation workflow. If public data fails or
falls back to synthetic data, report that explicitly. Keep costs, turnover,
data source, and caveats next to the headline result.
```

## Good Output

A useful DeepSeek Harness run produces an issue-ready report:

```text
Repository: initial-d/ml-quant-trading
Commit: <sha>
Workflow: protocol v1 CPU benchmark
Command: <exact command>
Environment: <OS / CPU / Python / PyTorch / CUDA availability>
Artifact: artifacts/benchmark-v1.json
Result: <raw table>
Caveats: <warnings, instability, thermal notes, data limits>
Interpretation: engineering benchmark only; no trading claim
```

Post the result through the benchmark issue or reproduction report template
rather than opening a new benchmark format.

## When A Plugin Would Make Sense

A dedicated DeepSeek Harness plugin is premature today. It would become useful
only if contributors repeatedly use the same agent workflow and need one-click
commands for:

- `mlquant demo`;
- protocol v1 CPU benchmark;
- benchmark JSON validation;
- benchmark-board update drafts;
- public-data validation report packaging.

Until then, a documented workspace recipe is the cleaner integration point.
