# Reproducibility Contract

This page defines the minimum evidence needed for a benchmark, validation run,
or agent-produced report to be useful after the original machine and session are
gone.

The contract is intentionally small. It does not require proprietary data,
perfect numerical identity, or a claim that a strategy is profitable. It asks
contributors to preserve enough context for another reader to tell what was run,
what changed, and whether the evidence supports the conclusion.

## Two Different Claims

Keep these claims separate in every report:

| Claim | Meaning | Good evidence |
|---|---|---|
| Engineering reproducibility | The same code, inputs, and environment produce compatible outputs within a stated tolerance. | Commit, command, environment, dependency lock, input fingerprints, artifacts, checksums, warnings. |
| Research robustness | The conclusion survives another date range, universe, cost assumption, data provider, or machine. | Cross-market runs, cost and slippage stress, bootstrap intervals, failed cases, negative or mixed results. |

A run can pass engineering reproducibility and fail research robustness. That is
still useful evidence. For example, a public-data validation may rerun cleanly
but show that turnover or transaction costs erase the signal.

## Required Fields

Every serious benchmark or validation report should include:

- repository commit SHA and whether the working tree was dirty;
- exact command, config file, and relevant CLI flags;
- OS, Python, PyTorch, CUDA, cuDNN, driver, CPU, GPU, RAM, and thread settings
  where applicable;
- resolved dependency state, preferably a lock file, `pip freeze`, or
  environment export;
- random seeds and deterministic-algorithm settings when the workflow uses
  randomness;
- data source, provider, retrieval timestamp, ticker universe, date range, and
  any failed or skipped symbols;
- a fingerprint of the normalized input panel when possible;
- wall time, peak memory when available, generated artifact paths, and artifact
  checksums;
- expected numerical tolerances for CPU, GPU, BLAS, solver, or provider
  differences;
- warnings, retries, rate limits, local patches, or other caveats.

If a field is not available, say so explicitly. A clear missing field is easier
to audit than a silent omission.

## Minimum Bundle Shape

A future `protocol v1` reproduction bundle should be enough for a maintainer or
external reviewer to run:

```bash
mlquant validate-report path/to/bundle
```

Until that validator exists, use this file structure as the human-readable
target:

```text
report/
  submission.md
  metadata.json
  summary.json
  command.txt
  environment.txt
  pip-freeze.txt
  checksums.txt
```

For benchmark-only submissions, `artifacts/benchmark-v1.json` plus the exact
command and environment table is enough. For public-data validation,
`submission.md`, `metadata.json`, and `summary.json` should travel together.

## Data Fingerprints

Public-data providers can revise history, throttle requests, adjust corporate
actions, or change ticker availability. A command that succeeds six months later
is not always using the same panel.

When redistribution is allowed, preserve the normalized input panel or a compact
sample. When redistribution is not allowed, preserve metadata and hashes:

- provider and endpoint or library version;
- retrieval timestamp and timezone;
- requested ticker universe and resolved ticker universe;
- date range and trading calendar;
- row count, missing-value counts, tradable-mask coverage;
- hash of the normalized prices, returns, factors, labels, or panel file.

The hash does not expose proprietary data, but it lets two evaluators detect
that they did not run the same dataset.

## Numerical Tolerances

Reports should avoid requiring byte-identical floating-point output across
machines. Instead, state tolerances that match the workflow:

- benchmark runtime: compare order of magnitude, mean, standard deviation, and
  caveats rather than exact milliseconds;
- synthetic pipeline: require the full pipeline to finish and preserve summary
  metrics, with small numerical drift allowed across PyTorch and solver
  versions;
- CPU versus GPU factors: compare finite-mask coverage and value differences
  within an explicit absolute or relative tolerance;
- public-data validation: compare the same input fingerprint first, then compare
  metrics with caveats for provider revisions and transaction-cost settings.

When a metric is unstable, report that instability instead of smoothing it away.

## Golden Synthetic Case

The highest-value next protocol milestone is a tiny golden synthetic case with:

- fixed seed and panel shape;
- expected factor summary metrics;
- expected portfolio/backtest summary metrics;
- explicit CPU and CUDA tolerances;
- checksums for generated JSON artifacts;
- one maintainer CPU baseline and at least one independent machine report.

That case should be small enough for CI and for first-time contributors, while
still covering masks, rolling windows, model training, portfolio construction,
cost drag, and report generation.

## Maintainer Review Rules

Maintainers should treat reports as evidence, not as marketing copy:

- accept successful, slow, failed, and caveated runs when the evidence is clear;
- reject reports that omit the commit, command, environment, or data source;
- ask for the normalized-data fingerprint before interpreting public-data
  differences as research differences;
- keep engineering reproducibility separate from performance, alpha, or
  investment claims;
- credit contributors for useful negative results and failure reports.

This contract should evolve only when it makes future reports easier to audit.
