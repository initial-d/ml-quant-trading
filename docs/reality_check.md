# Reality Check And Validation Status

This project is best described as a reproducible research-engineering baseline.
It is not a live trading system and does not claim independently verified,
deployable alpha.

## What Is Already Real

- The repository contains a runnable end-to-end factor-to-backtest pipeline.
- The factor engine, masking rules, bias-correction plumbing, ML baselines,
  Markowitz portfolio construction, and backtest metrics are implemented as
  auditable Python modules.
- CI runs linting, tests, and a CLI smoke test across Python 3.9, 3.10, and 3.11.
- The benchmark tooling reports environment details such as CPU, GPU, PyTorch,
  CUDA, OS, and command line.
- Public-data examples use reproducible commands and document expected outputs.

## What Is Still A Smoke Test

- `configs/small.yaml` is a synthetic quick start. It proves that the pipeline
  runs; it does not prove strategy profitability.
- `docs/public_data_mini_reproduction.md` uses a tiny 10-name yfinance universe.
  It validates the public-data API path and factor IC workflow, not a robust
  trading result.
- The first maintainer benchmark is an engineering throughput benchmark for
  tensor factor primitives, not a trading-performance benchmark.

## What Depends On Restricted Data

The paper-shaped result path targets a Wind / Tushare-style A-share panel. The
repository ships loaders and reproduction instructions, but it cannot redistribute
proprietary data. External readers therefore need their own licensed data to
compare the paper-style empirical results quantitatively.

## What Has Industrial Value

The industrial value today is infrastructure value:

- mask-aware cross-sectional factor computation
- handling for halts, missing cells, limit-up / limit-down cases, and tradability
- repeatable ML factor research workflow
- portfolio construction and vectorized backtest organization
- benchmark and validation harnesses that make assumptions explicit

This is useful as a starting point for research teams, students, and engineers
who want a clean baseline to audit or extend. It is not sufficient by itself for
live deployment.

## What Is Required Before Live Use

Before anyone interprets results as deployable, they should add:

- licensed point-in-time data with delisted names and historical universe
  membership
- corporate-action, survivorship-bias, and lookahead-bias controls
- walk-forward validation across market regimes
- transaction-cost, slippage, borrow, liquidity, and capacity models
- risk limits, exposure controls, attribution, and monitoring
- independent reproduction by someone other than the maintainer

## Current Validation Ladder

| Level | Status | Meaning |
|---|---|---|
| CI smoke test | Complete | The package installs and core workflows run across supported Python versions. |
| Synthetic end-to-end pipeline | Complete | The full engineering path runs without private data. |
| Tensor factor benchmark | Started | One maintainer CPU baseline exists; more CPU/GPU reports are needed. |
| Public-data IC note | Started | A small yfinance example validates the public-data factor workflow. |
| Larger public-data walk-forward benchmark | Started | One 100-stock yfinance maintainer run exists; community runs on larger universes are invited. |
| Paper-style empirical reproduction | Data-gated | Requires Wind / Tushare-style data that cannot be redistributed. |
| Industrial production readiness | Not claimed | Needs execution, risk, monitoring, and independent validation layers. |

The goal is to make every rung explicit so users can tell the difference between
engineering reproducibility, public-data diagnostics, paper reproduction, and
production readiness.
