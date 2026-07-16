# Research Card

This card summarizes what `ml-quant-trading` is meant to support today, what it
does not claim, and how to interpret the repository's validation artifacts.

## Intended Use

Use this repository as a reproducible research-engineering baseline for
multi-factor equity experiments:

- compute mask-aware cross-sectional and time-series factors
- train simple ML baselines on factor panels
- run portfolio construction and vectorized backtests
- compare results against equal-weight and momentum baselines
- document public-data, synthetic-data, and benchmark reports in a reviewable way

The project is designed for students, researchers, and engineers who want a
clean starting point for audited experiments rather than a black-box trading
system.

## Non-Goals

This repository is not:

- investment advice
- a production trading engine
- a claim of live deployable alpha
- a substitute for licensed point-in-time market data
- a broker-, exchange-, or order-book-aware execution simulator

Backtest results are research diagnostics. They should not be read as live
performance promises.

## Validation Ladder

| Level | Artifact | What it proves | What it does not prove |
|---|---|---|---|
| Synthetic smoke test | `make paper CONFIG=configs/small.yaml` | The end-to-end pipeline runs on a local machine | Strategy profitability |
| Public IC mini check | `docs/public_data_mini_reproduction.md` | Public-data loading and factor-IC plumbing work | Robust trading performance |
| Public validation run | `docs/public_data_validation.md` | Walk-forward baselines, costs, turnover, and reports are reproducible | Institutional data quality or live execution quality |
| Benchmark report | `docs/benchmark_board.md` | Runtime and environment characteristics are documented | Predictive power |
| Paper-style reproduction | `docs/reproducing_paper.md` | How code maps to the paper-shaped experiment path | Redistribution of restricted data |

## Data Assumptions

The repository supports three practical data paths:

- **Synthetic:** deterministic GBM panels for smoke tests and CI.
- **yfinance:** public US equity and ETF examples, subject to provider changes,
  missing data, and rate limits.
- **Baostock:** A-share loader support for users with a registered account.

The repository does not redistribute market data. Any serious empirical claim
requires point-in-time data, delisting coverage, corporate-action handling,
universe membership controls, and lookahead-bias checks.

## Current Evidence

The current evidence is infrastructure evidence:

- the factor engine, model baselines, portfolio construction, and backtest path
  are implemented as auditable Python modules
- CI runs tests and a CLI smoke test
- public-data notes document exact commands, universes, dates, and outputs
- validation docs include transaction costs, slippage, turnover, bootstrap
  intervals, and failure modes such as yfinance rate limiting

This is enough to support reproducible research iteration. It is not enough to
support a live trading claim.

## Main Risks

- Public-data vendors can revise history, throttle requests, or return partial
  panels.
- Short public-data windows make Sharpe and IC estimates noisy.
- High-turnover strategies can look promising before costs and deteriorate after
  realistic cost assumptions.
- Synthetic data validates the pipeline but contains no intended predictive
  signal.
- A-share research needs careful handling of halts, limit-up/down days,
  survivorship bias, and historical universe membership.

## Contributor Checklist

When sharing a benchmark or validation result, include:

- exact command
- commit SHA
- OS, Python, PyTorch, CPU/GPU, and CUDA availability
- data source, date range, ticker universe, and failed tickers
- cost and slippage assumptions
- whether the result is synthetic, public-data validation, or restricted-data
  reproduction
- any warnings, rate limits, solver issues, or missing artifacts

Prefer small, reproducible reports over broad performance claims.
