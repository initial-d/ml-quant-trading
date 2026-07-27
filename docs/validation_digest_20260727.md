# Validation Digest - 2026-07-27

This digest summarizes the follow-up validation and discovery surface for
`v0.2.1`. It extends the `v0.2.0` validation digest with entrypoint fixes,
visibility tracking, and lessons from external technical discussions.

It is not a trading-performance claim.

## Scope

The digest covers changes and operating notes after `v0.2.0`:

- Colab Baostock demo repository URL fix
- Hugging Face paper entrypoint confirmation
- public discovery and traffic pulse
- external validation discussions around replay safety, rate limits, and
  backtest contracts
- current contributor funnel for benchmark and public-data reports

## Current Public Reports

| Report | Source | Universe | Purpose | Status |
|---|---|---|---|---|
| [`validation_synthetic_20260716.md`](validation_synthetic_20260716.md) | Synthetic | Generated GBM panel | Cross-platform CLI and metric reproducibility | Merged |
| [`validation_baostock_20260716.md`](validation_baostock_20260716.md) | Baostock | A-share public data | Public-data validation on a China-market source | Merged |
| [`public_data_mini_reproduction.md`](public_data_mini_reproduction.md) | yfinance | ETF mini example | Factor IC and public-data smoke reproduction | Merged |
| [Issue #22 blocker report](https://github.com/initial-d/ml-quant-trading/issues/22#issuecomment-4989435338) | yfinance | ETF-50 attempt | Documents HTTP 429 rate limiting | Documented |

## Entrypoint Fix

The Baostock Colab demo now clones the canonical repository:

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
```

This matters because Colab is a high-intent path: a user who opens the notebook
is already trying to run the project. Sending that runtime to an old repository
URL weakens reproducibility and makes traffic harder to attribute.

## Outreach-Derived Validation Themes

Recent external technical discussions repeatedly pointed to the same validation
contracts:

- separate valuation prices from execution prices when bars are missing;
- keep rebalance-calendar semantics explicit;
- report data-source blockers instead of producing empty benchmark claims;
- record data vintage, source, calendar, and universe assumptions;
- distinguish synthetic, fixture, public-data, and production-data evidence;
- keep paper/live execution claims separate from research backtests;
- require post-solve or post-backtest audit artifacts for optimizers.

These themes match the project's current stance: the repository should be easy
to run and audit before it is impressive.

## Current Public Discovery Pulse

Recorded on 2026-07-27:

- Repository pulse during outreach: 55 stars, 28 forks, 4 watchers.
- GitHub traffic snapshot in `docs/visibility_status.md`: 1,206 views and 246
  unique visitors over GitHub's rolling 14-day window.
- Clone snapshot: 465 total clones and 249 unique cloners over the same window.
- High-interest paths include the Chinese README, factor handbook, issue #22,
  discussion #13, and contributor PRs.

## Contributor Gaps

The most useful next reports are still:

- CUDA GPU tensor-factor benchmark.
- Linux CPU benchmark on a common cloud instance.
- Apple Silicon benchmark with a larger panel.
- A rerun of ETF-50 validation after yfinance rate limits clear.
- Another public-data case study with clearly documented data provenance.
- A clean Colab/Baostock run report from a fresh runtime.

## Reproduction Entry Points

```bash
python scripts/benchmark_tensor_factors.py --device auto
```

```bash
python scripts/public_data_validation.py \
  --source synthetic \
  --models equal_weight,momentum_20,alpha101_mean \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 100
```

```bash
python scripts/public_data_validation.py \
  --source baostock \
  --preset cn-large-25 \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --models equal_weight,momentum_20,alpha101_mean \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 100
```

Report results in [Discussions #13](https://github.com/initial-d/ml-quant-trading/discussions/13)
or the pairing issue [#22](https://github.com/initial-d/ml-quant-trading/issues/22).
