# Validation Digest - 2026-07-20

This digest summarizes the public validation surface available for the `v0.2.0`
release. It is a release note for reproducibility and contributor planning, not
a trading-performance claim.

## Scope

The digest covers reports and blocker notes merged or documented after the
`v0.1.0` public baseline:

- synthetic validation baseline on Windows CPU
- Baostock A-share validation report
- yfinance ETF-50 rate-limit failure mode
- public-data validation harness improvements
- benchmark and validation contribution workflow

## Current Public Reports

| Report | Source | Universe | Purpose | Status |
|---|---|---|---|---|
| [`validation_synthetic_20260716.md`](validation_synthetic_20260716.md) | Synthetic | Generated GBM panel | Cross-platform CLI and metric reproducibility | Merged |
| [`validation_baostock_20260716.md`](validation_baostock_20260716.md) | Baostock | A-share public data | Public-data validation on a China-market source | Merged |
| [`public_data_mini_reproduction.md`](public_data_mini_reproduction.md) | yfinance | ETF mini example | Factor IC and public-data smoke reproduction | Merged |
| [Issue #22 blocker report](https://github.com/initial-d/ml-quant-trading/issues/22#issuecomment-4989435338) | yfinance | ETF-50 attempt | Documents HTTP 429 rate limiting | Documented |

## Main Takeaways

- The synthetic source is useful for checking plumbing, report generation,
  bootstrap paths, cost grids, and Windows CPU reproducibility. It is not
  expected to contain predictive signal.
- The Baostock path gives Chinese contributors a public-data route that does not
  depend on Yahoo Finance access.
- yfinance can fail with several different-looking errors when Yahoo Finance
  rate limits a network. `JSONDecodeError`, `YFTzMissingError`, and
  `YFRateLimitError` can all point to the same HTTP 429 root cause.
- Turnover and transaction costs dominate interpretation. High-turnover signals
  can deteriorate sharply under realistic cost assumptions.
- The current public evidence should be read as validation and reproducibility
  diagnostics, not as a deployable trading strategy.

## Contributor Gaps

The next useful reports are:

- CUDA GPU tensor-factor benchmark.
- Linux CPU benchmark on a common cloud instance.
- Apple Silicon benchmark with a larger panel.
- A rerun of ETF-50 validation after yfinance rate limits clear.
- Another public-data case study with clearly documented data provenance.

## Reproduction Entry Points

Use these entry points when adding a report:

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

