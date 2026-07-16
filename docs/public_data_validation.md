# Public-Data Validation Benchmark

This page describes the larger public-data validation path. It is designed to
move beyond the tiny IC smoke test while staying honest about what public data
can and cannot prove.

The validation script compares:

- equal-weight baseline
- 20-day momentum baseline
- Alpha101 subset score baseline
- walk-forward MLP baseline on Alpha101 features
- walk-forward Transformer baseline on Alpha101 features

It reports annual return, volatility, Sharpe, max drawdown, turnover, cost drag,
gross return, optional bootstrap uncertainty intervals, and active metrics
versus equal weight. Transaction costs and slippage are combined into the
effective cost charged on weight changes.

## Quick Synthetic Check

Use this first to verify that the validation harness works without downloading
data:

```bash
python scripts/public_data_validation.py \
  --source synthetic \
  --models equal_weight,momentum_20,alpha101_mean
```

The command writes:

```text
artifacts/public_data_validation/summary.md
artifacts/public_data_validation/summary.csv
artifacts/public_data_validation/summary.json
artifacts/public_data_validation/metadata.json
artifacts/public_data_validation/submission.md
```

Use `submission.md` when opening a GitHub issue. It contains the command,
environment, data coverage, result table, and interpretation notes in one
copy-ready report.

## Larger Public-Data Run

The default public run uses a 100-name US large-cap preset from yfinance:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --max-tickers 100 \
  --start 2021-01-01 \
  --end 2025-01-01
```

For an ETF universe:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset etf-50 \
  --max-tickers 50
```

For a mixed stock/ETF universe:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset mixed-150 \
  --max-tickers 150
```

You can also pass your own comma-separated ticker list:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --tickers AAPL,MSFT,NVDA,SPY,QQQ,IWM
```

## Maintainer 100-Stock Reference Run

This reference run was executed on the built-in `us-large-100` yfinance preset.
It is included to make the benchmark concrete and comparable. It is not evidence
of deployable alpha.

Command:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --max-tickers 100 \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --models equal_weight,momentum_20,alpha101_mean,mlp_alpha101,transformer_alpha101 \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32
```

Environment:

| Field | Value |
|---|---|
| Data source | yfinance |
| Universe | 100 US large-cap tickers |
| Returned panel | 1005 dates x 100 stocks |
| Returned date range | 2021-01-04 to 2024-12-31 |
| Costs + slippage | 5.00 + 2.00 bps |
| Walk-forward train/test/step | 504 / 63 / 63 trading days |
| Python | 3.9.6 |
| Platform | macOS 26.5.1 arm64 |
| PyTorch | 2.8.0 |

Results:

| Strategy | Ann Return | Ann Vol | Sharpe | Max DD | Turnover | Cost Drag | Gross Ann Return | Gross Sharpe | Info Ratio vs EW | Active Ann Return | Final Equity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equal_weight | 0.1818 | 0.1531 | 1.1681 | 0.1997 | 0.0005 | 0.0007 | 0.1820 | 1.1693 |  |  | 1.9470 |
| momentum_20 | 0.1072 | 0.1719 | 0.6782 | 0.2781 | 0.1675 | 0.2355 | 0.1745 | 1.0232 | -0.6280 | -0.0650 | 1.5008 |
| alpha101_mean | -0.0529 | 0.1608 | -0.2577 | 0.3043 | 0.5401 | 0.7592 | 0.1456 | 0.9266 | -3.4446 | -0.1994 | 0.8050 |
| mlp_alpha101 | 0.0323 | 0.0914 | 0.3934 | 0.1729 | 0.2915 | 0.4097 | 0.1439 | 1.5164 | -1.0431 | -0.1413 | 1.1351 |
| transformer_alpha101 | -0.0157 | 0.0932 | -0.1227 | 0.1603 | 0.2921 | 0.4105 | 0.0910 | 0.9793 | -1.3872 | -0.1811 | 0.9390 |

In this reference run, equal weight is the strongest net baseline. That is useful
negative evidence: the current public-data validation harness is not being used
to claim that the ML baselines already beat simple portfolios.

## Walk-Forward Design

By default, the ML baselines train on roughly two years of daily observations
and predict the next quarter:

```text
train window: 504 trading days
test window:   63 trading days
step:          63 trading days
```

Each split trains only on dates before the test window. The script then turns
predictions into long-only top-quantile portfolios and runs the same vectorized
backtest path as the rest of the repository.

## Cost And Slippage Analysis

The script exposes both transaction costs and slippage:

```bash
python scripts/public_data_validation.py \
  --costs-bps 5 \
  --slippage-bps 2
```

The effective cost is:

```text
effective_cost_bps = costs_bps + slippage_bps
```

To check whether the same strategies are fragile to different effective cost
assumptions, add a cost grid:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --cost-grid-bps 0,7,15,30
```

This does not retrain the ML models or reselect parameters. It re-scores the
same generated portfolio weights under each effective cost scenario and writes:

```text
artifacts/public_data_validation/cost_sensitivity.md
artifacts/public_data_validation/cost_sensitivity.csv
artifacts/public_data_validation/cost_sensitivity.json
```

The same rows are also embedded in `summary.json` under `cost_sensitivity` so
maintainers can audit or aggregate them later. Use this table to spot strategies
whose net results depend on one narrow cost assumption.

This is intentionally simple. It is useful for sensitivity checks, but it is not
a substitute for a broker-, exchange-, order-size-, and liquidity-aware execution
model.

## Bootstrap Uncertainty Intervals

Point estimates can be noisy, especially for short public-data windows. Add a
block bootstrap to report 95% intervals for annualized return and Sharpe:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --bootstrap-samples 500 \
  --bootstrap-block-size 20
```

The bootstrap samples contiguous return blocks instead of individual days so the
resampled paths preserve some short-horizon autocorrelation. The output adds:

```text
ann_return_ci_low
ann_return_ci_high
sharpe_ci_low
sharpe_ci_high
```

These intervals are diagnostics, not formal proof of statistical significance.
They are most useful for spotting fragile point estimates and for comparing
community reports that use the same universe and date range.

## Interpreting Results

This benchmark is stronger than the tiny public-data IC note because it includes
a larger universe, walk-forward prediction, portfolio construction, turnover,
drawdown, costs, uncertainty intervals, and baseline comparisons.

It still should not be read as proof of deployable alpha:

- yfinance data is convenient but not institutional-grade research data.
- Current presets are survivorship-biased because membership is fixed today.
- Slippage is modeled as a simple basis-point charge, not a market-impact model.
- The ML baselines are intentionally small and should be treated as reference
  comparisons, not tuned production models.
- Public benchmark results can drift when yfinance revises data.

The most valuable community contributions are additional benchmark reports with
clear universe definitions, dependency versions, hardware details, and the exact
command used.

## Sharing A Community Report

1. Run the validation command.
2. Audit `artifacts/public_data_validation/summary.json`.
3. Open `artifacts/public_data_validation/submission.md`.
4. Paste it into the `Public-data validation report` issue template.
5. Attach or paste `metadata.json` when the run uses a custom universe.

The generated `summary.json` is intended for future aggregation scripts and
leaderboards. It includes metadata, data coverage, and strategy metrics in a
machine-readable format.

## Troubleshooting yfinance Rate Limiting

Yahoo Finance may throttle requests from certain networks or IP ranges. When
this happens, the validation script fails during the data-download stage before
any backtest or report is generated. If no public-data result was produced,
report the blocker first instead of opening a validation PR.

### How to recognise rate limiting

Rate limiting can surface in several misleading forms. The underlying cause is
usually an HTTP 429 (`Edge: Too Many Requests`) response from Yahoo, but
yfinance may report it as one or more of:

- `YFRateLimitError: Too Many Requests. Rate limited. Try after a while.`
- `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` — the response
  body is not valid JSON because Yahoo returned an error page
- `YFTzMissingError: possibly delisted; No timezone found` — tickers that
  normally download fine suddenly appear "delisted"
- request timeouts or connection errors around the same run

Dozens of tickers failing at once — especially a preset like `etf-50` or
`us-large-100` that has worked in prior runs — is a strong signal that the
problem is network-side, not ticker-side.

### What to do

1. **Stop.** Do not retry the full 50-ticker run immediately — repeated
   requests may extend the rate-limit window.
2. **Report the blocker** in the relevant issue rather than opening a PR. An
   incomplete validation run is still useful as a blocker report, but it should
   not be submitted as a benchmark result. Mention the exact errors, the
   date/time, yfinance version, and any non-sensitive network context you are
   comfortable sharing.
3. **Wait and retry with a smoke test** before scaling back up:

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --tickers SPY,QQQ,TLT,GLD,AGG \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --models equal_weight,momentum_20 \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 50 \
  --bootstrap-block-size 20
```

4. If the 5-ticker smoke test succeeds, scale to 20 and then to the full
   preset. If it still returns 429, wait longer or retry from a different
   network.
5. Once the full run completes, open a PR with the generated `submission.md`
   report.

### Fallback: synthetic validation

If yfinance remains unavailable, the synthetic data path is always usable as a
reproducibility check:

```bash
python scripts/public_data_validation.py \
  --source synthetic \
  --models equal_weight,momentum_20,alpha101_mean
```

A synthetic run cannot replace a public-data report, but it confirms that the
validation harness itself is working on your platform.

## Auditing Reports

Before sharing a validation run, audit the generated `summary.json`:

```bash
python scripts/audit_validation_report.py \
  artifacts/public_data_validation/summary.json \
  --output-md artifacts/public_data_validation/audit.md \
  --output-json artifacts/public_data_validation/audit.json
```

The audit checks for missing metadata, missing result rows, low data coverage,
tickers with no data, missing equal-weight baseline, non-finite metrics, negative
cost settings, unusual turnover/drawdown, and non-positive final equity. It is a
quality gate for reproducibility reports, not a judgement about whether a
strategy is good.

For the default artifact path, the Make target is:

```bash
make audit-validation
```

## Aggregating Reports

Maintainers can aggregate one or more validation result directories:

```bash
python scripts/aggregate_validation_reports.py \
  artifacts/public_data_validation \
  --output-md artifacts/public_data_validation/leaderboard.md \
  --output-csv artifacts/public_data_validation/leaderboard.csv
```

The script scans for `summary.json` files and writes a compact leaderboard with
source, preset, date range, panel size, strategy, return, Sharpe, drawdown,
turnover, data coverage, and environment columns.

For the default artifact path, the Make target is:

```bash
make aggregate-validation
```
