# AkShare CSI 300 Daily 213-Factor Validation - 2026-07-29

> **Units note.** `cost_drag` in the tables below is *cumulative over the run*, not annualised — unlike `ann_return`, `gross_ann_return` and `ann_vol` beside it. Divide by the number of years before comparing it with a per-year figure, and do not compare it across reports of different lengths. The field is named `cost_drag_cumulative` in new output; see [the metric glossary](public_data_validation.md#metric-glossary).


This report records a paper-style public-data approximation on the AkShare CSI
300 universe. Unlike the lighter `v0.2.2` baseline report, this run uses the
full 213-factor library and daily portfolio evaluation.

It is still not an exact paper reproduction: the proprietary paper data is not
redistributed, the CSI 300 universe is current membership resolved at runtime,
and the public AkShare data path lacks industry, size, liquidity, and execution
metadata. Treat this as public validation evidence, not investment advice.

## Pipeline

The run executes:

1. Resolve current CSI 300 constituents from AkShare/CSI public endpoints.
2. Download daily A-share OHLCV data through AkShare.
3. Compute the full modern factor set: 213 factors.
4. Train a walk-forward MLP on the selected factor matrix.
5. Build daily factor and model portfolios.
6. Apply a daily evaluated buffered top-quantile factor rule to control churn.
7. Re-score all portfolios under transaction costs and cost-sensitivity grids.

The buffered rule is still daily: it evaluates scores every trading day, but it
does not replace a holding until it falls out of a wider exit band. This is meant
to distinguish factor usefulness from unnecessary rank noise and turnover.
No weekly rebalance shortcut is used in the main comparison, because weekly
rebalancing can hide exactly the turnover problem this validation is meant to
measure.

## Command

```bash
python scripts/akshare_csi300_full_pipeline.py \
  --factor-set all \
  --max-tickers 300 \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --train-window 504 \
  --test-window 63 \
  --step 63 \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32 \
  --factor-ic-window 252 \
  --covariance-window 126 \
  --optimizer-candidates 60 \
  --optimizer-weight-cap 0.05 \
  --optimizer-risk-aversion 1.0 \
  --rebalance-step 1 \
  --bootstrap-samples 100 \
  --bootstrap-block-size 20 \
  --cost-grid-bps 0,7,15,30 \
  --output-dir artifacts/akshare_csi300_daily_all_factor_buffered_20260729
```

## Data And Environment

| Field | Value |
|---|---|
| Source | AkShare |
| Universe | Current CSI 300 constituents |
| Factor set | `all` |
| Factor count | 213 |
| Returned panel | 969 dates x 300 stocks |
| Returned date range | 2021-01-04 to 2024-12-31 |
| Tradable ratio | 0.9625 |
| Stocks with no data | `001280`, `600930` |
| Python | 3.12.13 |
| PyTorch | 2.13.0+cpu |
| Platform | Windows-11-10.0.26200-SP0 |

## Main Results

Effective cost is 7 bps.

| strategy | ann_return | ann_vol | sharpe | max_dd | turnover | cost_drag | gross_ann_return | gross_sharpe | alpha_ann | final_equity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_weight_daily | 0.1775 | 0.2102 | 0.8821 | 0.2576 | 0.0010 | 0.0013 | 0.1779 | 0.8838 |  | 1.8744 |
| factor_mean_daily | 0.1580 | 0.2552 | 0.7012 | 0.3251 | 0.3627 | 0.4916 | 0.3157 | 1.2012 | -0.0185 | 1.7579 |
| factor_mean_buffered_daily | 0.2220 | 0.2526 | 0.9190 | 0.2776 | 0.1397 | 0.1894 | 0.2836 | 1.1134 | 0.0359 | 2.1616 |
| factor_ic_weighted_daily | -0.0156 | 0.2070 | 0.0277 | 0.3206 | 0.3715 | 0.5035 | 0.1220 | 0.6591 | -0.1785 | 0.9413 |
| factor_mean_optimized_daily | 0.1580 | 0.2826 | 0.6584 | 0.3557 | 0.3945 | 0.5346 | 0.3306 | 1.1498 | -0.0172 | 1.7580 |
| factor_ic_optimized_daily | 0.0608 | 0.3075 | 0.3450 | 0.4324 | 0.2932 | 0.3973 | 0.1768 | 0.6846 | -0.0959 | 1.2547 |

The MLP rows were retained in the generated machine-readable report but are not
used as evidence here: in this all-factor public run the current compact MLP path
did not produce tradable non-zero portfolios. That is a model-path issue to
stabilize later, not a reason to hide the factor result.

## Interpretation

- The full 213-factor mean signal has strong gross evidence: `gross_ann_return`
  is `0.3157` versus `0.1779` for equal weight.
- The naive daily top-quantile factor portfolio almost loses that edge to
  turnover: net annual return falls to `0.1580` because turnover is `0.3627`.
- The buffered daily factor rule keeps the daily evaluation cadence while
  reducing turnover to `0.1397`. Net annual return rises to `0.2220`, Sharpe to
  `0.9190`, and final equity to `2.1616`.
- Under the same 7 bps cost assumption, `factor_mean_buffered_daily` beats
  `equal_weight_daily` on annual return, Sharpe, and final equity.
- The result supports the framework story: factor construction alone is not
  enough; the portfolio layer must control churn so that gross factor edge
  survives transaction costs.
- This is why the daily comparison is intentionally strict: rough daily
  selection is penalized by turnover and costs, while a better daily holding
  rule has to preserve the signal without pretending execution is free.

## Audit

The generated `summary.json` was audited with
`scripts/audit_validation_report.py`. The only warning was expected for this
current-constituent universe:

- `panel.stocks_with_no_data`: `001280`, `600930`

## Limitations

- Current CSI 300 membership is used as the public universe; this is not
  historical point-in-time index membership.
- AkShare public endpoints can change, throttle, or revise records.
- The public run does not include industry, size, liquidity, beta, or production
  execution constraints.
- The buffered rule is a transparent turnover-control layer, not a proof that
  this exact parameterization will generalize.
- The compact MLP path needs follow-up stabilization for the 213-factor public
  matrix.
