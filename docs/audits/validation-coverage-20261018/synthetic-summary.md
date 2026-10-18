# AkShare CSI 300 Full-Pipeline Validation Summary

This report is a synthetic engineering diagnostic, not a trading claim.

| Field | Value |
|---|---|
| Source | synthetic |
| Preset | custom |
| Dates x stocks | 120 x 20 |
| Date range | 2020-01-02T00:00:00.000000 to 2020-06-17T00:00:00.000000 |
| Tradable ratio | 1.0000 |
| Model | mlp_alpha101 implementation trained on selected factor matrix |
| Factor set | all |
| Features | 213 factors |
| Neutralization | cross_sectional_zscore_on_model_scores |
| Factor IC window | 30 trading days |
| Portfolio optimizer | LedoitWolf diagonal risk-scaled long-only mean-variance weights on top scored candidates |
| Optimizer candidates / cap | 10 / 0.1000 |
| Rebalance | daily |
| Costs + slippage | 5.00 + 2.00 bps |
| Python / PyTorch | 3.12.14 / 2.14.1 |

## Execution Coverage

Factor-valid signal cells: 1040 / 2400; historical label cells: 1020.

MLP splits trained / planned: 0 / 3; finite prediction cells: 0. Per-split sample counts and skip reasons are in `summary.json`.

| Strategy | Invested dates | Cash dates | P&L exposure dates | First / last P&L exposure |
|---|---:|---:|---:|---|
| equal_weight_daily | 119 | 1 | 118 | 2020-01-06T00:00:00.000000 / 2020-06-17T00:00:00.000000 |
| factor_mean_daily | 52 | 68 | 51 | 2020-04-08T00:00:00.000000 / 2020-06-17T00:00:00.000000 |
| factor_mean_buffered_daily | 52 | 68 | 51 | 2020-04-08T00:00:00.000000 / 2020-06-17T00:00:00.000000 |
| factor_ic_weighted_daily | 0 | 120 | 0 | None / None |
| factor_mean_optimized_daily | 52 | 68 | 51 | 2020-04-08T00:00:00.000000 / 2020-06-17T00:00:00.000000 |
| factor_ic_optimized_daily | 0 | 120 | 0 | None / None |
| mlp_top20_raw_daily | 0 | 120 | 0 | None / None |
| mlp_top20_neutralized_daily | 0 | 120 | 0 | None / None |
| mlp_optimized_daily | 0 | 120 | 0 | None / None |

- **Coverage warning:** No walk-forward split was trained; zero-return MLP rows are not evidence of model performance.
- **Coverage warning:** The model produced no finite predictions.
- **Coverage warning:** factor_ic_weighted_daily: no invested dates; an all-cash row is not evidence of model performance.
- **Coverage warning:** factor_ic_optimized_daily: no invested dates; an all-cash row is not evidence of model performance.
- **Coverage warning:** mlp_top20_raw_daily: no invested dates; an all-cash row is not evidence of model performance.
- **Coverage warning:** mlp_top20_neutralized_daily: no invested dates; an all-cash row is not evidence of model performance.
- **Coverage warning:** mlp_optimized_daily: no invested dates; an all-cash row is not evidence of model performance.
- **Coverage warning:** Strategies have different P&L exposure dates. Full-panel metrics include warm-up cash and are not matched-window comparisons.

Full-panel metrics include warm-up cash. P&L exposure uses the previous date's weights; cash dates can still incur exit costs.

| strategy | ann_return | ann_vol | sharpe | max_dd | turnover | cost_drag_cumulative | gross_ann_return | gross_sharpe | info_ratio | alpha_ann | final_equity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_weight_daily | -0.3687 | 0.1715 | -2.5950 | 0.2103 | 0.0042 | 0.0007 | -0.3678 | -2.5862 |  |  | 0.8033 |
| factor_mean_daily | 0.1034 | 0.1021 | 1.0148 | 0.0464 | 0.2143 | 0.0357 | 0.1893 | 1.7512 | 3.5897 | 0.7099 | 1.0480 |
| factor_mean_buffered_daily | 0.0223 | 0.1125 | 0.2515 | 0.0486 | 0.1050 | 0.0175 | 0.0606 | 0.5815 | 3.0690 | 0.5858 | 1.0106 |
| factor_ic_weighted_daily | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5950 | 0.5374 | 1.0000 |
| factor_mean_optimized_daily | -0.0722 | 0.1008 | -0.6936 | 0.0662 | 0.1471 | 0.0245 | -0.0231 | -0.1837 | 2.6235 | 0.4401 | 0.9650 |
| factor_ic_optimized_daily | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5950 | 0.5374 | 1.0000 |
| mlp_top20_raw_daily | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5950 | 0.5374 | 1.0000 |
| mlp_top20_neutralized_daily | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5950 | 0.5374 | 1.0000 |
| mlp_optimized_daily | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.5950 | 0.5374 | 1.0000 |

## Cost Sensitivity

| cost_scenario | strategy | effective_costs_bps | ann_return | sharpe | max_dd | turnover | cost_drag_cumulative | final_equity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00_bps | equal_weight_daily | 0 | -0.3678 | -2.5862 | 0.2097 | 0.0042 | 0.0000 | 0.8038 |
| 0.00_bps | factor_mean_daily | 0 | 0.1893 | 1.7512 | 0.0427 | 0.2143 | 0.0000 | 1.0861 |
| 0.00_bps | factor_mean_buffered_daily | 0 | 0.0606 | 0.5815 | 0.0432 | 0.1050 | 0.0000 | 1.0284 |
| 0.00_bps | factor_ic_weighted_daily | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 0.00_bps | factor_mean_optimized_daily | 0 | -0.0231 | -0.1837 | 0.0555 | 0.1471 | 0.0000 | 0.9889 |
| 0.00_bps | factor_ic_optimized_daily | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 0.00_bps | mlp_top20_raw_daily | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 0.00_bps | mlp_top20_neutralized_daily | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 0.00_bps | mlp_optimized_daily | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7.00_bps | equal_weight_daily | 7 | -0.3687 | -2.5950 | 0.2103 | 0.0042 | 0.0007 | 0.8033 |
| 7.00_bps | factor_mean_daily | 7 | 0.1034 | 1.0148 | 0.0464 | 0.2143 | 0.0357 | 1.0480 |
| 7.00_bps | factor_mean_buffered_daily | 7 | 0.0223 | 0.2515 | 0.0486 | 0.1050 | 0.0175 | 1.0106 |
| 7.00_bps | factor_ic_weighted_daily | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7.00_bps | factor_mean_optimized_daily | 7 | -0.0722 | -0.6936 | 0.0662 | 0.1471 | 0.0245 | 0.9650 |
| 7.00_bps | factor_ic_optimized_daily | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7.00_bps | mlp_top20_raw_daily | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7.00_bps | mlp_top20_neutralized_daily | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 7.00_bps | mlp_optimized_daily | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15.00_bps | equal_weight_daily | 15 | -0.3698 | -2.6049 | 0.2109 | 0.0042 | 0.0015 | 0.8026 |
| 15.00_bps | factor_mean_daily | 15 | 0.0128 | 0.1745 | 0.0614 | 0.2143 | 0.0765 | 1.0061 |
| 15.00_bps | factor_mean_buffered_daily | 15 | -0.0199 | -0.1209 | 0.0547 | 0.1050 | 0.0375 | 0.9905 |
| 15.00_bps | factor_ic_weighted_daily | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15.00_bps | factor_mean_optimized_daily | 15 | -0.1253 | -1.2683 | 0.0851 | 0.1471 | 0.0525 | 0.9383 |
| 15.00_bps | factor_ic_optimized_daily | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15.00_bps | mlp_top20_raw_daily | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15.00_bps | mlp_top20_neutralized_daily | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 15.00_bps | mlp_optimized_daily | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 30.00_bps | equal_weight_daily | 30 | -0.3718 | -2.6231 | 0.2121 | 0.0042 | 0.0030 | 0.8014 |
| 30.00_bps | factor_mean_daily | 30 | -0.1378 | -1.3575 | 0.1035 | 0.2143 | 0.1530 | 0.9318 |
| 30.00_bps | factor_mean_buffered_daily | 30 | -0.0943 | -0.8011 | 0.0780 | 0.1050 | 0.0750 | 0.9539 |
| 30.00_bps | factor_ic_weighted_daily | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 30.00_bps | factor_mean_optimized_daily | 30 | -0.2168 | -2.3121 | 0.1284 | 0.1471 | 0.1050 | 0.8902 |
| 30.00_bps | factor_ic_optimized_daily | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 30.00_bps | mlp_top20_raw_daily | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 30.00_bps | mlp_top20_neutralized_daily | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 30.00_bps | mlp_optimized_daily | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Interpretation Notes

- The model attempts walk-forward training; inspect execution coverage before interpreting its metrics.
- Score neutralization is cross-sectional only, not industry or size neutralization.
- Optimized portfolios are constrained to top-scored candidates to keep the public run tractable.
- Read net return together with gross return, turnover, and cost drag.
- This is not investment advice and not a full paper reproduction.
