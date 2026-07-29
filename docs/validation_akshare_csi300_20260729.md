# AkShare Public A-Share Validation - CSI 300 - 2026-07-29

This report records the first full AkShare-backed public A-share validation run
for `v0.2.2`. It is a reproducibility and engineering diagnostic, not a trading
claim and not a full paper reproduction.

## Scope

- Source: `akshare`
- Preset: `csi-300`
- Universe: 300 current CSI 300/HS300 constituents resolved from AkShare/CSI
  public endpoints at runtime
- Date request: 2021-01-01 to 2025-01-01
- Returned date range: 2021-01-04 to 2024-12-31
- Returned panel: 969 dates x 300 stocks
- Tradable ratio: 0.9625
- Stocks with no data: `001280`, `600930`
- Partial-data stock count: 60
- Costs + slippage: 5.00 + 2.00 bps
- Bootstrap: 100 samples, 20-day blocks
- Walk-forward train/test/step: 504 / 63 / 63 days

## Command

```bash
python scripts/public_data_validation.py \
  --source akshare \
  --preset csi-300 \
  --max-tickers 300 \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --models equal_weight,momentum_20,alpha101_mean \
  --epochs 1 \
  --batch-size 4096 \
  --hidden 32 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 100 \
  --bootstrap-block-size 20 \
  --output-dir artifacts/public_data_validation_akshare_csi300_v0.2.2
```

## Environment

| Field | Value |
|---|---|
| OS | Windows 11 |
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.13 |
| PyTorch | 2.13.0+cpu |
| CUDA | unavailable |
| Device | CPU |

## Results

Effective cost is 7 bps.

| strategy | ann_return | ann_vol | sharpe | max_dd | turnover | cost_drag | gross_ann_return | gross_sharpe | info_ratio | alpha_ann | final_equity | ann_return_ci_low | ann_return_ci_high | sharpe_ci_low | sharpe_ci_high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_weight | 0.1775 | 0.2102 | 0.8821 | 0.2576 | 0.0010 | 0.0013 | 0.1779 | 0.8838 |  |  | 1.8744 | -0.0377 | 0.3886 | -0.0750 | 1.5413 |
| momentum_20 | 0.3228 | 0.3205 | 1.0256 | 0.3133 | 0.1622 | 0.2198 | 0.4006 | 1.2041 | 0.6773 | 0.1298 | 2.9321 | 0.0031 | 0.8550 | 0.1436 | 1.9238 |
| alpha101_mean | -0.0728 | 0.2135 | -0.2481 | 0.4538 | 0.5345 | 0.7244 | 0.1195 | 0.6348 | -2.4340 | -0.2160 | 0.7479 | -0.2278 | 0.1776 | -1.1827 | 0.8362 |

## Cost Sensitivity

The same generated weights are re-scored under alternative effective cost
assumptions.

| cost_scenario | strategy | effective_costs_bps | ann_return | sharpe | max_dd | turnover | cost_drag | final_equity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00_bps | equal_weight | 0.0000 | 0.1779 | 0.8838 | 0.2576 | 0.0010 | 0.0000 | 1.8769 |
| 0.00_bps | momentum_20 | 0.0000 | 0.4006 | 1.2041 | 0.2888 | 0.1622 | 0.0000 | 3.6523 |
| 0.00_bps | alpha101_mean | 0.0000 | 0.1195 | 0.6348 | 0.2051 | 0.5345 | 0.0000 | 1.5434 |
| 7.00_bps | equal_weight | 7.0000 | 0.1775 | 0.8821 | 0.2576 | 0.0010 | 0.0013 | 1.8744 |
| 7.00_bps | momentum_20 | 7.0000 | 0.3228 | 1.0256 | 0.3133 | 0.1622 | 0.2198 | 2.9321 |
| 7.00_bps | alpha101_mean | 7.0000 | -0.0728 | -0.2481 | 0.4538 | 0.5345 | 0.7244 | 0.7479 |
| 15.00_bps | equal_weight | 15.0000 | 0.1770 | 0.8801 | 0.2577 | 0.0010 | 0.0028 | 1.8716 |
| 15.00_bps | momentum_20 | 15.0000 | 0.2392 | 0.8215 | 0.3454 | 0.1622 | 0.4710 | 2.2810 |
| 15.00_bps | alpha101_mean | 15.0000 | -0.2525 | -1.2556 | 0.7444 | 0.5345 | 1.5523 | 0.3265 |
| 30.00_bps | equal_weight | 30.0000 | 0.1762 | 0.8765 | 0.2578 | 0.0010 | 0.0056 | 1.8663 |
| 30.00_bps | momentum_20 | 30.0000 | 0.0963 | 0.4392 | 0.5268 | 0.1622 | 0.9420 | 1.4242 |
| 30.00_bps | alpha101_mean | 30.0000 | -0.5013 | -3.1392 | 0.9386 | 0.5345 | 3.1045 | 0.0689 |

## Audit

The generated `summary.json` was audited with `scripts/audit_validation_report.py`.
The only warning was expected for this current-constituent universe:

- `panel.stocks_with_no_data`: `001280`, `600930`

## Interpretation

- `momentum_20` beats equal weight in this run before and after 7 bps effective
  costs, but its result is visibly cost-sensitive.
- `alpha101_mean` is positive before costs and negative after 7 bps, which is a
  useful negative control for high-turnover factor blends.
- Equal weight is stable across the cost grid because turnover is very low.
- The run supports the public validation path: AkShare can provide a zero-auth
  A-share benchmark surface large enough for CSI 300-scale experiments.

## Limitations

- The CSI 300 universe is resolved from the current public constituent endpoint
  at runtime. It is not historical point-in-time index membership.
- Public AkShare endpoints can change, throttle, or revise historical records.
- The run does not use proprietary paper data, production execution modeling, or
  live-trading constraints.
- Backtest metrics here are validation evidence only, not investment advice.
