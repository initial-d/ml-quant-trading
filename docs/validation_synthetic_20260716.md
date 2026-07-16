# Synthetic Validation Baseline

This report records a Windows CPU run of `scripts/public_data_validation.py`
using the deterministic synthetic data source. It is a reproducibility baseline
for the validation harness, not evidence of deployable alpha.

The synthetic GBM panel has no intended predictive signal, so negative strategy
results are expected. The useful signal from this run is that the CLI, feature
path, backtest metrics, transaction-cost grid, and bootstrap path all execute on
Windows with CPU-only PyTorch.

## Environment

| Field | Value |
|---|---|
| Run date | 2026-07-16 |
| OS | Windows 11 (10.0.26100) |
| Python | 3.12.7 |
| PyTorch | 2.13.0+cpu |
| CPU | 13th Gen Intel i9-13900HX |
| CUDA | Not available |

Key packages reported by the contributor:

```text
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
scikit-learn==1.8.0
torch==2.13.0+cpu
cvxpy==1.9.2
scs==3.2.11
click==8.4.1
rich==15.0.0
```

## Command

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

## Results At 7 Bps Effective Cost

| Strategy | Ann Return | Sharpe | Max DD | Turnover | Cost Drag |
|---|---:|---:|---:|---:|---:|
| equal_weight | -35.2% | -2.59 | 37.2% | 0.011 | 0.40% |
| momentum_20 | -35.3% | -2.39 | 37.9% | 0.172 | 6.24% |
| alpha101_mean | -37.5% | -2.87 | 40.1% | 0.530 | 19.21% |

All three strategies are negative on this synthetic panel. That is consistent
with the purpose of the synthetic source: it checks reproducibility and plumbing
rather than presenting a public-market performance claim.

## Cost Sensitivity

The cost grid shows the expected relationship between turnover and cost drag.
At 30 bps effective cost, high-turnover strategies deteriorate much more than
low-turnover baselines:

| Strategy | Sharpe At 0 Bps | Sharpe At 30 Bps | Cost Drag At 30 Bps |
|---|---:|---:|---:|
| equal_weight | -2.57 | -2.67 | 1.7% |
| alpha101_mean | -1.70 | -6.63 | 82.3% |

The run also enabled block bootstrap uncertainty intervals with 100 samples and
a 20-day block size. The detailed bootstrap artifacts were generated locally;
this committed note keeps the portable summary that can be reviewed without the
gitignored `artifacts/` directory.

## Data-Source Notes

- yfinance ETF-50 validation was attempted first, but Yahoo Finance returned
  HTTP 429 rate-limit responses from the contributor's network. All 50 tickers
  failed, so no public-data validation report was produced from that run. The
  blocker is documented in
  https://github.com/initial-d/ml-quant-trading/issues/22#issuecomment-4989435338.
- A small Baostock loader smoke test also succeeded for `sh.600000`, returning
  seven rows. That is only a loader check, not a benchmark result. Adding a
  first-class Baostock CLI source can be handled in a separate change.
