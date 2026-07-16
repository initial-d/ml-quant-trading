# Baostock A-Share Public-Data Validation Report

> 2026-07-16 · Windows 11 · PyTorch 2.13.0+cpu · Python 3.12.7

## Command

```bash
python scripts/public_data_validation.py \
  --source baostock \
  --preset cn-large-25 \
  --max-tickers 25 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 100
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 (10.0.26100) |
| Python | 3.12.7 |
| PyTorch | 2.13.0+cpu |
| CUDA | Not available |

## Data Coverage

| Field | Value |
|------|-------|
| Source | baostock |
| Tickers | 25 (SSE 15 + SZSE 10 A-share blue chips) |
| Date range | 2021-01-04 to 2024-12-31 |
| Trading days | 969 |
| Download coverage | all 25 requested tickers returned data |
| Stocks with no data | none |

## Results (7 bps effective cost)

| strategy | ann_return | sharpe | max_dd | turnover | cost_drag |
| --- | ---: | ---: | ---: | ---: | ---: |
| equal_weight | +1.7% | 0.18 | 28.8% | 0.001 | 0.07% |
| momentum_20 | +0.2% | 0.14 | 41.6% | 0.179 | 24.2% |
| alpha101_mean | −23.2% | −1.02 | 73.1% | 0.533 | 72.2% |
| mlp_alpha101 | −11.9% | −0.81 | 53.5% | 0.272 | 36.8% |
| transformer_alpha101 | −6.0% | −0.38 | 35.3% | 0.282 | 38.2% |

The public A-share run is weak after costs: equal weight is only slightly
positive, while higher-turnover strategies deteriorate materially after
transaction costs. Treat this as a reproducibility and validation report, not an
alpha claim.

## Cost sensitivity

At 30 bps, high-turnover strategies suffer disproportionately:
- alpha101_mean Sharpe drops from −0.21 (0 bps) → −3.62 (30 bps), with very large cost drag
- equal_weight Sharpe barely moves (0.185 → 0.182), because turnover is minimal

Bootstrap CIs (100 samples, block 20 days) included for all scenarios.

## Key packages

numpy==2.4.4 · pandas==3.0.2 · scipy==1.17.1 · scikit-learn==1.8.0 · torch==2.13.0+cpu · cvxpy==1.9.2 · scs==3.2.11 · click==8.4.1 · baostock installed (exact version not verified in this PR)
