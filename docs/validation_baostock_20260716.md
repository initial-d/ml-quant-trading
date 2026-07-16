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
| Tradable ratio | 100% (all 25 tickers downloaded successfully) |
| Stocks with no data | none |

## Results (7 bps effective cost)

| strategy | ann_return | sharpe | max_dd | turnover | cost_drag |
| --- | ---: | ---: | ---: | ---: | ---: |
| equal_weight | +1.7% | 0.18 | 28.8% | 0.001 | 0.07% |
| momentum_20 | +0.2% | 0.14 | 41.6% | 0.179 | 24.2% |
| alpha101_mean | −23.2% | −1.02 | 73.1% | 0.533 | 72.2% |
| mlp_alpha101 | −11.9% | −0.81 | 53.5% | 0.272 | 36.8% |
| transformer_alpha101 | −6.0% | −0.38 | 35.3% | 0.282 | 38.2% |

Equal-weight positive (+1.7%), factor and ML strategies negative — consistent with A-share conditions over 2021–2024 where buy-and-hold a diversified basket modestly outperformed naive factor rotation.

## Cost sensitivity

At 30 bps, high-turnover strategies suffer disproportionately:
- alpha101_mean Sharpe drops from −0.21 (0 bps) → −3.62 (30 bps), cost drag 309%
- equal_weight Sharpe barely moves (0.185 → 0.182), cost drag only 0.3%

Bootstrap CIs (100 samples, block 20 days) included for all scenarios.

## Key packages

numpy==2.4.4 · pandas==3.0.2 · scipy==1.17.1 · scikit-learn==1.8.0 · torch==2.13.0+cpu · cvxpy==1.9.2 · scs==3.2.11 · click==8.4.1 · baostock==00.9.10
