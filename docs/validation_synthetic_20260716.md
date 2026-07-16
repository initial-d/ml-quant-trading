# Synthetic Validation Baseline

Run on: 2026-07-16 | Windows 11 · Python 3.12.7 · PyTorch 2.13.0+cpu

## Environment

- CPU: 13th Gen Intel i9-13900HX
- CUDA: not available

## Command

```bash
python scripts/public_data_validation.py \
  --source synthetic \
  --models equal_weight,momentum_20,alpha101_mean \
  --epochs 1 --batch-size 4096 --hidden 32 \
  --cost-grid-bps 0,7,15,30 \
  --bootstrap-samples 100
```

## Results Summary

See the full report at `artifacts/public_data_validation/` (generated locally, not committed -- artifacts/ is gitignored).

## Additional Data-Source Tests

- **yfinance (ETF-50):** HTTP 429 rate-limited from this IP. All 50 tickers failed.
- **Baostock (A-shares):** `load_baostock_panel()` works (`sh.600000` → 7 rows). CLI currently only exposes `yfinance|synthetic` — adding `baostock` as a source would help Chinese contributors.
