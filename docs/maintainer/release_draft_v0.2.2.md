# v0.2.2 - AkShare Public A-Share Validation

`v0.2.2` adds a zero-auth A-share validation path through AkShare and records a
CSI 300-scale public-data run. This release is still validation-first: it makes
the project easier to reproduce and audit, but it does not claim deployable
alpha or investment performance.

## Highlights

- AkShare is now available in `scripts/public_data_validation.py` via
  `--source akshare`.
- New `csi-300` / `hs300` presets resolve the current CSI 300 constituent list
  from public AkShare/CSI endpoints at runtime.
- A-share ticker normalization accepts common forms such as `600519`,
  `sh.600000`, `000001.SZ`, and `SZ300750`.
- Generated reports now record source limitations for AkShare and dynamic CSI
  300 universes.
- A tracked CSI 300 report is available at
  `docs/validation_akshare_csi300_20260729.md`.

## Reference Command

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
  --bootstrap-block-size 20
```

## Validation Result

The maintained Windows CPU run returned a 969 x 300 panel over
2021-01-04 to 2024-12-31 with a 96.25% tradable-cell ratio.

At 7 bps effective cost:

| strategy | ann_return | sharpe | max_dd | turnover | final_equity |
|---|---:|---:|---:|---:|---:|
| equal_weight | 0.1775 | 0.8821 | 0.2576 | 0.0010 | 1.8744 |
| momentum_20 | 0.3228 | 1.0256 | 0.3133 | 0.1622 | 2.9321 |
| alpha101_mean | -0.0728 | -0.2481 | 0.4538 | 0.5345 | 0.7479 |

The negative `alpha101_mean` net result is a cost-and-turnover lesson, not a
blanket statement against factor research. Its gross annual return is positive
(`0.1195`), but high turnover (`0.5345`) creates a large 7 bps cost drag
(`0.7244`) and turns the net annual return negative. In other words, this run
shows that naive factor averaging needs turnover control, neutralization, and
portfolio construction before it can be interpreted as a serious alpha model.

## Caveats

- The `csi-300` preset uses the current constituent list available from public
  endpoints at runtime. It is not historical point-in-time index membership.
- AkShare is a convenient public data source, but upstream endpoints can change,
  throttle, or revise data.
- The report is not a full paper reproduction because the proprietary paper data
  cannot be redistributed.
- Results are research diagnostics, not investment advice or live-trading
  evidence.

## Verification

- `pytest tests/test_akshare.py tests/test_public_data_validation.py`
- `scripts/audit_validation_report.py` on the generated CSI 300 `summary.json`
