# v0.4.0 - Out-of-Sample Evaluation and Lighter Install

`v0.4.0` makes the default pipeline evaluate honestly and makes the base
install lighter. It contains breaking changes; read **Upgrading** first.

All evidence in this release is synthetic or engineering evidence. It is not
investment advice and not a claim of deployable alpha.

## Upgrading from 0.3.x

```bash
python -m pip install -U mlquantx            # synthetic demo only
python -m pip install -U 'mlquantx[data]'    # plus AkShare / Baostock / yfinance loaders
```

- Public-data loaders need the `data` extra. Without it they raise an
  `ImportError` that names the extra.
- `mlquant demo` metrics now cover only the out-of-sample window, so they will
  not match 0.3.x reports, which were in-sample.
- Code that calls `Trainer` directly now gets a chronological validation split.
  Pass `val_split_mode="random"` to reproduce 0.3.x training.
- `run_backtest` charges the initial entry trade. Results change only when the
  first weight row is non-zero.

## Highlights

### The demo is out-of-sample

Up to 0.3.x, `mlquant demo` trained, predicted, and backtested on the same
dates, so its metrics were in-sample. In 0.4.0 the model trains on the first
`train_frac` (default `0.6`) of dates, and the backtest scores only the
remaining dates. `summary.json` records both windows under `evaluation`.

On the packaged synthetic config, the out-of-sample Sharpe after 5 bps costs is
about -0.54. That is the expected result for synthetic data with no predictable
signal, and it is the point of the change: the old in-sample figure suggested
skill that was not there.

### Board-aware limit-move masks

`limit_pct: board` (or `mlquant.features.board_limit_pct`) applies a ±20 %
proxy to STAR Market and ChiNext names (ChiNext from 2020-08-24) and ±30 % to
Beijing Stock Exchange names. Before, a single 9.8 % threshold masked ordinary
moves on those boards and missed their real limit days. Ticker codes cannot
reveal ST status or IPO no-limit days; supply exchange limit prices when those
matter.

### Correctness fixes

- The validation split no longer shuffles adjacent trading days between train
  and validation.
- The backtest charges the cost of building the first book.
- `gen-data` keeps the synthetic panel's exchange limit prices, so the demo
  mask uses the real-limit regime.

### Lighter install and cleaner codebase

- AkShare, Baostock, and yfinance moved to the `data` extra.
- Removed `src/mlquant/features.py`, a module shadowed by the
  `mlquant.features` package that could never be imported.
- Ruff enforces the full pyflakes rule family.
- Launch and release-draft pages moved to `docs/maintainer/`.

### Also included since 0.3.1

- `mlquant systemone build-state` builds typed decision state from pipeline
  artifacts, and `mlquant systemone evaluate` scores decisions against offline
  realized returns.

## Verification

- CI: lint, tests, and CLI smoke test on Python 3.9, 3.10, and 3.11 (PR #72).
- Maintainer run on Windows 11 with Python 3.14.4 and PyTorch 2.11.0+cpu:
  129 tests passed, `mlquant demo` completed, and
  `scripts/technical_pipeline_audit.py` reported overall PASS.

## Known limitations

- The demo uses one chronological split, not walk-forward;
  `scripts/public_data_validation.py` runs walk-forward splits.
- ST (±5 %) limits and IPO no-limit days are not inferred from ticker codes.
- Paper-style empirical reproduction still requires licensed Wind or
  Tushare-style data.
