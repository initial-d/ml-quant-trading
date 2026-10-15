# v0.4.1 - Mask-First Factors and Halt-Safe Returns

`v0.4.1` is a correctness release. Its main fix makes the package implement
the paper's core contribution, the mask-first factor protocol, in the CLI and in
the public-data validation scripts. It has no API removals, but factor values
and backtest metrics change on any panel with limit moves or halts.

All evidence in this release is synthetic or engineering evidence. It is not
investment advice and not a claim of deployable alpha.

## Upgrading from 0.4.0

```bash
python -m pip install -U mlquantx            # or 'mlquantx[data]' for public-data loaders
```

Re-run any report produced with 0.4.0 or earlier on A-share or halt-bearing
data. Results on panels without limit moves or halts are unchanged.

## Fixes

### Factors are now computed mask-first

The paper argues that non-executable limit-move prices must be masked *before*
rolling-window factors are computed, because filtering rows afterwards cannot
remove a price that has already entered a window aggregate.

Up to 0.4.0 the package did not follow that rule:

- `mlquant features` computed factors on the unmasked panel and intersected the
  limit mask with the output afterwards. The limit day was dropped, but its
  price still entered the next `window` days of every rolling factor.
- `scripts/public_data_validation.py` and
  `scripts/akshare_csi300_full_pipeline.py` applied no limit-move mask at all.

In 0.4.1 all three compute factors on a panel whose mask already excludes
limit-move days. The validation scripts use the board-aware thresholds
(±10 % Main Board, ±20 % STAR/ChiNext, ±30 % BSE) and leave non-A-share tickers
untouched. A new helper, `mlquant.features.apply_limit_mask`, returns a panel
with limit days masked and prices unchanged. P&L still uses unmasked returns,
because a position held into a limit move still earns that move.

### Returns are halt-safe

Loaders zero-fill halted prices. Without `last_close`, `Panel.returns` divided
by the previous row, so a halt was booked as a -100 % return and the resumption
as a huge gain. `Panel.returns` now measures each return from the last tradable
close, and `mlquant portfolio`, `mlquant backtest`, and the limit-move return
proxy all use it. A resumption day is no longer mistaken for a limit move.

## Effect on reported numbers

- On the packaged synthetic demo, the out-of-sample Sharpe after 5 bps costs
  moves from about -0.54 to about -0.65. Synthetic data carries no predictable
  signal, so neither figure is evidence of skill.
- The archived A-share validation reports and the validation dashboard were
  produced before this fix. They keep their recorded numbers and now carry a
  protocol note saying so.

## Verification

- CI: lint, tests, and CLI smoke test on Python 3.9, 3.10, and 3.11 (PR #74).
- Maintainer run on Windows 11 with Python 3.14.4 and PyTorch 2.11.0+cpu: 136
  tests passed and `mlquant demo` completed. Spot checks confirmed that a
  return across a zero-filled halt equals the resumption close divided by the
  last pre-halt close, minus one, and that the resumption day is not flagged
  as a limit move.

## Known limitations

- ST (±5 %) limits and IPO no-limit days are still not inferred from ticker
  codes.
- With timezone-aware dates, `board_limit_pct` converts to UTC dates. For an
  Asia/Shanghai midnight timestamp this shifts the date back by one day, which
  only matters on the ChiNext rule-change date (2020-08-24). The current
  AkShare and Baostock loaders return timezone-naive dates.
