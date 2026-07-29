# Changelog

## Unreleased

### Dashboard

- Added a cost-sensitivity line chart to the README and validation dashboard.
- The full AkShare CSI 300 pipeline now exports `equity_curves.csv` for future
  daily equity-curve visualizations.
- AkShare ticker downloads now retry transient request failures before marking a
  ticker as unavailable.

## 0.2.3 - Daily 213-Factor Validation Dashboard

This patch release promotes the daily 213-factor AkShare CSI 300 validation into
a first-class dashboard entry. It keeps the project validation-first: the result
is a public-data approximation with transaction costs, not a deployable trading
claim or exact paper reproduction.

### Highlights

- Added a README Validation Dashboard so the latest maintained public-data
  snapshot is visible from the project homepage.
- Added `docs/validation_dashboard.md` as the stable dashboard entry point for
  daily 213-factor validation results and caveats.
- Added `scripts/akshare_csi300_full_pipeline.py` for the daily full-factor
  AkShare CSI 300 validation path.
- Documented the full 213-factor daily report at
  `docs/validation_akshare_csi300_full_pipeline_20260729.md`.
- Added contributor acknowledgement for
  [@redamancy231-create](https://github.com/redamancy231-create), whose AkShare
  loader PR made the zero-auth A-share validation path possible.
- Updated the package version to `0.2.3`.

### Validation Notes

- The main comparison stays daily. Weekly rebalancing can be useful as a
  sensitivity check, but it can also hide the turnover problem this validation
  is meant to measure.
- At 7 bps effective cost, `factor_mean_buffered_daily` outperformed
  `equal_weight_daily` on annual return, Sharpe, and final equity.
- The naive daily 213-factor portfolio retained gross evidence but lost much of
  the edge to turnover and cost drag, which is documented as part of the result.
- The run uses current CSI 300 membership resolved from public endpoints, not
  historical point-in-time membership.

## 0.2.2 - AkShare Public A-Share Validation

This patch release adds a zero-auth A-share public-data validation path through
AkShare and records the first CSI 300-scale public run. It does not change the
research claim or convert the project into a trading recommendation.

### Highlights

- Added AkShare as a CLI-accessible source for `scripts/public_data_validation.py`.
- Added dynamic `csi-300` / `hs300` presets that resolve current CSI 300
  constituents from AkShare/CSI public endpoints at runtime.
- Added AkShare ticker normalization for common A-share code formats such as
  `600519`, `sh.600000`, `000001.SZ`, and `SZ300750`.
- Added source-limitation metadata to generated validation reports.
- Documented the AkShare CSI 300 command in `docs/public_data_validation.md`.
- Added the tracked report
  `docs/validation_akshare_csi300_20260729.md`.
- Added a daily 213-factor AkShare CSI 300 public-data approximation in
  `docs/validation_akshare_csi300_full_pipeline_20260729.md`.

### Validation Notes

- The CSI 300 report used 300 current constituents, 969 returned trading dates,
  and a 96.25% tradable-cell ratio over 2021-01-04 to 2024-12-31.
- The generated report audit passed with one expected warning: two current
  constituents had no data over the requested window.
- The resolved CSI 300 universe is not historical point-in-time membership; it
  is a public-data reproducibility benchmark, not a full paper reproduction or
  deployable alpha claim.
- In the daily 213-factor public run, the buffered factor portfolio outperformed
  equal weight after 7 bps effective costs, while the naive daily factor
  portfolio showed how excessive turnover can consume gross factor edge.

## 0.2.1 - Validation Entrypoints and Outreach Follow-through

This patch release improves the public reproduction funnel after the `v0.2.0`
validation release. It does not change the research claim or add a new trading
signal; it makes the existing validation surface easier to find, run, and
follow up.

### Highlights

- Fixed the Colab Baostock demo bootstrap so fresh runtimes clone the canonical
  `initial-d/ml-quant-trading` repository.
- Added the Colab demo to the tracked public entry points.
- Added a follow-up validation digest for the outreach and reproduction surface.
- Tracked external validation-oriented discussions around backtest correctness,
  public-data blockers, Qlib reconciliation, and replay-safe research contracts.
- Refreshed the visibility pulse after the outreach window.

### Validation Notes

- The project remains research-only and validation-first.
- Public-data failures, rate limits, and blocked reproductions should be reported
  as blockers rather than converted into empty benchmark claims.
- `v0.2.1` is mainly an entrypoint and documentation patch over `v0.2.0`.

## 0.2.0 - Public Validation and Contributor Workflow

This release moves `ml-quant-trading` beyond the first public baseline and into
a more reviewable open-source research workflow.

### Highlights

- Added first-class public validation paths:
  - synthetic reproducibility baseline
  - Baostock A-share validation report
  - yfinance rate-limit troubleshooting and smoke-test workflow
- Added Baostock as a CLI-accessible public-data source.
- Hardened neutralization and Baostock data handling with additional tests.
- Added an English factor handbook for the 213-factor research surface.
- Added Chinese README summaries in `README.zh-CN.md` and `README.zh-TW.md`.
- Added a contributor pairing workflow for public-data validation and benchmark
  reports.
- Added security reporting guidance in `SECURITY.md`.
- Refreshed community outreach and visibility tracking docs.

### Validation Notes

- The public reports are validation diagnostics, not evidence of deployable
  alpha.
- Public-data reports currently show that simple baselines remain difficult to
  beat after costs.
- yfinance may return HTTP 429 rate limits; contributors should run a small
  smoke test before scaling to larger ETF universes.

## 0.1.0 - Public Research Baseline

This is the first public baseline release of `ml-quant-trading`.

### Highlights

- End-to-end synthetic factor-to-backtest pipeline.
- 213 factor dimensions:
  - 204 hand-crafted legacy factors.
  - 9 curated Alpha101-style factors.
- Mask-aware PyTorch tensor primitives for cross-sectional panels.
- Limit-up, limit-down, and halt bias correction.
- MLP and Transformer model baselines.
- Sign-aware losses: AdjMSE, IC, and RankIC.
- Cross-sectional Markowitz portfolio optimization.
- Vectorized backtesting and metrics.
- Public-data factor IC notebook.
- Tensor factor benchmark script.
- CI across Python 3.9, 3.10, and 3.11.
- Citation metadata through `CITATION.cff`.

### Good First Contributions

- Run `make benchmark` and submit a benchmark issue.
- Try the public-data notebook and report reproducibility issues.
- Add one public-data case study.
- Improve factor family documentation.
- Add tests for neutralization and backtest edge cases.

### Notes

The repository is for research and engineering experimentation. It does not provide
financial advice or live trading recommendations. Proprietary market data used in the
paper cannot be redistributed, so the public reproduction paths focus on synthetic data
and public-data examples.
