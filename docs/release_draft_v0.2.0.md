# v0.2.0 - Public Validation and Contributor Workflow

`v0.2.0` turns the first public research baseline into a more reviewable
open-source workflow. The release adds public validation reports, a Baostock
data path, stronger contributor guidance, and clearer limitations around
public-data results.

## Highlights

- Added Baostock public-data validation support and an A-share validation report.
- Added synthetic validation as a Windows CPU reproducibility baseline.
- Documented yfinance HTTP 429 rate-limit behavior and a smoke-test workflow.
- Added an English factor handbook for the 213-factor research surface.
- Added Chinese README summaries for simplified and traditional Chinese readers.
- Hardened neutralization and Baostock handling with additional tests.
- Added contributor pairing guidance for validation and benchmark reports.
- Added `SECURITY.md` for vulnerability reporting.

## Validation Position

This is still a research and education toolkit, not a trading-signal release.
The public reports are useful because they make assumptions, costs, data-source
limits, and baseline behavior easier to inspect.

The current public results do not claim deployable alpha. They mainly show:

- how the validation harness behaves across synthetic and public data;
- how turnover and transaction costs affect interpretation;
- why simple baselines are important controls;
- where contributors can add more reproducible reports.

See [`docs/validation_digest_20260720.md`](validation_digest_20260720.md) for
the release validation summary.

## Useful Entry Points

- Start here: [`docs/start_here.md`](start_here.md)
- Public-data validation: [`docs/public_data_validation.md`](public_data_validation.md)
- Validation digest: [`docs/validation_digest_20260720.md`](validation_digest_20260720.md)
- Benchmark board: [`docs/benchmark_board.md`](benchmark_board.md)
- Reality check: [`docs/reality_check.md`](reality_check.md)
- Pairing issue: <https://github.com/initial-d/ml-quant-trading/issues/22>

## Compatibility Notes

- Package version is now `0.2.0`.
- The project remains Python 3.9+ and PyTorch based.
- The paper remains the research snapshot; `main` continues to evolve as a
  living open-source toolkit.

