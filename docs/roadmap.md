# Roadmap

This roadmap is organized around contributions that make the project easier to run,
easier to trust, and easier to extend.

## Near Term

- Collect larger public-data validation reports using `scripts/public_data_validation.py`.
- Collect benchmark results from different CPUs, GPUs, CUDA versions, and PyTorch versions.
- Expand first-run onboarding based on new user feedback.
- Add more examples for factor IC, attribution, and regime-specific diagnostics.
- Add independent reproductions with explicit survivorship-bias and point-in-time data controls.

## Completed Launch Items

- Published `v0.1.0` as the first public research baseline.
- Added a maintainer CPU benchmark baseline to `docs/benchmark_board.md`.
- Added a yfinance public-data mini reproduction to `docs/public_data_mini_reproduction.md`.
- Added a larger public-data validation harness with walk-forward baselines, costs, slippage, turnover, and drawdown.
- Added `docs/reality_check.md` to separate engineering validation, smoke tests, data-gated paper reproduction, and production-readiness claims.
- Added a Dev Container for reproducible contributor setup.
- Added a Mermaid architecture diagram in `docs/architecture.md`.

## Contributor-Friendly Tasks

- Translate key docs between English and Chinese.
- Add tests for neutralization and backtest edge cases.
- Add a small example using a custom CSV data source.
- Add benchmark results through the benchmark issue template.
- Run `scripts/public_data_validation.py` on a new public-data universe and report the exact command.
- Improve docstrings for factor families.
- Add one new ETF or larger-universe public-data example with a clearly documented universe.

## Research Extensions

- Expand Alpha101 formula coverage.
- Add factor selection examples.
- Add cross-validation and walk-forward evaluation examples.
- Add ablation scripts for bias correction, losses, and transaction costs.
- Add portfolio attribution reports.

## Engineering Extensions

- Add optional GPU benchmark reporting.
- Add parquet-based data loading examples.
- Add reproducible environment files for CUDA and CPU-only users.
- Add a minimal web dashboard for benchmark and backtest summaries.
- Tighten linting gradually after a dedicated formatting pass.

## Community Milestones

- First external benchmark result.
- First public-data reproduction issue.
- First external PR.
- First tagged release.
- First Zenodo archive or DOI-backed software release.
- First third-party tutorial or blog post.
