# Roadmap

This roadmap is organized around contributions that make the project easier to run,
easier to trust, and easier to extend.

## Near Term

- Add a public-data case study with a small, stable ticker universe.
- Collect benchmark results from different CPUs, GPUs, and PyTorch versions.
- Add release notes for the first public release.
- Add more examples for factor IC, turnover, drawdown, and attribution.
- Document assumptions around transaction costs, survivorship bias, and slippage.

## Contributor-Friendly Tasks

- Translate key docs between English and Chinese.
- Add tests for neutralization and backtest edge cases.
- Add a small example using a custom CSV data source.
- Add benchmark results through the benchmark issue template.
- Improve docstrings for factor families.
- Add one new public-data notebook with a clearly documented universe.

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
