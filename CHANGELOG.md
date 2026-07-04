# Changelog

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
