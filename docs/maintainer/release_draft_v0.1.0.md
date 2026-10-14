# Release Draft: v0.1.0

Title:

> v0.1.0 - Public research baseline for ML-enhanced multi-factor trading

Body:

`ml-quant-trading` is now ready as a public research baseline for people interested in
factor research, tensorized financial data, portfolio optimization, and reproducible
backtesting.

## What is included

- 213 factor dimensions.
- Mask-aware PyTorch tensor factor primitives.
- A-share oriented bias correction for limit-up, limit-down, and halt cases.
- MLP and Transformer baselines.
- AdjMSE, IC, and RankIC losses.
- Cross-sectional Markowitz optimization.
- Vectorized backtesting and metrics.
- Synthetic-data end-to-end pipeline.
- Public-data factor IC notebook.
- CPU/GPU tensor factor benchmark script.
- CI, tests, citation metadata, and contribution templates.

## Quick start

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
pip install -e .[dev]
make paper CONFIG=configs/small.yaml
```

## Community ask

I would especially appreciate:

- benchmark results from different CPUs/GPUs
- public-data reproduction reports
- factor-engine edge cases
- documentation fixes
- small examples that make the project easier to learn

Use the benchmark issue template to submit performance results.

## Disclaimer

This repository is for research and engineering experimentation. It is not financial
advice, investment advice, or a trading recommendation.
