# mlquant

**A reproducible PyTorch research stack for machine-learning multi-factor
trading: 213 factors, bias correction, portfolio optimization, and vectorized
backtesting.**

[![CI](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2507.07107-b31b1b.svg)](https://arxiv.org/abs/2507.07107)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

## Install and run

```bash
python -m pip install mlquant
mlquant demo
```

The demo needs no market-data account or API key. It runs the deterministic
synthetic pipeline from data generation through 213 factor dimensions, model
training, portfolio construction, cost-aware backtesting, and Markdown/JSON
report generation. The default config ships inside the wheel, so the command
works outside a repository checkout.

## What is included

- 204 hand-crafted factors plus 9 curated Alpha101-style factors
- mask-aware PyTorch tensor primitives for cross-sectional panels
- limit-up, limit-down, halt, and missing-data bias handling
- MLP and Transformer research baselines
- constrained Markowitz portfolio construction
- vectorized backtesting with turnover and transaction costs
- AkShare, Baostock, yfinance, and deterministic synthetic data paths
- auditable public-data validation reports, including negative results

## Start here

- [Source and full documentation](https://github.com/initial-d/ml-quant-trading)
- [Google Colab quick start](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)
- [Public validation dashboard](https://github.com/initial-d/ml-quant-trading/blob/main/docs/validation_dashboard.md)
- [Research card and limitations](https://github.com/initial-d/ml-quant-trading/blob/main/docs/research_card.md)
- [100,000-row synthetic dataset](https://huggingface.co/datasets/dddyym/ml-quant-trading-synthetic)
- [213-input MLP checkpoint](https://huggingface.co/dddyym/ml-quant-trading-synthetic-mlp)
- [Paper: arXiv:2507.07107](https://arxiv.org/abs/2507.07107)

## Research boundary

`mlquant` is research and educational software. It is not investment advice or
a production trading system. Synthetic smoke tests verify engineering behavior,
not profitability. Public-data backtests depend on data quality, survivorship,
transaction costs, slippage, and modeling assumptions and do not represent live
or guaranteed out-of-sample performance.

## License

MIT. See the [repository license](https://github.com/initial-d/ml-quant-trading/blob/main/LICENSE).
