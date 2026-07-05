# ml-quant-trading

> **Machine Learning Enhanced Multi-Factor Quantitative Trading**
> — A Cross-Sectional Portfolio Optimization Approach with Bias Correction
>
> [arXiv:2507.07107](https://arxiv.org/abs/2507.07107) &nbsp;|&nbsp; Yimin Du, 2025

[![CI](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

![ml-quant-trading social preview](docs/assets/social-preview.jpg)

---

## What is this?

A **clean, fork-friendly, end-to-end** A-share quantitative trading system:

**In one clone, you get:** a tensor factor engine, 213 factor dimensions, bias correction,
ML baselines, Markowitz portfolio optimization, vectorized backtesting, synthetic/public-data
demos, CI, tests, and benchmark tooling.

| Module | What it does |
|--------|-------------|
| `features.tensor_factors` | GPU-vectorised masked primitives (`rank`, `corr`, `ewma`, `ts_*`) |
| `features.legacy_factors` | **204 hand-crafted alpha factors** ([handbook](docs/factor_handbook.md)) |
| `features.alpha101` | Alpha101-style formulaic factors |
| `features.neutralize` | Cross-sectional & industry neutralisation |
| `features.bias` | Limit-up / limit-down / halt bias correction |
| `training.augment` | GBM data augmentation |
| `models.nets` | MLP / Transformer |
| `models.losses` | AdjMSE, IC, RankIC losses |
| `portfolio.markowitz` | Cross-sectional Markowitz (shrunk cov, no-short) |
| `backtest.engine` | Vectorised backtest → Sharpe / IC / IR / DD |

---

## Why Star or Fork This Repository?

Star or fork this repo if you want a compact baseline for:

- Building ML factor research pipelines in PyTorch.
- Reproducing cross-sectional factor and portfolio experiments.
- Testing new alpha features against a working backtest loop.
- Comparing CPU/GPU performance on masked tensor operations.
- Contributing public-data case studies for quant research.

If you are new here, start with [Start Here](docs/start_here.md).

---

## Quick Start

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the synthetic demo
make paper CONFIG=configs/small.yaml

# Run checks
make test
make lint
```

See [Start Here](docs/start_here.md) for a 10-minute walkthrough and [FAQ](docs/faq.md) for common setup questions.

---

## Public-Data Demo

The original paper experiments use proprietary A-share data. This repository includes a lightweight
public-data path so readers can validate the pipeline shape without private data:

- Notebook: [`notebooks/public_data_demo.ipynb`](notebooks/public_data_demo.ipynb)
- Script: [`scripts/run_public_data_demo.py`](scripts/run_public_data_demo.py)
- Benchmark docs: [`docs/benchmark_board.md`](docs/benchmark_board.md)

```bash
python scripts/run_public_data_demo.py --tickers AAPL MSFT NVDA SPY --start 2020-01-01
```

---

## Community and Growth

- [Community guide](docs/community.md)
- [Promotion kit](docs/promotion_kit.md)
- [Launch playbook](docs/launch_playbook.md)
- [Community outreach targets](docs/community_outreach.md)
- [Roadmap](docs/roadmap.md)
- [Content calendar](docs/content_calendar.md)

Want to help? Open a benchmark result, a public-data case study, or a reproducibility issue.

---

## Repository Map

```text
ml-quant-trading/
├── src/ml_quant_trading/     # package code
├── tests/                    # pytest suite
├── configs/                  # experiment configs
├── scripts/                  # runnable demos and benchmark tooling
├── notebooks/                # public-data walkthroughs
└── docs/                     # handbook, roadmap, community assets
```

---

## Citation

If this repository or paper helps your work, please cite it with [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{du2025mlquanttrading,
  title={Machine Learning Enhanced Multi-Factor Quantitative Trading: A Cross-Sectional Portfolio Optimization Approach with Bias Correction},
  author={Du, Yimin},
  year={2025},
  eprint={2507.07107},
  archivePrefix={arXiv},
  primaryClass={q-fin.PM}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
