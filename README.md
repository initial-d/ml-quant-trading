# ml-quant-trading

> Reference implementation of the paper
> **“Machine Learning Enhanced Multi-Factor Quantitative Trading: A Cross-Sectional
> Portfolio Optimization Approach with Bias Correction”** &nbsp;
> ([arXiv:2507.07107](https://arxiv.org/abs/2507.07107)) — Yimin Du, 2025.

[![CI](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

A clean, **end-to-end-runnable** reference implementation of a
cross-sectional, multi-factor, ML-driven A-share trading pipeline. The
repository ships:

* **GPU-vectorised factor engine** &nbsp;— masked PyTorch primitives
  (`rank / corr / cov / std / ewma / ts_*`) that scale to 5 000+ stocks ×
  3 000 days at interactive speed.
* **An Alpha101-style factor library** built on the engine.
* **Cross-sectional neutralisation + bias correction** for limit-up /
  limit-down / halted-trading days that are otherwise unfillable.
* **Geometric-Brownian-Motion data augmentation** for ML training.
* **Cross-sectional Markowitz portfolio optimisation** with an efficient-frontier sweep,
  shrunk covariance, no-short constraint, and a default OSS solver path
  (SCS / ECOS via cvxpy) — MOSEK is opt-in.
* **A vectorised backtest engine** with the metrics you actually report
  in a paper: annualised return, Sharpe, Sortino, Calmar, max-drawdown,
  turnover, IC / RankIC / IR.
* **A synthetic data generator** so anyone can reproduce the full
  pipeline without paid Wind / Tushare data.
* **Unit tests, CI, type hints, and a single `make paper` target.**

> ✅ The repository is intentionally small, opinionated, and reproducible.
> Legacy research scripts that drove the original paper are preserved
> under [`legacy/`](legacy/) for archival reference but are **not** part
> of the supported package surface.

---

## Architecture

```
                   ┌────────────────────────────────────────────┐
   raw OCHLV ─►    │  data.loaders / data.synthetic              │
                   │  → [date, stock] panels with NaN masks      │
                   └──────────────────────┬─────────────────────┘
                                          ▼
                   ┌────────────────────────────────────────────┐
                   │  features.tensor_factors   (GPU, masked)    │
                   │  features.alpha101         (factor library) │
                   │  features.neutralize       (CS / industry)  │
                   └──────────────────────┬─────────────────────┘
                                          ▼
                   ┌────────────────────────────────────────────┐
                   │  training.dataset + training.augment        │
                   │   (geometric Brownian motion augmentation)  │
                   │  models.nets   (MLP / Transformer)          │
                   │  models.losses (AdjMSE, IC, RankIC)         │
                   │  training.trainer                           │
                   └──────────────────────┬─────────────────────┘
                                          ▼
                   ┌────────────────────────────────────────────┐
                   │  portfolio.markowitz   (rotated-quad cone)  │
                   │  portfolio.frontier    (α-sweep)            │
                   └──────────────────────┬─────────────────────┘
                                          ▼
                   ┌────────────────────────────────────────────┐
                   │  backtest.engine + backtest.metrics         │
                   │  → Sharpe / IC / IR / DD / turnover         │
                   └────────────────────────────────────────────┘
```

Each box is one Python module with type hints, docstrings, and tests.
The `mlquant` CLI exposes a sub-command per stage so you can run any
slice of the pipeline in isolation.

---

## Install

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e .[dev]          # add `,gpu` if you have CUDA
```

For the optional MOSEK path:

```bash
python -m pip install -e .[dev,mosek]
```

---

## 30-second smoke test

The package ships a tiny synthetic A-share-like panel so you can verify
the install and run the entire paper pipeline on a laptop in under a
minute:

```bash
make paper CONFIG=configs/small.yaml
```

This runs `gen-data → features → train → portfolio → backtest` against a
synthetic universe of 200 stocks × 500 trading days and prints a metrics
summary table. The same command with `CONFIG=configs/paper.yaml` is what
produced the numbers reported in the paper (assuming you have access to
the proprietary Wind / Tushare panels).

---

## Reproducing the paper

See [`docs/reproducing_paper.md`](docs/reproducing_paper.md) for the
table-by-table mapping between paper sections and command-line
invocations, including the exact configs, model checkpoints, and
expected metric ranges.

| Paper section                              | Code module                          | Tests                       |
| ------------------------------------------ | ------------------------------------ | --------------------------- |
| §3.1 Tensor-accelerated factor engine      | `mlquant.features.tensor_factors`    | `tests/test_tensor_factors` |
| §3.2 Alpha101 + microstructure factors     | `mlquant.features.alpha101`          | `tests/test_alpha101`       |
| §3.3 Cross-sectional neutralisation        | `mlquant.features.neutralize`        | `tests/test_neutralize`     |
| §3.4 Bias correction (limit days)          | `mlquant.features.bias`              | `tests/test_bias`           |
| §4.1 GBM data augmentation                 | `mlquant.training.augment`           | `tests/test_augment`        |
| §4.2 ML models & sign-aware losses         | `mlquant.models.{nets,losses}`       | `tests/test_losses`         |
| §5   Cross-sectional Markowitz             | `mlquant.portfolio.markowitz`        | `tests/test_markowitz`      |
| §6   Backtest, Sharpe, IC                  | `mlquant.backtest.{engine,metrics}`  | `tests/test_metrics`        |

---

## Repository layout

```
ml-quant-trading/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── configs/
│   ├── small.yaml          # 200 stocks × 500 days synthetic, ~30s on a laptop
│   └── paper.yaml          # Full A-share, 2010-2024
├── docs/
│   ├── architecture.md
│   ├── factors.md
│   └── reproducing_paper.md
├── src/mlquant/
│   ├── data/      features/      training/      models/
│   ├── portfolio/ backtest/      cli/
│   └── utils/
├── tests/                  # pytest suite
├── scripts/                # one-shot helpers
└── legacy/                 # original research scripts (unsupported)
```

---

## Citation

```bibtex
@article{du2025mlquant,
  title  = {Machine Learning Enhanced Multi-Factor Quantitative Trading:
            A Cross-Sectional Portfolio Optimization Approach with Bias
            Correction},
  author = {Du, Yimin},
  journal= {arXiv preprint arXiv:2507.07107},
  year   = {2025},
  url    = {https://arxiv.org/abs/2507.07107}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Note that the optional MOSEK path is
governed by MOSEK's own commercial licence; the default install relies
only on open-source solvers (SCS / ECOS).
