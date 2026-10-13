# ml-quant-trading

**A reproducible PyTorch research stack for cross-sectional factor research:
213 mask-aware factors, public-data validation, cost-aware backtests, and
System-One-style portfolio decisions.**

[![CI](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ml-quant-trading/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/initial-d/ml-quant-trading?style=flat&logo=github&label=Stars)](https://github.com/initial-d/ml-quant-trading/stargazers)
[![Release](https://img.shields.io/github/v/release/initial-d/ml-quant-trading?display_name=tag)](https://github.com/initial-d/ml-quant-trading/releases)
[![PyPI](https://img.shields.io/pypi/v/mlquantx.svg)](https://pypi.org/project/mlquantx/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.07107-b31b1b.svg)](https://arxiv.org/abs/2507.07107)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Languages: [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

[Run in Colab](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)
| [System-One Portfolio Decisions](docs/systemone_adapter.md)
| [Validation Dashboard](docs/validation_dashboard.md)
| [Project Facts](docs/answer_index.md)
| [Citation Guide](docs/citation_conversion.md)
| [Share This Project](docs/share_this_project.md)
| [Research Card](docs/research_card.md)

## What This Is

`ml-quant-trading` is a research baseline, not a live trading system. It gives
students, researchers, engineers, and agent harnesses a concrete way to run and
inspect a full factor-to-backtest workflow:

```text
data -> masks -> 213 factors -> model scores -> portfolio weights -> cost-aware backtest
```

The paper-specific contribution is the mask-first factor-computation protocol:
non-tradable limit-move prices should be masked before rolling-window factors,
not only filtered later at the portfolio layer.

New in the current direction is a System-One-compatible portfolio decision
adapter. It wraps pipeline state as typed asset decisions in a Jev-style shape:

```text
portfolio state + typed questions -> typed portfolio-intent answers
```

It answers research questions such as `asset_action`, `target_weight`,
`trade_allowed`, `risk_bucket`, and `max_participation_rate`. It does not create
broker orders, connect to accounts, or run live execution.

## Quick Start

```bash
python -m pip install mlquantx
mlquant demo
```

The demo needs no market-data account or API key. It runs a deterministic
synthetic pipeline and writes shareable Markdown and JSON reports under
`artifacts/small/`.

For development:

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e '.[dev]'
python -m pytest
```

## System-One Portfolio Decisions

The decision adapter is the project's first step toward a quant-specific
System One model surface. The analogy to Jev is the interface style: a structured
state plus typed questions returns typed answers. This repository does not claim
official Jev certification or calibrated probabilities.

Example state:

```json
{
  "as_of": "2024-12-31",
  "universe": ["600519", "000001"],
  "predictions": {
    "score": {"600519": 0.018, "000001": -0.006},
    "rank": {"600519": 0.91, "000001": 0.23}
  },
  "tradability": {
    "600519": {"tradable": true, "halted": false, "limit_up": false, "limit_down": false}
  },
  "liquidity": {
    "600519": {"amount": 1523000000, "adv20": 1340000000}
  },
  "portfolio": {
    "current_weight": {"600519": 0.02},
    "target_weight_baseline": {"600519": 0.04}
  },
  "risk": {
    "asset_volatility_20d": {"600519": 0.021},
    "max_position_weight": 0.05
  },
  "cost_model": {
    "effective_cost_bps": 7,
    "max_participation_rate": 0.1
  }
}
```

Run typed decisions:

```bash
mlquant systemone decide \
  --state artifacts/systemone/decision_state.json \
  --output artifacts/systemone/portfolio_decision.json
```

Default questions expand over every asset in `universe`. A custom question file
can ask only for selected assets or fields. See
[docs/systemone_adapter.md](docs/systemone_adapter.md).

## Why It Matters

| Capability | Why it matters |
|---|---|
| 213 factor dimensions | A broad mask-aware factor surface for cross-sectional research |
| Mask-first pipeline | Avoids upstream contamination from non-tradable limit-move observations |
| Public and synthetic data paths | Start without private data, then test AkShare, Baostock, or yfinance |
| Cost-aware validation | Keeps turnover, costs, drawdown, baselines, and caveats close to results |
| Typed decision adapter | Exposes portfolio intent as a callable System-One/Jev-style interface |
| Auditable reports | Preserves commands, environment, data source, assumptions, and failures |

## Public Validation Snapshot

Latest maintained public-data snapshot:
[AkShare CSI 300 Daily 213-Factor Validation](docs/validation_dashboard.md).

| Strategy | Annual return | Sharpe | Turnover | Final equity | Note |
|---|---:|---:|---:|---:|---|
| `equal_weight_daily` | 17.75% | 0.882 | 0.0010 | 1.8744 | Public-data baseline |
| `factor_mean_daily` | 15.80% | 0.701 | 0.3627 | 1.7579 | Gross edge, high turnover |
| `factor_mean_buffered_daily` | 22.20% | 0.919 | 0.1397 | 2.1616 | Buffered daily factor rule |

This is a public-data approximation using current CSI 300 constituents, not an
exact paper reproduction or investment claim. The useful point is framework
level: factor scores need tradability, turnover, and cost-aware portfolio
construction before backtests are interpreted.

## Core Modules

| Module | Role |
|---|---|
| `mlquant.data` | Synthetic, yfinance, Baostock, and AkShare panel loaders |
| `mlquant.features` | Mask-aware tensor primitives, Alpha101 subset, and 204 legacy factors |
| `mlquant.training` | Factor dataset, augmentation, and trainer |
| `mlquant.models` | MLP, Transformer, AdjMSE, IC, and RankIC losses |
| `mlquant.portfolio` | Long-only Markowitz-style portfolio construction |
| `mlquant.backtest` | Vectorized backtest and metrics |
| `mlquant.systemone` | Typed portfolio-intent decisions from structured state |

## Common Paths

| Goal | Start here |
|---|---|
| Run the zero-account demo | `mlquant demo` |
| Try public data | [Public-Data Validation](docs/public_data_validation.md) |
| Inspect maintained results | [Validation Dashboard](docs/validation_dashboard.md) |
| Use typed portfolio decisions | [System-One Adapter](docs/systemone_adapter.md) |
| Share the project safely | [Share This Project](docs/share_this_project.md) |
| Cite the project | [Citation Conversion Guide](docs/citation_conversion.md) |
| Understand boundaries | [Research Card](docs/research_card.md) and [Reality Check](docs/reality_check.md) |
| Benchmark tensor factors | [Benchmarking Guide](docs/benchmarking.md) |
| Evaluate an agent | [Agent Quant Benchmark Challenge](docs/agent_quant_benchmark_challenge.md) |

## Data Sources

| Source | Market | Access | Notes |
|---|---|---|---|
| Synthetic | N/A | Zero-config | Deterministic smoke tests and CI |
| AkShare | A-shares | Public, no API key | Useful public A-share path, endpoints may change |
| Baostock | A-shares | Free registration | Account-backed public data route |
| yfinance | US equities / ETFs | Public, rate-limited | Convenient validation and notebook examples |

The repository does not redistribute market data. AkShare, Baostock, and
yfinance data are downloaded on demand by the loader scripts. Public upstream
interfaces can change or rate-limit requests. Synthetic data is generated
deterministically from a fixed seed.

## Citation

If you write about A-share factor pipelines, cite the mask-first treatment of
non-tradable limit-move observations and rolling-window contamination:

```bibtex
@article{du2025mlquant,
  title  = {Machine Learning Enhanced Multi-Factor Quantitative Trading:
            A Cross-Sectional Portfolio Optimization Approach with Bias Correction},
  author = {Du, Yimin},
  journal= {arXiv preprint arXiv:2507.07107},
  year   = {2025},
  doi    = {10.48550/arXiv.2507.07107},
  url    = {https://arxiv.org/abs/2507.07107}
}
```

## Boundary

This project is for research and engineering experimentation. It is not
financial advice, not investment advice, not a broker-connected system, and not
a claim of live deployable alpha. Synthetic demos, public-data validations, and
typed portfolio decisions are research artifacts.

## License

MIT. See [LICENSE](LICENSE).
