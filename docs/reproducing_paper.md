# Reproducing the paper

The paper reports results on a proprietary Wind A-share panel that we
cannot redistribute. We support two reproduction paths:

1. **Synthetic-data path** (works for everyone).
   Runs the full pipeline against a GBM-simulated 3 000-stock × 14-year
   panel that obeys the same masking, OCHL, and price-limit constraints
   as Wind data. Numbers match the paper *qualitatively* (positive
   Sharpe, IC > 0.02, monotone risk-aversion → variance frontier) but
   not quantitatively.
2. **Real-data path** (requires Wind / Tushare access).
   Drop a tab-separated OCHLV file at the path referenced from
   ``configs/paper.yaml`` and rerun. The repository ships the loader;
   it does not ship the data.

Before comparing backtest numbers, read the [backtest assumptions and limitations](backtest_assumptions.md). It explains transaction costs, slippage, survivorship bias, data leakage, limit-up / limit-down handling, and the minimum reporting checklist for credible results.

## Quick reproduction (synthetic)

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e .[dev]

# Tiny config: ~30 seconds end-to-end.
make paper CONFIG=configs/small.yaml

# Paper-shaped config: ~10-20 minutes on a single GPU.
make paper CONFIG=configs/paper.yaml
```

The terminal prints a summary table:

```
Backtest summary
----------------------------------------
  ann_return     0.21  ── annualised return
  ann_vol        0.10
  sharpe         2.05
  sortino        2.93
  calmar         1.74
  max_dd         0.12
  turnover       0.27
  cost_drag      0.03
  n_periods      500
```

## Table-by-table mapping

| Paper figure / table                     | Command                                                  | Module                                       |
|------------------------------------------|----------------------------------------------------------|----------------------------------------------|
| Table 1 — factor engine throughput       | `pytest tests/test_tensor_factors.py -v`                 | `mlquant.features.tensor_factors`            |
| Table 2 — Alpha101 IC                    | `python scripts/eval_factor_ic.py --config configs/...`  | `mlquant.features.alpha101`                  |
| Figure 3 — efficient frontier            | `python scripts/plot_frontier.py`                        | `mlquant.portfolio.frontier`                 |
| Table 3 — backtest by year               | `make paper CONFIG=configs/paper.yaml`                   | `mlquant.backtest.engine`                    |
| Table 4 — bias correction ablation       | `python scripts/ablation_bias.py`                        | `mlquant.features.bias`                      |
| Table 5 — sign-aware loss ablation       | `python scripts/ablation_loss.py`                        | `mlquant.models.losses`                      |

The ``scripts/`` helpers are short orchestrators around the package
APIs; they are not required to use the package.

## How to plug in your own data

```python
from mlquant.data.loaders import load_ochlv_csv

panel = load_ochlv_csv("/path/to/wind_dump.tsv", sep="\t")
panel.assert_consistent()
```

Field aliases are tried in this order:

| Output field | Aliases tried                            |
|--------------|------------------------------------------|
| `open`       | `open`, `S_FWDS_ADJOPEN`, `OPEN`         |
| `close`      | `close`, `S_FWDS_ADJCLOSE`, `CLOSE`      |
| `high`       | `high`, `S_FWDS_ADJHIGH`, `HIGH`         |
| `low`        | `low`, `S_FWDS_ADJLOW`, `LOW`            |
| `volume`     | `volume`, `S_DQ_VOLUME`                  |
| `vwap`       | `vwap`, `S_DQ_AVGPRICE`                  |

If your column names don't match, pass a mapping in the loader call or
rename the columns upstream.
