# Start Here

This page is the fastest path from discovering the repository to running something useful.

## 1. Pick Your Path

| If you are... | Start with |
|---|---|
| New to the project | `make paper CONFIG=configs/small.yaml` |
| A quant researcher | `notebooks/public_factor_ic.ipynb` |
| An ML engineer | `make benchmark` |
| A contributor | Issues labeled `good first issue` |
| A paper reader | `docs/reproducing_paper.md` |

## 2. Run the Small Pipeline

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e .[dev]
make paper CONFIG=configs/small.yaml
```

This runs the synthetic-data pipeline end to end:

1. generate a masked OCHLV panel
2. compute factor features
3. train a model
4. build a portfolio
5. run a backtest

## 3. Try a Public-Data Notebook

Open:

```text
notebooks/public_factor_ic.ipynb
```

The notebook uses public data when available and falls back to a synthetic panel so the
workflow remains runnable.

## 4. Submit a Useful First Contribution

The fastest useful contributions are:

- run `make benchmark` and submit the result
- report whether the public-data notebook worked for you
- add one caveat or assumption to the docs
- improve a factor-family explanation
- add one small test for an edge case

## 5. What Not to Expect

This project is not:

- a live trading bot
- financial advice
- a source of proprietary market data
- a guarantee that historical backtests will generalize

It is a research and engineering baseline for factor computation, model training,
portfolio construction, and backtesting.
