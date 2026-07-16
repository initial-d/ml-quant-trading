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

Expected local runtime:

- CPU-only laptop: about 30-90 seconds after dependencies are installed.
- Memory: the small config is designed for ordinary laptops and CI runners.
- GPU: not required for the quick start.

This runs the synthetic-data pipeline end to end:

1. generate a masked OHLCV panel
2. compute factor features
3. train a model
4. build a portfolio
5. run a backtest

Expected artifact directories:

```text
artifacts/small/
data/processed/
```

If the command succeeds, you should see each pipeline stage finish without a traceback.
The exact metric values can vary across Python, PyTorch, BLAS, and solver versions, so
the quick-start check is "the full pipeline completes" rather than "the final number is
identical on every machine."

Typical successful output includes lines like:

```text
wrote artifacts/small/panel.pt  (500 dates x 200 stocks)
wrote features: torch.Size([500, 200, 213])
wrote predictions: torch.Size([500, 200])
wrote weights: (500, 200)

Backtest summary
```

During the portfolio step, `cvxpy` or `scipy` may print numerical warnings on some
machines. Treat them as warnings unless the command exits with an error or the final
`Backtest summary` is missing.

## 3. Try a Public-Data Notebook

Before interpreting any result, skim the
[Research Card](research_card.md). It explains the intended use, non-goals,
data assumptions, and current validation status in one place.

Open:

```text
notebooks/public_factor_ic.ipynb
```

The notebook uses public data when available and falls back to a synthetic panel so the
workflow remains runnable.

For a fixed public-data reference run, see:

```text
docs/public_data_mini_reproduction.md
```

That note records the ticker universe, date range, factor subset, and expected one-day
rank IC summary from the maintainer run.

## 4. Use A Reproducible Container

If you use VS Code or GitHub Codespaces, open the repository in the included Dev
Container:

```text
.devcontainer/devcontainer.json
```

The container installs Python 3.11 and runs:

```bash
python -m pip install -e .[dev]
```

This is the most reproducible path for first-time contributors who do not want to debug
local Python environments.

## 5. Submit a Useful First Contribution

The fastest useful contributions are:

- run `make benchmark` and submit the result
- report whether the public-data notebook worked for you
- add one caveat or assumption to the docs
- improve a factor-family explanation
- add one small test for an edge case

Good current entry points:

- [Collect community CPU/GPU benchmark results](https://github.com/initial-d/ml-quant-trading/issues/7)
- [Share benchmark and public-data reproductions](https://github.com/initial-d/ml-quant-trading/issues/12)
- [Add an ETF or larger-universe public-data reproduction](https://github.com/initial-d/ml-quant-trading/issues/16)

## 6. What Not to Expect

This project is not:

- a live trading bot
- financial advice
- a source of proprietary market data
- a guarantee that historical backtests will generalize

It is a research and engineering baseline for factor computation, model training,
portfolio construction, and backtesting.
