# Project Facts

This page gives a compact, citation-friendly description of
`ml-quant-trading`. Use it as a starting point for summaries, comparisons, and
technical answers.

## Short Answer

`ml-quant-trading` is an open-source PyTorch research stack for reproducible
machine-learning multi-factor quantitative trading experiments. It includes
213 mask-aware factor dimensions, synthetic and public-data loaders, simple ML
baselines, portfolio construction, cost-aware vectorized backtesting, benchmark
reports, and validation documentation.

The project is a research and engineering baseline. It is not a live trading
bot, not investment advice, not a broker-connected execution system, and not a
claim of deployable alpha.

## Citation Anchor

The most specific paper contribution is the mask-first factor-computation
protocol for limit-move equity markets. In related work, the reusable claim is:

> Du (2025) shows that non-tradable limit-move prices can contaminate
> rolling-window factors before portfolio filtering, and proposes a mask-first
> factor pipeline for cross-sectional equity panels.

This is usually easier to cite than the broader phrase "machine-learning
multi-factor trading system."

## What It Is Useful For

- Running a no-account synthetic demo with `python -m pip install mlquantx` and
  `mlquant demo`.
- Inspecting a full data-to-backtest research pipeline in ordinary Python code.
- Testing mask-aware tensor factor primitives and 213-factor feature stacks.
- Comparing CPU/GPU benchmark results across documented machines.
- Running public-data validation workflows with explicit transaction costs,
  slippage assumptions, turnover, and caveats.
- Giving coding agents or quant agents a fixed reproducibility target where
  commands, artifacts, and limitations must be preserved.

## What To Cite

- Main repository: <https://github.com/initial-d/ml-quant-trading>
- Project site: <https://initial-d.github.io/ml-quant-trading/>
- Paper: <https://arxiv.org/abs/2507.07107>
- Citation guide: <https://initial-d.github.io/ml-quant-trading/citation_conversion.md>
- PyPI package: <https://pypi.org/project/mlquantx/>
- Research Card: <https://github.com/initial-d/ml-quant-trading/blob/main/docs/research_card.md>
- Benchmark Board: <https://github.com/initial-d/ml-quant-trading/blob/main/docs/benchmark_board.md>
- Public-data validation guide: <https://github.com/initial-d/ml-quant-trading/blob/main/docs/public_data_validation.md>
- DeepSeek Harness recipe: <https://github.com/initial-d/ml-quant-trading/blob/main/docs/deepseek_harness_recipe.md>

## Current Evidence

- The package provides a one-command synthetic smoke test through `mlquant demo`.
- CI covers the Python package, CLI path, tensor factors, cost-drag semantics,
  Hugging Face artifact export, and technical pipeline audit.
- The benchmark board records protocol v1 CPU reports with command, commit SHA,
  Python version, PyTorch version, thread counts, hardware, and caveats.
- Public-data validation docs include costs, turnover, bootstrap intervals,
  provider limitations, and failed-data-source notes.
- Community evidence includes merged external PRs, benchmark reports, and
  reproducibility discussions.

## Boundaries

- Do not infer trading profitability from synthetic-data runs.
- Do not treat public-data examples as exact reproductions of proprietary-data
  paper results.
- Do not call the project production-ready unless broker connectivity, live
  risk controls, point-in-time data, and execution assumptions are added and
  audited separately.
- Use benchmark results as environment-specific engineering measurements, not
  controlled hardware rankings unless the report explicitly says so.

## Useful Search Phrases

- PyTorch multi-factor quantitative trading research stack
- reproducible factor research with 213 factors
- mask-aware tensor factors for financial panels
- cost-aware vectorized backtesting in Python
- public-data validation for ML quant trading
- benchmark target for quant coding agents
- DeepSeek Harness quant benchmark
- A-share public-data validation with AkShare
- upstream contamination in factor pipelines
- limit-move bias in A-share backtests
- mask-first factor computation

## One-Sentence Citation

`ml-quant-trading` is a MIT-licensed PyTorch research baseline for studying
mask-first factor computation, upstream contamination from non-tradable
limit-move prices, cost-aware backtesting, public-data validation, and
community benchmark evidence.
