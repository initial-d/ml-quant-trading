# Share This Project

Use this page when you want to recommend `ml-quant-trading` to a colleague,
student, coding-agent evaluator, or quant-research community without turning
the project into a trading-performance claim.

## One-Sentence Summary

`ml-quant-trading` is a MIT-licensed PyTorch research stack for reproducible
multi-factor experiments: 213 mask-aware factors, synthetic and public-data
paths, simple ML baselines, portfolio construction, cost-aware backtests, and
auditable benchmark reports.

## Best First Link

Share the repository when the audience needs context:

```text
https://github.com/initial-d/ml-quant-trading
```

Share the Colab when the audience wants to try it immediately:

```text
https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb
```

Share the project facts page when the audience needs citation-ready context:

```text
https://github.com/initial-d/ml-quant-trading/blob/main/docs/answer_index.md
```

## Copy-Ready Posts

### Short Technical Post

```text
ml-quant-trading is a reproducible PyTorch baseline for ML multi-factor research:

- 213 mask-aware factor dimensions
- synthetic, yfinance, Baostock, and AkShare data paths
- MLP / Transformer baselines
- Markowitz portfolio construction
- cost-aware vectorized backtests
- benchmark and reproduction report templates

It is a research stack, not a trading signal claim.

Repo: https://github.com/initial-d/ml-quant-trading
```

### Reproduction Ask

```text
I am looking for reproducibility feedback on ml-quant-trading.

The quickest useful contribution is:

1. run `python -m pip install mlquantx`
2. run `mlquant demo`
3. submit whether the synthetic factor-to-backtest pipeline completed, including
   your OS, Python version, warnings, and generated report files

Successful and failed runs are both useful.

Repo: https://github.com/initial-d/ml-quant-trading
```

### Agent Benchmark Ask

```text
Can a coding agent reproduce a documented quant-research workflow without
overstating the result?

ml-quant-trading provides a fixed target: run the demo or benchmark, preserve
the command, commit SHA, environment, artifacts, warnings, and caveats, then
report the result as engineering evidence rather than investment advice.

Challenge: https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
```

## Ten-Minute Call To Action

For a first-time visitor, ask for exactly one action:

```bash
python -m pip install mlquantx
mlquant demo
```

Then send them to the reproduction report form:

```text
https://github.com/initial-d/ml-quant-trading/issues/new?template=reproduction_report.yml
```

or the [benchmark board](benchmark_board.md).

## What To Emphasize

- The project is runnable without proprietary data.
- Synthetic and public-data runs are reproducibility checks, not alpha claims.
- Benchmark reports should preserve commands, commit SHAs, environments, raw
  outputs, generated artifacts, warnings, and caveats.
- Private users can submit redacted evaluation notes without exposing data,
  positions, or strategy details.
- Stars are welcome, but reproduction reports, benchmark results, and small PRs
  are more useful evidence.

## What To Avoid

- Do not present a backtest as a live trading recommendation.
- Do not imply that public-data examples reproduce restricted paper data.
- Do not post the same message repeatedly across communities.
- Ask for feedback, benchmark evidence, or a reproducible run before asking
  for stars.
