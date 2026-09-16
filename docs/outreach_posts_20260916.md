# Outreach Drafts: Reproducibility Challenge

This file keeps the September 2026 outreach drafts for the reproducibility
challenge. The tone is intentionally evidence-first: invite benchmark and
validation reports, avoid star requests, and avoid investment-performance
claims.

## Status

| Platform | Status | URL | Notes |
|---|---|---|---|
| GitHub Community | Posted | https://github.com/orgs/community/discussions/208010 | Programming Help category; framed as reproducibility feedback |
| Hacker News | Skipped for now |  | `news.ycombinator.com` timed out in the in-app browser |
| Reddit r/algotrading | Drafted |  | Awaiting manual review/submission |
| Chinese quant community | Drafted |  | Platform TBD |
| DeepSeek Harness community | Posted | https://github.com/deepseek-ai/deepseek-harness/discussions/6831 | Show Your Plugins category; framed as an optional DSH benchmark target |

## Hacker News

Title:

```text
Show HN: A reproducible PyTorch benchmark target for ML quant research
```

Body:

```text
I built ml-quant-trading, a MIT-licensed PyTorch research stack for multi-factor quantitative trading experiments:

https://github.com/initial-d/ml-quant-trading

The useful part is not a trading signal claim. It is a reproducibility target: one command runs a 213-factor tensor pipeline, model baseline, portfolio construction, and cost-aware backtest on synthetic data, then writes inspectable reports.

I am trying to collect public benchmark/reproduction records across machines and runtimes. Successful runs, slow runs, and failed runs are all useful if they include the command, commit SHA, environment, and caveats.

Current asks:

- CUDA GPU protocol v1 benchmark
- Linux CPU benchmark
- larger-panel stress test
- public-data validation report
- redacted private-evaluation note if you cannot share data

No star request, no investment advice, no live trading claim. I would especially appreciate criticism of the benchmark protocol and validation framing.
```

## Reddit r/algotrading

Title:

```text
Looking for reproducibility feedback on an open-source PyTorch factor research stack
```

Body:

```text
I maintain an open-source project called ml-quant-trading:

https://github.com/initial-d/ml-quant-trading

It is a PyTorch research stack for multi-factor experiments: 213 mask-aware factor dimensions, synthetic/public data paths, simple ML baselines, portfolio construction, and cost-aware vectorized backtesting.

I am not posting this as a profitable strategy or live trading bot. The goal is to make the research pipeline auditable and collect independent reproduction evidence.

The most useful feedback right now would be one of these:

- run the protocol v1 benchmark on Linux CPU or CUDA GPU
- try the public-data validation path and report where it breaks
- critique the benchmark/reporting protocol
- share a redacted private evaluation note without exposing data or strategy details

Benchmark board:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/benchmark_board.md

Research card / caveats:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/research_card.md

Successful, slow, and failed runs are all useful as long as the command, commit SHA, environment, and limitations are preserved.
```

## GitHub Community

Title:

```text
Reproducibility challenge for a PyTorch ML quant research stack
```

Body:

```text
I am looking for reproducibility feedback on `ml-quant-trading`, a MIT-licensed PyTorch research stack for machine-learning multi-factor quantitative trading experiments:

https://github.com/initial-d/ml-quant-trading

The project includes 213 mask-aware factor dimensions, synthetic and public-data loaders, model baselines, portfolio construction, cost-aware vectorized backtesting, and benchmark/validation reports.

This is not a live trading bot and not an alpha claim. The current goal is to collect small, reviewable evidence from different machines and environments:

- CUDA GPU protocol v1 benchmark
- Linux CPU benchmark
- larger-panel stress test with memory notes
- public-data validation report
- redacted private-evaluation note if data or strategy details cannot be shared

Start here:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/start_here.md

Benchmark board:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/benchmark_board.md

Research card:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/research_card.md

Successful, slow, failed, or caveated reports are all useful if they preserve the command, commit SHA, environment, artifacts, and limitations.
```

## Chinese Quant Community

Title:

```text
征集复现：一个 PyTorch 213 因子量化研究栈的 benchmark / validation 结果
```

Body:

```text
我在维护一个开源项目 `ml-quant-trading`：

https://github.com/initial-d/ml-quant-trading

它是一个用于多因子量化研究的 PyTorch 工程基线，包含 213 个 mask-aware 因子维度、synthetic / AkShare / Baostock / yfinance 数据路径、简单 ML baseline、组合构建、带交易成本的向量化回测，以及 benchmark / validation 报告。

这不是实盘交易机器人，也不是收益率宣传。我现在更想收集的是可复现证据：不同机器、不同系统、不同数据路径下，能不能跑通，哪里慢，哪里失败，哪些假设需要写清楚。

目前最有价值的反馈：

- CUDA GPU protocol v1 benchmark
- Linux CPU benchmark
- 大 panel stress test，附内存说明
- AkShare / Baostock / yfinance public-data validation
- 如果不能公开数据，也可以交脱敏的 private evaluation note

Benchmark board：
https://github.com/initial-d/ml-quant-trading/blob/main/docs/benchmark_board.md

Research card / 边界说明：
https://github.com/initial-d/ml-quant-trading/blob/main/docs/research_card.md

成功、失败、很慢、有 caveat 的结果都欢迎。关键是保留 commit SHA、运行命令、环境、输出和限制条件。
```

## DeepSeek Harness Community

Title:

```text
DeepSeek Harness benchmark target for a PyTorch quant research pipeline
```

Body:

```text
I added an optional DeepSeek Harness workflow for `ml-quant-trading`, a PyTorch research stack for machine-learning multi-factor quantitative trading experiments:

https://github.com/initial-d/ml-quant-trading

The DSH recipe is here:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/deepseek_harness_recipe.md

Optional plugin:
https://github.com/initial-d/dsh-plugin-mlquant-benchmark

The target is intentionally bounded: run the benchmark/validation command, preserve the command, commit SHA, environment, raw table, generated JSON artifact, and caveats. This is an agent reproducibility exercise, not a live trading claim or investment recommendation.

I am looking for:

- DSH runs on different machines
- failures where the harness loses artifacts or caveats
- CUDA / Linux CPU benchmark reports
- suggestions for making the task harder without turning it into a black-box trading contest

Benchmark board:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/benchmark_board.md

Agent target:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/quant_agent_reproducibility_target.md
```
