# Agent Challenge Outreach Kit

Use this page when sharing the Agent Quant Benchmark Challenge with coding-agent,
LLM tooling, quantitative-finance, and reproducibility communities.

The positioning is deliberately narrow:

> Can an agent reproduce a documented quant research workflow, preserve the
> evidence bundle, and avoid overstating the result?

Do not frame the project as an alpha source, live-trading system, or investment
recommendation.

## Primary links

- Repository: https://github.com/initial-d/ml-quant-trading
- Challenge page: https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
- Discussion thread: https://github.com/initial-d/ml-quant-trading/discussions/66
- v0.3.0 release: https://github.com/initial-d/ml-quant-trading/releases/tag/v0.3.0
- Colab quickstart: https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb
- DeepSeek Harness recipe: https://github.com/initial-d/ml-quant-trading/blob/main/docs/deepseek_harness_recipe.md

## English short post

```text
I turned ml-quant-trading into an Agent Quant Benchmark Challenge:

Can your coding agent reproduce a 213-factor PyTorch quant pipeline, preserve
the evidence bundle, and avoid calling a backtest "alpha"?

Tracks:
- zero-account demo
- fixed CPU benchmark
- DeepSeek Harness path
- public-data validation
- redacted private evaluation notes

This is a reproducibility challenge, not investment advice.

https://github.com/initial-d/ml-quant-trading
```

## LinkedIn / Research Software

```text
I released v0.3.0 of ml-quant-trading as an Agent Quant Benchmark Challenge.

The question is intentionally practical:

Can a coding agent read a real quant research repository, run a documented
workflow, preserve the command/environment/artifacts, and report caveats without
turning a synthetic or public-data backtest into an unsupported trading claim?

The challenge includes:

- a zero-account synthetic smoke test
- a fixed protocol v1 CPU benchmark
- a DeepSeek Harness benchmark path
- cost-aware public-data validation
- redacted private-evaluation notes for users who cannot disclose proprietary
  data or strategy details

I am looking for successful runs, failed runs, setup friction, and agent
transcripts that are useful as reproducibility evidence.

Repo: https://github.com/initial-d/ml-quant-trading
Challenge: https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
```

## Reddit / Agent Community

```text
I made a quant-research benchmark for coding agents.

The task is not "make money". It is:

Can your agent reproduce a 213-factor PyTorch quant pipeline, keep the evidence
bundle intact, and avoid overstating backtest output?

It has a zero-account synthetic demo, fixed CPU benchmark, public-data
validation path, and a DeepSeek Harness recipe. Reports can be successful or
failed as long as they preserve commit SHA, command, environment, generated
artifacts, warnings, and caveats.

Challenge page:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
```

## Reddit / Quant Community

```text
I am looking for reproducibility feedback on an open-source PyTorch
multi-factor research stack.

The repo includes 213 factor dimensions, masked tensor primitives,
bias-correction logic, portfolio construction, vectorized backtesting, a
zero-account synthetic demo, public-data validation docs, and structured report
templates.

The new v0.3.0 challenge is aimed at coding agents and agent harnesses, but
human reports are welcome too. The useful contribution is not a headline
Sharpe; it is a reproducible run with environment, command, costs, caveats, and
artifacts.

Repo: https://github.com/initial-d/ml-quant-trading
Challenge: https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
```

## Chinese short post

```text
我把 ml-quant-trading 做成了一个 Agent Quant Benchmark Challenge。

问题不是「AI 能不能炒股赚钱」，而是：

AI coding agent 能不能复现一个真实的 213 因子 PyTorch 量化研究流程，保留
commit、命令、环境、artifact、warning 和 caveat，并且不把回测结果夸成 alpha？

支持：
- 零账户 synthetic demo
- 固定 CPU benchmark
- DeepSeek Harness 路径
- public-data validation
- 脱敏 private evaluation note

这是复现性挑战，不是投资建议。

https://github.com/initial-d/ml-quant-trading
```

## Zhihu / Long-form opener

```text
标题：AI Agent 能不能真的复现量化研究？我做了一个 213 因子挑战

这不是一个「AI 炒股」项目，也不是一个收益率展示。

我更关心一个朴素但很难的问题：当一个 coding agent 面对真实量化研究仓库时，
它能不能读懂文档、安装环境、运行固定 benchmark、保留证据包，并且在报告里
诚实地区分工程吞吐、public-data validation 和交易收益声明？

所以我把 ml-quant-trading 的 v0.3.0 做成了 Agent Quant Benchmark Challenge。

挑战包括五条路径：

1. 零账户 synthetic smoke test：`mlquant demo`
2. 固定 protocol v1 CPU benchmark
3. DeepSeek Harness benchmark workflow
4. 带交易成本、换手和 caveat 的 public-data validation
5. 给机构/私有数据用户准备的脱敏 evaluation note

我希望收集的不是漂亮结果，而是可复现记录：commit SHA、命令、环境、artifact、
warning、失败原因、成本假设和数据边界。

如果一个 agent 能跑完并正确报告失败，这比编一个好看的 Sharpe 更有价值。

Repo: https://github.com/initial-d/ml-quant-trading
Challenge: https://github.com/initial-d/ml-quant-trading/blob/main/docs/agent_quant_benchmark_challenge.md
```

## Where to post first

Prioritize communities where the challenge itself is on-topic:

- coding-agent and LLM tooling communities;
- DeepSeek Harness / plugin users;
- quantitative-finance engineering groups;
- reproducibility, research-software, and benchmark communities;
- Chinese technical communities that accept long-form engineering retrospectives.

Avoid generic investment, stock-picking, and trading-signal communities. They
will pull the discussion toward returns, which is not the project's strongest
or safest story.

## Follow-up rule

After posting, do not repost the same copy. Wait for a concrete event:

- a submitted benchmark;
- a failed run with useful logs;
- a DSH report;
- a public-data validation note;
- a fix or documentation change prompted by feedback.

Turn that event into the next update.
