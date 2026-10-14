# Promotion Kit

Use this page when announcing releases, asking for benchmark results, or sharing the project
with research and engineering communities.

For a shorter visitor-facing link with copy-ready snippets, use
[`share_this_project.md`](../share_this_project.md).

## Short Description

`ml-quant-trading` is an end-to-end PyTorch research stack for ML-enhanced
multi-factor trading, covering tensor factor computation, A-share bias
correction, portfolio optimization, reproducible backtesting, and a preview
System-One/Jev-style portfolio-decision surface.

## Copy-Ready Posts

For the v0.3.0 Agent Quant Benchmark Challenge, use the dedicated
[Agent Challenge Outreach Kit](agent_challenge_outreach.md). It contains
English, Chinese, LinkedIn, Reddit, quant-community, and Zhihu-ready copy while
keeping the framing on reproducibility rather than trading claims.

### GitHub / Release Note

I open-sourced `ml-quant-trading`, a runnable research implementation for ML-enhanced
multi-factor trading.

It includes:

- 213 alpha/factor dimensions
- masked PyTorch tensor primitives for cross-sectional panels
- limit-up / limit-down / halt bias correction
- MLP / Transformer baselines
- Markowitz portfolio optimization
- vectorized backtesting and metrics
- synthetic and public-data demos

Repo: https://github.com/initial-d/ml-quant-trading
Paper: https://arxiv.org/abs/2507.07107

I would especially welcome benchmark results, public-data reproductions, and small PRs that
improve examples or tests.

### v0.3.1 System-One / Jev-Style Release Note

`ml-quant-trading` v0.3.1 adds a preview System-One-compatible portfolio
decision surface.

The useful quant framing is:

```text
pipeline state + typed portfolio questions -> typed portfolio-intent answers
```

The adapter builds decision state from local pipeline artifacts and answers
typed questions such as `asset_action`, `target_weight`, `trade_allowed`,
`risk_bucket`, and `max_participation_rate`. It can also run offline diagnostics
against realized forward returns kept out of the decision step.

This is a quant Jev-style exploration at the interface level: structured state
and typed questions in, typed answers out. It is not official Jev integration,
not a hosted System One model, and not broker/order/execution infrastructure.

Repo: https://github.com/initial-d/ml-quant-trading
Docs: https://github.com/initial-d/ml-quant-trading/blob/main/docs/systemone_adapter.md
Chinese article: https://github.com/initial-d/ml-quant-trading/blob/main/docs/article_zh_quant_jev_portfolio_decision.md

### 中文短帖：量化 Jev / Portfolio Decision

我在 `ml-quant-trading` 里补了一个 System-One/Jev-style portfolio decision
surface。

这次不是把外部模型塞进量化 pipeline，而是把 pipeline 本身封装成一个 typed decision
interface：

```text
因子、预测、mask、持仓、流动性、风险、成本
  + typed portfolio questions
  -> asset_action / target_weight / trade_allowed / risk_bucket 等结构化答案
```

它输出的是研究型 portfolio intent，不接券商、不生成订单、不做实盘服务。意义在于让
多因子 pipeline 的最终判断可以被调用、保存、比较和离线评估。

GitHub: https://github.com/initial-d/ml-quant-trading
中文长文: https://github.com/initial-d/ml-quant-trading/blob/main/docs/article_zh_quant_jev_portfolio_decision.md

### X / Short Social Post

I open-sourced `ml-quant-trading`: an end-to-end PyTorch stack for ML multi-factor research.

213 factors, masked tensor ops, bias correction, Markowitz optimization, backtesting, CI,
synthetic demo, and public-data notebook.

Repo: https://github.com/initial-d/ml-quant-trading

### LinkedIn

I have open-sourced `ml-quant-trading`, a research-oriented implementation of
ML-enhanced multi-factor quantitative trading.

The project focuses on reproducibility and practical engineering:

- a complete factor-to-portfolio-to-backtest pipeline
- 213 factor dimensions
- PyTorch tensor primitives for masked financial panels
- limit-up / limit-down / halt bias correction
- synthetic and public-data demos for users without proprietary data
- CI, tests, benchmark scripts, and contributor templates

I am looking for feedback from quant researchers, students, and engineers who care about
reproducible factor research and backtesting infrastructure.

https://github.com/initial-d/ml-quant-trading

### Community Post

I built an open-source research stack for ML-enhanced multi-factor trading and would like
feedback from people who work on factor research, backtesting, or tensorized finance data.

The repo includes a runnable synthetic-data pipeline, a public-data notebook, 213 factors,
masked PyTorch tensor primitives, bias correction, Markowitz optimization, and backtesting.

I am especially interested in:

- reproducibility feedback
- CPU/GPU benchmark results
- public-data examples
- factor-engine edge cases
- documentation improvements

Repo: https://github.com/initial-d/ml-quant-trading
Paper: https://arxiv.org/abs/2507.07107

## Launch Checklist

Before posting:

- Confirm CI is green.
- Confirm the README quick start still works.
- Create or update a GitHub release.
- Add a short benchmark result or screenshot.
- Link to one runnable notebook or command.
- Ask for one specific contribution.

After posting:

- Reply to every substantive comment.
- Convert repeated questions into docs.
- Convert good suggestions into issues.
- Thank benchmark contributors in release notes.

## Good Calls to Action

- "Run `make benchmark` and share your CPU/GPU result."
- "Try the public-data notebook and report data-source issues."
- "Suggest one missing factor family."
- "Open an issue if a reproduction step is unclear."
- "Send a PR with a small public-data example."

## Automation Ideas That Are Safe

- Weekly draft of one post from recent commits and issues.
- Monthly release notes generated from merged PRs and closed issues.
- Scheduled benchmark call asking users to submit results.
- A recurring maintainer checklist: CI, README quick start, issues, release notes.

Avoid automation that posts to external communities without review. It usually looks like spam
and damages trust.
