# v0.2.4 Launch Kit — Metric Clarity

Use one channel at a time. Reply to technical questions before opening another
thread. Zhihu and JoinQuant are intentionally excluded because the project has
already been posted there.

Canonical links:

- Repository: <https://github.com/initial-d/ml-quant-trading>
- Technical story: <https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md>
- Pull request: <https://github.com/initial-d/ml-quant-trading/pull/47>
- Release: <https://github.com/initial-d/ml-quant-trading/releases/tag/v0.2.4>
- Colab: <https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb>

## Hacker News

### Title

Show HN: A contributor found our backtest cost metric was easy to misread

### Text

An external contributor noticed that our `cost_drag` value was cumulative over
the entire backtest, while the metrics beside it (`ann_return`,
`gross_ann_return`, `ann_vol`) were annualized.

The calculation was correct, but the label made a wrong comparison look
reasonable. Doubling the backtest length roughly doubled the reported cost drag
while turnover and cost per year stayed flat.

We renamed the primary field to `cost_drag_cumulative`, kept a compatibility
alias for archived reports, added unit/time-basis documentation, and pinned the
behavior with regression tests. I wrote up the evidence and why simply
subtracting annualized returns does not produce an exact cost identity.

I would value criticism of the reporting contract and examples of similar unit
ambiguities in other backtest engines.

Story: https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md

## Reddit — r/algotrading or r/quant

### Title

Our cost metric was mathematically correct but semantically misleading

### Text

An external contributor caught a subtle reporting issue in my open-source
backtest stack. `cost_drag` was the arithmetic sum of costs over the full run,
but it appeared beside annualized return and volatility metrics.

Same configuration, different durations: 600 periods produced 0.1592 cost
drag, 1,200 produced 0.3112, and 2,400 produced 0.6351. Cost per year and
turnover stayed roughly flat. The implementation was doing what it said in the
code, but the table did not communicate the time basis.

The fix is intentionally boring: call it `cost_drag_cumulative`, preserve the
old key for compatibility, document every metric's unit/time basis, and add
duration-scaling tests. We did not invent an exact-looking “annualized cost
drag,” because arithmetic cost sums do not reconcile exactly with compounded
annual returns.

What other backtest metrics have you seen misread because their time basis was
implicit?

Technical write-up and code: https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md

## V2EX

### 标题

[分享创造] 一个计算正确、却很容易被读错的回测成本指标

### 正文

项目最近收到一个外部贡献者 PR。他发现回测表里的 `cost_drag` 是整个区间的累计
成本，但它旁边的 `ann_return`、`gross_ann_return` 和 `ann_vol` 都是年化指标。

同一配置只改变回测长度，600、1200、2400 个周期的成本分别约为 0.1592、
0.3112、0.6351；换手和每年成本基本不变。计算没有错，但命名会诱导读者把累计量
和年化量直接比较。

现在主字段改成了 `cost_drag_cumulative`，旧字段保留兼容；同时给每个指标补了
单位和时间口径，并增加长度缩放测试。没有硬造一个“年化成本拖累”，因为算术成本
和复合年化收益并不能精确对账。

想请大家帮忙挑刺：你还见过哪些“代码算对了、接口却容易让人读错”的指标？

技术记录与代码：
https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md

## 掘金

### 标题

回测指标的隐藏陷阱：累计交易成本为什么不能直接对比年化收益

### 导语

一次外部代码审查暴露了一个典型工程问题：公式和实现都正确，并不代表报告不会
误导。本文用长度缩放实验解释累计成本、年化收益和复合路径之间的差异，并展示
如何在不破坏旧报告的前提下修复字段契约。

正文直接改编
[`backtest_cost_drag_story.md`](backtest_cost_drag_story.md)，保留实验表格、迁移
代码和“审计自己的回测”章节。结尾只放技术故事和仓库链接，不使用收益截图。

## X / LinkedIn

An external contributor found a subtle issue in my backtest project:

`cost_drag` was cumulative over the full run. The metrics beside it were
annualized.

The math was correct. The interface made a wrong comparison look reasonable.

We renamed the field, kept backward compatibility, documented every metric's
unit/time basis, and added duration-scaling tests.

Open-source review working as intended:
https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md

## Posting Order

1. Publish the GitHub release and Discussion first so every external link has a
   stable landing page.
2. Hacker News or Reddit next — choose one, not both on the same day.
3. Reply for 24 hours and incorporate legitimate corrections.
4. Publish V2EX or Juejin 48 hours later with a rewritten opening.
5. Use X or LinkedIn as a short pointer, not a duplicate long post.

Before posting, re-check each community's current self-promotion rules. Ask for
technical criticism or reproduction evidence, never votes or stars.
