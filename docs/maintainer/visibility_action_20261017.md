# October 17 Visibility and Onboarding Update

The maintainer observed a star count of 91 before today's API count of 90.
That is a one-star decrease (1.1%); its timing and cause are unknown. The saved
[GitHub snapshot](visibility_snapshots/2026-10-17.json) and
[traffic comparison](visibility_status.md#traffic-snapshot) establish the current
baseline, not the reason for a cancellation. This note contains repository
activity and documentation findings, not a new research or benchmark result.

## What the evidence supports

- Compared with the September 20 snapshot, rolling-window views fell 38.1%
  and unique visitors fell 24.1%. Clone counts fell only 6.4%, while unique
  cloners increased 5.6%. Traffic has cooled; adoption and user sentiment cannot
  be inferred from these counters.
- Main changed on October 16, after v0.4.1 was released on October 15. A lack
  of recent maintenance is not supported by the commit or release history.
- Chinese onboarding still receives visits, but its active invitation pointed
  to an August challenge. Start Here had the same outdated invitation.
- English and Chinese README headline returns were from the July 29 validation.
  The linked dashboard already warned that those numbers predate v0.4.1's
  mask-first and halt-safe return fixes, but the READMEs did not put that warning
  beside their headline numbers.

These documentation gaps are concrete friction to fix. They are not established
causes of the one-star decline, and fixing them does not guarantee star growth.

## Changes prepared today

1. Update Chinese README and Start Here invitations to the current Agent Quant
   Benchmark Challenge and the structured reproduction form.
2. Explain the optional `mlquantx[data]` install in the first-run path and give
   returning users an upgrade command.
3. Mark headline July backtests as archived, pre-v0.4.1 evidence in both
   READMEs. Preserve the original numbers and link the release protocol change.
4. Save a dated JSON traffic snapshot with exact API commands and limitations.
   Keep the recalled 91-star count explicitly separate from API observations.

The update is documentation-only and is based on main commit
`7490ab331029ded71fa9df6343f37782e127bcc2`. No market data was downloaded and no
backtest was re-run. The existing release and report artifacts remain the
sources for technical claims.

## Draft Chinese technical post

This is a draft for maintainer review; it has not been posted.

```text
涨跌停价格进入滚动因子后，再过滤当日输出，还能消除影响吗？

ml-quant-trading v0.4.1 修正了两个具体问题：

1. 在滚动因子计算前应用涨跌停 mask，避免限价观测继续影响后续窗口。
2. 对用 0 填充的停牌日，从最近可交易收盘价计算复牌收益，避免把停牌
   记成 -100%，再把复牌记成异常巨额收益。

这是研究管线的正确性修复。7 月归档的 A 股回测数字生成于修复前，
尚未按新口径重跑；它们不能作为当前版本的收益证据。

如果你愿意帮忙，先用无需行情账号的 synthetic demo 检查安装和管线：

python -m pip install --upgrade mlquantx
mlquant demo

欢迎提交成功或失败报告，并保留版本/commit、命令、环境和 warning。
公开数据验证另需安装 'mlquantx[data]'，与 synthetic 运行分开记录。

项目：https://github.com/initial-d/ml-quant-trading
修复说明：https://github.com/initial-d/ml-quant-trading/releases/tag/v0.4.1
报告：https://github.com/initial-d/ml-quant-trading/issues/new?template=reproduction_report.yml
```

For supporting details, link the [v0.4.1 release](https://github.com/initial-d/ml-quant-trading/releases/tag/v0.4.1)
and [PR #74](https://github.com/initial-d/ml-quant-trading/pull/74). They contain
the implementation changes, verification, and remaining limit-rule caveats.
Keep the request focused on one runnable contribution and answer any responses
before publishing another post.

## Next substantive research task

Revalidate a small A-share panel with v0.4.1 or later before expanding the
universe or replacing the archived dashboard. This requires a separate explicit
data-validation run. Preserve the commit, exact command, Python/PyTorch/OS and
hardware, universe and coverage, cost assumptions, masks, warnings, and raw
artifacts. Keep provider revisions and current-membership survivorship bias
visible. A synthetic run or CPU benchmark cannot replace this public-data
evidence.

Use the saved snapshot as the baseline for a later comparison of visitors,
reproduction reports, first-run failures, and stars. GitHub traffic does not
identify a visitor-to-star conversion rate or isolate a post's effect. This
update does not configure background monitoring or scheduled runs.
