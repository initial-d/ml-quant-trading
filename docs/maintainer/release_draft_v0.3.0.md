# v0.3.0 - Agent Quant Benchmark Challenge

`v0.3.0` frames `ml-quant-trading` as a concrete benchmark target for coding
agents, quant agents, and agent harnesses.

The release does not claim live tradability or deployable alpha. It packages a
realistic research workflow that agents can run, audit, and report without
needing broker credentials or proprietary market data.

## Headline

Can your coding agent reproduce a 213-factor, cost-aware quant research
pipeline without overstating the result?

## What changed

- Add the [Agent Quant Benchmark Challenge](../agent_quant_benchmark_challenge.md).
- Connect the challenge from the README top links and fast-path table.
- Point agent users to existing reproduction, benchmark, DSH, public-data, and
  private-evaluation report templates.
- Keep private evaluation explicitly redaction-safe for users who cannot expose
  strategy details, vendor data, or institutional infrastructure.
- Preserve the research boundary: synthetic and public-data outputs are
  reproducibility evidence, not trading recommendations.

## Challenge tracks

| Track | Best for |
| --- | --- |
| Zero-account smoke test | First-time users and package smoke tests |
| Protocol v1 CPU benchmark | Agent and hardware reproducibility reports |
| DeepSeek Harness run | DSH users who want tool-assisted benchmark validation |
| Public-data validation | Cost-aware public-data reports with caveats |
| Private evaluation note | Redacted institutional or proprietary-data evaluations |

## Suggested social copy

```text
I turned ml-quant-trading into an Agent Quant Benchmark Challenge:

Can your coding agent reproduce a 213-factor PyTorch quant pipeline, preserve
the evidence bundle, and avoid calling a backtest "alpha"?

Zero-account demo, fixed CPU benchmark, DSH path, public-data validation, and
redacted private-evaluation notes are all supported.

https://github.com/initial-d/ml-quant-trading
```

## Release checklist

- [x] Confirm CI is green on the release commit.
- [x] Confirm README links point to the challenge page.
- [x] Publish the GitHub release from this draft.
- [x] Open or update the GitHub Discussion challenge thread.
- [ ] Share one English post focused on agent reproducibility.
- [ ] Share one Chinese post focused on "AI agent 能不能真的复现量化研究".
- [ ] Invite DSH, coding-agent, and quant-research users to submit reports.

## Evidence boundary

This release is for research and engineering evaluation. It is not investment
advice, not a live-trading system, and not a claim that synthetic or public-data
benchmarks predict deployable trading performance.
