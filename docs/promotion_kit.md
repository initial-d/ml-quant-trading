# Promotion Kit

Use this page when announcing releases, asking for benchmark results, or sharing the project
with research and engineering communities.

## Short Description

`ml-quant-trading` is an end-to-end PyTorch research stack for ML-enhanced multi-factor
trading, covering tensor factor computation, A-share bias correction, portfolio optimization,
and reproducible backtesting.

## Copy-Ready Posts

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
