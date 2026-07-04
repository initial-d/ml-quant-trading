# Launch Playbook

This playbook turns repository improvements into a public launch that is useful rather
than noisy.

## Pre-Launch Checklist

- CI is green on `main`.
- README quick start is still accurate.
- The social preview image is uploaded in GitHub repository settings.
- Repository topics are set.
- The first release notes are ready.
- At least one benchmark command is easy to copy.
- The public launch checklist issue is open.

## Recommended Repository Settings

Description:

> End-to-end PyTorch research stack for ML-enhanced multi-factor trading, factor tensors,
> bias correction, portfolio optimization, and reproducible backtesting.

Website:

> https://arxiv.org/abs/2507.07107

Topics:

`quantitative-finance`, `algorithmic-trading`, `machine-learning`, `pytorch`,
`portfolio-optimization`, `alpha-factors`, `backtesting`, `research`,
`quant`, `factor-models`, `financial-machine-learning`

Social preview image:

> `docs/assets/social-preview.png`

GitHub currently requires uploading the social preview image through the repository
Settings page.

## Launch Post Order

1. Create a GitHub release.
2. Post a short technical launch note on GitHub issue/discussion.
3. Share the short social post.
4. Share the longer LinkedIn/community post.
5. Reply to questions and turn feedback into issues.

## Launch Message

Use a concrete, reproducible promise:

> Clone the repo and run a full synthetic factor-to-backtest pipeline with one command.

Avoid performance hype. Lead with reproducibility, engineering depth, and extensibility.

## First Week Goals

- 3 benchmark reports from different machines.
- 1 public-data reproduction report.
- 1 external issue that identifies unclear documentation.
- 1 external PR, even if it is a small docs fix.

## Where to Share Carefully

Share only where the audience is relevant and the post includes technical value:

- quantitative finance communities
- ML engineering communities
- research software communities
- university or student quant groups
- personal LinkedIn/X account

Do not post the same text everywhere. Adjust the ask to the community:

- For quant users: ask for factor/backtest feedback.
- For ML users: ask for tensor benchmark results.
- For students: ask for reproducibility feedback.
- For engineers: ask for API, tests, and packaging feedback.
