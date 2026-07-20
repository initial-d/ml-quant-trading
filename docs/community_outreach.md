# Community Outreach Targets

Use this page to share the project with relevant communities without spam.

## Primary Audience

- Quant researchers and students.
- ML engineers interested in tensorized financial panels.
- Research software users who care about reproducibility.
- Portfolio and backtesting tool builders.

## Recommended Communities

Post only where you already have an account and can reply to comments. Prefer
the `v0.2.0` validation release when the audience cares about reproducibility
more than launch novelty.

| Channel | Angle | Suggested ask |
|---|---|---|
| GitHub Issues / Discussions | Project launch and benchmark call | Submit CPU/GPU benchmark results |
| Hugging Face Papers | Paper and reproducibility artifacts | Try the public-data and synthetic reproduction paths |
| Awesome Quant | Curated open-source discovery | Evaluate the documented research stack |
| Hacker News Show HN | Runnable open-source research stack | Run the quick start and critique the design |
| QuantConnect forum | Reproducible factor and backtest research | Compare assumptions and public-data results |
| LinkedIn | Research software announcement | Feedback from quant/ML engineers |
| X / Twitter | Short technical launch | Star, benchmark, or try notebook |
| Quant StackExchange chat / communities | Factor-engine discussion | Edge cases and assumptions |
| University quant clubs | Student-friendly baseline | Run Start Here and report issues |

Reddit communities often remove direct repository promotion. Share there only when a
substantive, self-contained technical write-up is allowed by the current community rules;
the repository link should support the analysis rather than be the purpose of the post.

## Short Post

I released `ml-quant-trading` v0.2.0, an end-to-end PyTorch stack for ML multi-factor research.

It includes 213 factors, masked tensor ops, bias correction, MLP/Transformer baselines,
Markowitz optimization, vectorized backtesting, a public-data notebook, CI, and benchmark
tooling.

Repo: https://github.com/initial-d/ml-quant-trading
Release: https://github.com/initial-d/ml-quant-trading/releases/tag/v0.2.0
Paper: https://arxiv.org/abs/2507.07107

I am looking for CPU/GPU benchmark results and public-data reproduction feedback.

## Longer Community Post

I released `ml-quant-trading` v0.2.0, a research-oriented implementation of
ML-enhanced multi-factor quantitative trading.

The goal is to provide a reproducible baseline rather than a trading signal claim. The repo
contains:

- 213 factor dimensions
- mask-aware PyTorch tensor primitives
- limit-up / limit-down / halt bias correction
- MLP and Transformer baselines
- Markowitz portfolio construction
- vectorized backtesting and metrics
- synthetic and public-data demos
- public validation digest and Baostock A-share validation report
- CI, tests, benchmark scripts, and contribution templates

The paper used proprietary data, so the public repo focuses on synthetic and public-data
reproduction paths. I would especially appreciate:

- CPU/GPU benchmark reports
- public-data case studies
- factor-engine edge cases
- documentation and setup feedback

Repo: https://github.com/initial-d/ml-quant-trading
Release: https://github.com/initial-d/ml-quant-trading/releases/tag/v0.2.0
Paper: https://arxiv.org/abs/2507.07107

## Posting Rules

- Customize the first paragraph for each community.
- Disclose limitations clearly.
- Do not imply live trading profitability.
- Do not ask directly for stars before asking for useful feedback.
- Verify the current community rules before posting, especially for self-promotion.
- Reply to comments and convert repeated questions into docs.
