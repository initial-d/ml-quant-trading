# v0.3.1 - System-One Portfolio Decision Adapter

`v0.3.1` introduces a preview System-One-compatible decision surface for
`ml-quant-trading`.

The main idea is to expose a quant research pipeline in a Jev-style shape:

```text
portfolio state + typed questions -> typed portfolio-intent answers
```

This is a new direction for the project: instead of only running a factor
pipeline and reading a report, downstream tools can ask typed asset-level
questions about actions, weights, tradability, liquidity, and risk.

## Highlights

- Added `mlquant.systemone`, a preview portfolio-decision adapter.
- Added `mlquant systemone decide` for local typed decisions over structured
  portfolio state.
- Supported default and custom typed questions:
  - `asset_action`
  - `target_weight`
  - `trade_allowed`
  - `risk_bucket`
  - `max_participation_rate`
- Reworked the README to emphasize the runnable baseline, mask-first factor
  contribution, cost-aware validation, and System-One portfolio decision module.
- Added `docs/systemone_adapter.md` with the state contract, question contract,
  output shape, and release boundary.
- Added a Chinese positioning article for the quant Jev-style portfolio
  decision surface:
  `docs/article_zh_quant_jev_portfolio_decision.md`.
- Synchronized package metadata, `mlquant.__version__`, and citation metadata at
  `0.3.1`.

## Why This Matters

This release is an early exploration of what a quant-native Jev-like model
interface can look like.

The project does not ask a language model to write a trading opinion. It exposes
the existing factor, prediction, tradability, liquidity, risk, and cost state as
typed decisions:

```json
{
  "asset_action:600519": {
    "type": "choice",
    "value": "increase"
  },
  "target_weight:600519": {
    "type": "score",
    "value": 0.04
  },
  "trade_allowed:600519": {
    "type": "noul",
    "value": true
  }
}
```

That makes the pipeline callable as a structured decision model while preserving
the research boundary.

## Jev / System One Relationship

The adapter is Jev-style in interface shape:

- structured state in;
- typed questions in;
- typed answers out;
- no free-form recommendation text required.

It is not an official Jev integration, not Jev-certified, and not a hosted
System One model. It also does not publish calibrated probabilities yet.
Probabilities should only be added after label design, held-out evaluation, and
calibration checks.

## Research Boundary

This release does not add:

- broker execution;
- account access;
- exchange connectivity;
- background trading services;
- live order generation.

The output is research portfolio intent, not live execution. `ml-quant-trading`
remains a reproducible quant research baseline, not an investment product or
live trading system.

## Try It

```bash
mlquant systemone decide \
  --state artifacts/systemone/decision_state.json \
  --output artifacts/systemone/portfolio_decision.json
```

See [docs/systemone_adapter.md](../systemone_adapter.md) for the full state and
question contract.

Chinese explanation:
[量化 Jev 不一定要接入模型：把多因子 Pipeline 封装成 Portfolio Decision Surface](../article_zh_quant_jev_portfolio_decision.md).

## Verification

- `python -m pytest tests/test_systemone.py tests/test_cli.py -q`
- `python -m ruff check src/mlquant/systemone.py src/mlquant/cli/main.py tests/test_systemone.py`
- `python -m pytest -q`

All tests passed on the release workspace. The full test run retained the
existing CVXPY DPP warning in Markowitz tests.
