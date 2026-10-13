# System-One-Compatible Portfolio Decision Adapter

This page describes the preview adapter that exposes `ml-quant-trading` pipeline
state as typed portfolio-intent decisions.

The adapter is intentionally scoped to research:

- it answers asset, weight, tradability, liquidity, and risk questions;
- it does not create broker orders;
- it does not connect to accounts or exchanges;
- it does not run a background trading service;
- it does not change the existing paper claims or validation results;
- it is not a Jev-certified or externally hosted System One model.

## Why This Exists

The useful System-One-style framing for this project is not "summarize a
report." It is:

```text
pipeline state + typed portfolio questions -> typed portfolio-intent answers
```

The state is a compact object produced from the research pipeline. It can
contain factor values, masks, prediction scores, current holdings, baseline
target weights, tradability flags, volume or amount, risk estimates, and cost
assumptions.

The questions are constrained typed questions such as:

- should this asset be increased, held, decreased, or excluded?
- what is the research target weight?
- is this asset allowed by tradability and risk checks?
- what risk bucket should be assigned?
- what maximum participation rate is reasonable for the research simulation?

The adapter outputs research portfolio intent. It is not an execution layer.

## State Shape

A minimal state looks like:

```json
{
  "as_of": "2024-12-31",
  "universe": ["600519", "000001", "300750"],
  "predictions": {
    "horizon": "1d",
    "score": {
      "600519": 0.018,
      "000001": -0.006,
      "300750": 0.012
    },
    "rank": {
      "600519": 0.91,
      "000001": 0.23,
      "300750": 0.75
    }
  },
  "tradability": {
    "600519": {
      "tradable": true,
      "halted": false,
      "limit_up": false,
      "limit_down": false
    }
  },
  "liquidity": {
    "600519": {
      "amount": 1523000000,
      "adv20": 1340000000
    }
  },
  "portfolio": {
    "current_weight": {
      "600519": 0.02
    },
    "target_weight_baseline": {
      "600519": 0.04
    }
  },
  "risk": {
    "asset_volatility_20d": {
      "600519": 0.021
    },
    "portfolio_volatility": 0.132,
    "max_position_weight": 0.05
  },
  "cost_model": {
    "effective_cost_bps": 7,
    "max_participation_rate": 0.1
  },
  "evidence": {
    "data_source": "akshare",
    "claim_boundary": "public_data_validation_not_alpha"
  }
}
```

This state can be generated from existing artifacts later. The preview adapter
currently expects the state JSON directly.

## Questions

Questions can be provided as JSON or YAML:

```json
{
  "questions": [
    {
      "id": "asset_action",
      "type": "choice",
      "asset": "600519",
      "options": ["increase", "hold", "decrease", "exclude"]
    },
    {
      "id": "target_weight",
      "type": "score",
      "asset": "600519",
      "min": 0,
      "max": 0.05
    },
    {
      "id": "trade_allowed",
      "type": "noul",
      "asset": "600519"
    },
    {
      "id": "risk_bucket",
      "type": "choice",
      "asset": "600519",
      "options": ["low", "medium", "high", "blocked"]
    },
    {
      "id": "max_participation_rate",
      "type": "score",
      "asset": "600519",
      "min": 0,
      "max": 0.1
    }
  ]
}
```

If no question file is supplied, the adapter expands the default question set
over every asset in `universe`.

## Run It

```bash
mlquant systemone decide \
  --state artifacts/systemone/decision_state.json \
  --questions configs/systemone/portfolio_questions.json \
  --output artifacts/systemone/portfolio_decision.json
```

Without `--output`, the JSON is printed to stdout.

## Output Shape

The output is JSON:

```json
{
  "model": "mlquant-systemone-portfolio-decision",
  "model_version": "0.1.0-preview",
  "contract": "system-one-compatible-v0",
  "as_of": "2024-12-31",
  "answers": {
    "asset_action:600519": {
      "type": "choice",
      "asset": "600519",
      "value": "increase",
      "confidence": null,
      "probabilities": null,
      "rationale": "target weight compared with current weight after prediction, risk, liquidity, and cost constraints"
    },
    "target_weight:600519": {
      "type": "score",
      "asset": "600519",
      "value": 0.03,
      "confidence": null,
      "probabilities": null,
      "range": {
        "min": 0,
        "max": 0.05
      },
      "rationale": "research target weight capped by max position size and adjusted for risk and costs"
    }
  }
}
```

The preview adapter reports deterministic decisions and heuristic scores. It
does not publish calibrated probabilities. A future trained decision model
should add probability distributions only after label design, held-out
evaluation, and calibration checks.

## Release Boundary

Recommended release wording:

> This release adds a preview System-One-compatible portfolio-decision adapter
> for typed research portfolio intent. It does not add broker execution,
> account access, live trading services, or calibrated model probabilities.

That boundary keeps the project a reproducible research baseline while making
the pipeline callable as a typed decision model.
