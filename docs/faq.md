# FAQ

## Is this a live trading bot?

No. This repository is a research and engineering baseline. It does not connect to a broker,
place orders, manage live risk, or provide trading recommendations.

## Does the repository include proprietary market data?

No. The paper used proprietary market data that cannot be redistributed. The public repository
therefore provides synthetic-data and public-data reproduction paths.

## Can I reproduce the exact paper numbers?

Not from the public repository alone, because the original proprietary data is not included.
You can reproduce the pipeline structure and qualitative behavior with synthetic data, and you
can add your own data through the loader interfaces.

## What should I run first?

Run:

```bash
make paper CONFIG=configs/small.yaml
```

Then open `notebooks/public_factor_ic.ipynb` if you want a public-data walkthrough.

## Why does the project use masks?

Financial panels have missing observations, suspensions, holidays, limit-up / limit-down cases,
and varying security histories. Mask-aware tensors keep factor computation explicit and safer.

## Why include legacy factors?

The legacy factor library provides a large practical testbed for tensorized factor computation.
It also gives researchers many small signals to ablate, inspect, and replace.

## How can I help without being a quant expert?

Good contributions include:

- benchmark results
- setup feedback
- documentation fixes
- notebook cleanup
- tests for edge cases
- clearer examples

## How should I report benchmark results?

Use the `Benchmark result` issue template and include:

- commit SHA
- command
- Python version
- PyTorch version
- CPU/GPU model
- CUDA availability
- printed result table

## Is public-data performance meaningful?

Public-data examples are mainly for workflow demonstration and reproducibility. They may have
survivorship bias, missing fields, adjusted-price differences, stale liquidity assumptions, and
transaction-cost limitations. Treat them as examples, not claims of deployable alpha.

## Can I add another data source?

Yes. A good adapter should produce a `Panel` with consistent `open`, `high`, `low`, `close`,
`volume`, `vwap`, and `mask` fields, and should document missing fields and adjustments.
