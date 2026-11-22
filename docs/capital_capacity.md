# Frozen-portfolio capital-capacity sensitivity

A strategy can request the same weights at one million and ten billion, but the
monetary orders differ by four orders of magnitude. A weight-only backtest hides
that difference. This entry point replays a frozen portfolio at each specified
account size with shared historical monetary turnover budgets, and records what
was filled, what remained in cash and how far actual holdings missed the targets.

After the synthetic pipeline has written its artifacts:

```bash
mlquant capacity --config configs/small.yaml \
  --capital 1000000 --capital 100000000 --capital 10000000000 \
  --participation-rate 0.01 --lookback 20 \
  --currency CNY --data-category synthetic --data-source synthetic-demo
```

This reuses `panel.pt`, `weights.pt` and optional `predictions.pt`; it does not
train or choose a capital based on the results. Each account starts in cash at
the saved evaluation start. By default it writes `artifacts_dir/capacity/`:

- `summary.json`: every scenario, coverage, source category, units, environment,
  costs, exact command, commit and input/implementation hashes.
- `summary.md`: the capital table, reference and assumptions.
- `capacity.pkl`: complete per-scenario ledgers, reference, monetary caps and
  liquidity coverage. This local generated artifact is not needed in a clone.

Ordinary backtest files and inputs remain in their original directory. An
explicit `--output-dir` must also use a separate directory.

## Monetary and timing contract

For asset `i` and order row `t`, with lookback `L` and participation budget `p`:

```text
cap[t,i] = p * mean(turnover[t-L:t,i])
```

The window excludes the order date. Budgets are computed on the full saved
history before slicing the evaluation period, so prior history can fund the
first evaluation row's cap. A full finite window is required. Initial warm-up
or NaN history gives cap zero and `liquidity_ready=False`; observed zero turnover
counts as valid history. Negative values and infinities fail. No partial-window
or share-volume fallback is used.

The CLI requires `panel.amount` and an explicit currency label. Convert vendor
units with `--amount-scale`, for example `1000` when the recorded amount is in
thousands of the account currency. Column names do not establish units; some
providers also quote share volume in lots or use adjusted prices. Declare and
verify the source's monetary convention before comparing account sizes.

Each scenario uses the [constrained fill policy](constrained_backtest.md):
directional gates, absolute trade caps, sales followed by cash-funded pro-rata
buys, and fees only on fills. Its uncapped reference uses **the same permission
gates** without monetary caps. That isolates adding the turnover budget from
already-declared price/volume permissions. Targets stay frozen; actual orders
are recomputed from each scenario's actual holdings, so their future requested
notionals can differ.

## Python API

```python
from mlquant.backtest import run_capacity_backtest

result = run_capacity_backtest(
    weights, returns, monetary_turnover,
    capitals=[1_000_000, 100_000_000, 10_000_000_000],
    start_index=300, lookback=20, participation_rate=0.01,
    buy_allowed=buy_mask, sell_allowed=sell_mask, costs_bps=5,
)
```

All market/target/permission arrays cover the full matching `[T,N]` grid; optional
benchmark returns also cover full history. Capital values must be unique,
finite and positive, and use the same currency as turnover. The function
preserves the supplied order of scenarios. No implicit best-capital selection,
resampling, model fitting or portfolio reselection occurs.

Each `CapacityResult.rows` entry includes final NAV and normalized multiple,
gap to the uncapped reference, gross-notional fill ratio, average cash weight,
average target L1 distance, capital-normalized cap misses and fee totals, and
the complete original metrics. Cash weights average only positive-NAV dates;
their count and bankruptcy index remain visible. Zero requests have fill ratio
one by convention. All `scenarios`, `reference`, `trade_caps` and
`liquidity_ready` arrays remain inspectable.

This is sensitivity to participation in **lagged mean turnover**, not a claim
about fills as a fraction of today's realized volume or a proven maximum
deployable capital. Market impact, slippage, settlement, queue priority and lot
sizes are not modeled; permitted orders fill at the mark and use same-row sale
proceeds. Capacity-limited portfolios may lose less simply because they stayed
in cash. Read exposure and target error alongside returns.

The [500×200 synthetic audit](audits/capital-capacity-20261122/README.md) retains
all three capital scenarios, independent Decimal budgets/ledgers and turnover
mutation/unit-scaling checks. It is engineering evidence; no public-data market
snapshot has been revalidated under this policy.
