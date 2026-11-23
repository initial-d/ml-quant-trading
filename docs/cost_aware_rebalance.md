# Cost-aware rebalancing from actual holdings

The existing Markowitz path optimizes expected returns and risk, then pays costs
in the backtest. This opt-in path also puts actual proportional trading fees
inside the planning objective and budget. Small forecast improvements can leave
the account unchanged instead of automatically paying to replace a position.

```bash
mlquant rebalance --config configs/small.yaml \
  --currency CNY --data-category synthetic --data-source synthetic-demo
```

It reuses saved `panel.pt` and `predictions.pt`, without training a model or
requiring `weights.pt`. Forecasts must represent one-period expected **simple
returns**, matching the repository's training label. Ranks or probabilities
need a separately validated return calibration. Forecast quality is not proved
by using this planner.

The account starts in cash at the saved evaluation start. Every row first marks
the **actual previous holdings**, then estimates covariance from strictly prior
return rows, plans trades, pays fees and carries actual holdings/cash forward.
Today’s return can mark a held asset; it does not enter that row’s covariance
window. Price/volume and supplied directional-band gates apply in both planning
and fills. No future availability or label mask is consulted.

## Cash-paid convex plan

Normalize monetary quantities by pre-trade NAV. Let `a` be marked asset values,
`h` the planned asset values, `c` the one-way proportional fee, `mu` the forecast
and `Sigma` trailing covariance. The planner solves:

```text
maximize mu @ h - risk_aversion * h @ Sigma @ h - c * sum(abs(h - a))
subject to permitted lower/upper asset-value bounds
           sum(h) + c * sum(abs(h - a)) <= 1 - cash_floor
```

An absolute-value epigraph makes this convex. It pays fees from the account,
leaves residual cash and does not require full investment. The risk factor form
keeps CVXPY's problem DPP; each universe size reuses its cached problem. Supplied
trade caps divided by pre-trade NAV and directional permissions set the bounds.

`position_value_cap` and `cash_floor` are fractions of **pre-trade NAV**. The
position cap is not the legacy Markowitz post-fee target weight cap. A drifted
holding already above the value cap may remain but cannot grow above that
current value. This permits a locked position to be held without inventing a
sale. Post-fee portfolio fractions can be slightly higher after fees.

Planned holdings are converted to post-fee target fractions for the existing
self-financing execution ledger. Planning and execution use the same one-way
fee. Inspect the recorded maximum plan/fill and plan/fee errors. Numerical
feasibility repair stays within permitted bounds; feasible hold and maximum-sale
baselines replace a solver candidate only when their predicted utility is no
worse (strictly better for the maximum-sale baseline). This also handles flat
objective/no-trade boundaries without introducing a fee-threshold heuristic.

## Configure and inspect

```yaml
costs_bps: 5                 # shared by planning and execution
cov_lookback: 60             # excludes the order row; minimum two rows
rebalance:
  risk_aversion: 1.0
  position_value_cap: 0.05
  cash_floor: 0.0
  shrinkage: true
  solver: CLARABEL
```

This profile is separate from `portfolio:` and leaves the original Markowitz
and fixed-target backtests unchanged. It supports long-only holdings plus cash.
CLARABEL is the default; an unavailable solver falls back to SCS and the actual
solver/status is recorded. Insufficient history, infeasible cash floors and
failed solves **hold actual positions** and record a reason. Inspect optimized,
warm-up and failed-hold date counts before treating a run as successful.

For monetary participation budgets derived from prior turnover:

```bash
mlquant rebalance --config configs/small.yaml \
  --initial-capital 1000000 --participation-rate .01 \
  --liquidity-lookback 20 --amount-scale 1 \
  --currency CNY --data-category synthetic --data-source synthetic-demo
```

This requires `panel.amount` in verified monetary units. The cap uses the full
strictly prior window; missing history gives zero cap. It is a participation
budget against historical average amount, not today's realized volume.

Outputs go to `artifacts_dir/rebalance/` (or a separate `--output-dir`):
`summary.json`, `summary.md`, and the full `RebalancedResult` in `backtest.pkl`.
Reports preserve decisions, solver selection, baseline choice, failures, unit
conventions, source category, input hashes, environment and metrics. The ledger
also stores planned holdings/fees and actual requests/fills/cash. Ordinary
portfolio and backtest artifacts remain unchanged.

Python APIs are `RebalanceConfig`, `RebalanceOptimizer.solve()` in
`mlquant.portfolio`, and `run_rebalance_backtest()` in `mlquant.backtest`.
The rolling API takes full-history forecast/return/permission/cap arrays, a
lookback and optional evaluation start; monetary cap units match initial NAV.
The single-row solver takes current holdings and caps as pre-trade NAV fractions.

The [synthetic audit](audits/cost-aware-rebalance-20261123/README.md) includes
analytic global optima, independent Decimal cash ledgers, equal-execution-fee
controls, actual-holding feedback and future-input mutation. Reduced fees do
not guarantee higher realized returns; both directions remain in the report.
Covariance quality, forecast calibration, corporate actions, settlement, queue
priority, slippage, impact and lot sizes still require market-specific work.
Permitted fills use the mark and same-row sale proceeds. This is historical
research engineering evidence.
