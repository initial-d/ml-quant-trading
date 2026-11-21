# Self-financing target-weight accounting

Constant target weights do not imply zero trades: price changes drift the
existing asset fractions. Restoring 50/50 after one asset gains and another
loses requires selling the winner and buying the loser. This mode measures
those trades and pays their fees from portfolio NAV.

```python
from mlquant.backtest import run_self_financing_backtest
result = run_self_financing_backtest(weights, returns, costs_bps=5)
```

Reuse the CLI's saved portfolio and panel without retraining:

```bash
mlquant backtest --config configs/small.yaml --accounting self_financing
```

Or set `backtest_accounting: self_financing` in the config, including for
`mlquant demo`. The CLI option overrides the config. Default `target_difference`
keeps the existing approximation for archived compatibility. Reports explicitly
record the chosen mode; different modes require separate result labels.

## Cash and timing contract

Row `t` first earns the return of holdings from `t-1`, then pays fees and applies
the targets for `t -> t+1`. The account starts entirely in cash. Targets are
fractions of **post-fee** NAV. With marked holdings `H`, marked NAV `V`, target
fractions `w` and proportional one-way rate `c`, solve:

```text
V_after = V - c * sum(abs(w * V_after - H))
trades  = w * V_after - H
cash_after = (1 - sum(w)) * V_after
```

Thus purchases and fees use existing funds. Entry and exit are charged too.
`costs_bps` is a **one-way fee per absolute traded notional**, not a total fee
for a buy-and-sell round trip. The default cash balance earns zero interest.
The initial pre-trade NAV also enters drawdown measurement, so entry-fee losses
are visible even for a one-row run.

The mode supports long-only positions plus cash. Inputs must share finite,
nonempty `[T,N]` grids; simple returns cannot be below −100%. Budget overruns up
to `1e-6` are normalized on a copy to accommodate float32 portfolio rounding and
are recorded. Larger overruns and negative targets fail. An asset with a
recorded −100% return cannot be bought again; a zero-NAV account is absorbing.

## Inspect the ledger

The CLI saves the complete `SelfFinancingResult` in `backtest.pkl`, including
marked pre-trade weights, executed trade values, fees, cash, pre-trade NAV,
post-trade NAV and actual post-trade weights. `trade_values`, `fees` and the
equity arrays use the same monetary unit as `initial_equity` (default one).

- `turnover_l1`: each row's gross traded notional / marked pre-trade NAV.
- `turnover`: mean half-L1 executed turnover, excluding initial entry.
- `turnover_including_entry`: the same mean including entry.
- `fees_total`: monetary fees; `fees_fraction_initial` divides by initial NAV.
- `cost_drag_cumulative`: sum of each fee / that period's starting NAV; it is
  not an annualized figure or the same quantity as `fees_fraction_initial`.
- `max_self_financing_residual`: maximum floating-point residual of
  `V_before - V_after - fee`, in monetary units.

This is historical accounting with complete fills at each row's mark, not a
tradability, settlement, order-book, market-impact or capacity simulator.
For blocked sells, directional permissions and cash-funded partial purchases,
use the opt-in [constrained mode](constrained_backtest.md).
Factor selection and input weights still require their own causality checks.
Cost assumptions need market-specific calibration before empirical use.

The [cash-ledger audit](audits/self-financing-20261120/README.md) retains
independent Decimal reconstructions, future-mutation controls and the full
synthetic comparison, including cases where the difference changes direction.
