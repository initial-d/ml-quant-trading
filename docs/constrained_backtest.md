# Constrained historical rebalancing

A zero target is an intent to exit, not evidence that the position was sold.
If a stock halts, clearing its target must not turn the locked holding into
cash or allow a replacement purchase funded by that imaginary sale. This mode
keeps actual holdings and applies explicit buy/sell permissions to each order.

Reuse the saved panel and portfolio without retraining:

```bash
mlquant backtest --config configs/small.yaml --accounting constrained
```

`backtest_accounting: constrained` also works in a config, including for
`mlquant demo`. The CLI option overrides it. The default remains
`target_difference`; `self_financing` retains its complete-fill behavior.
Reports identify each mode. Historical snapshots are not silently revalidated.

## Execution policy

For each row, using only that row and prior account state:

1. Mark yesterday's **actual** holdings with today's simple returns.
2. Compute the hypothetical complete-fill request using the
   [self-financing post-fee NAV solve](self_financing_backtest.md).
3. Remove forbidden buy/sell directions, then clip permitted order notionals
   to optional per-asset caps.
4. Fill permitted sales. Scale permitted buys pro rata so purchases **and their
   one-way fees** fit cash plus net sale proceeds.
5. Retain blocked holdings. Charge fees only on fills and carry actual cash and
   positions into the next row.

Targets are reference post-fee weights under complete fills. Constraints can
prevent attaining them, including their intended cash weight. This deterministic
policy does not reoptimize around trapped positions. Unfilled requests expire;
next-row requests are recomputed from actual holdings. There is no pending-order
queue. Reopening returns therefore accrue to a retained position before that
row's new orders, using the panel's halt-safe return series.

## Permissions and caps

The CLI uses `panel.mask`, finite positive close and finite positive volume.
Unavailable cells block both directions. Supplied `limit_up` prices block
buys at or above the upper band; supplied `limit_down` prices block sells at or
below the lower band. Invalid cells in a supplied band block that direction.
Absent columns impose **no inferred ±10% rule**. Forward-label availability
and the feature-research limit mask are not used as order permissions.

A close at a price band does not establish actual queue fills. These rules are
a conservative declared close-mark policy; permissive directions still assume
full fills at that mark. They are not exchange execution evidence.

With your own point-in-time order eligibility and notional budgets:

```python
from mlquant.backtest import run_constrained_backtest

result = run_constrained_backtest(
    weights, returns,
    buy_allowed=buy_mask,           # boolean [T,N]
    sell_allowed=sell_mask,         # boolean [T,N]
    max_trade_values=notional_caps, # finite, nonnegative [T,N], optional
    initial_equity=1_000_000,
    costs_bps=5,
)
```

Each cap limits the absolute traded value of that asset on that row. Caps use
the same currency units as `initial_equity`; they are not portfolio weights or
volume participation fractions. Scaling account size requires scaling caps
too. The CLI applies no notional cap. None means unrestricted. No broadcasting
or numeric-to-boolean coercion is accepted for permissions.

To derive strictly lagged monetary caps from `panel.amount` and replay the same
targets across account sizes, use [the capacity API/CLI](capital_capacity.md).

## Inspect fills and missed targets

`backtest.pkl` retains the complete `ConstrainedResult`, including the original
cash ledger, actual weights, permission arrays, and:

- `requested_trade_values`: complete-fill reference requests.
- `trade_values`: actual simulated fills, positive buys and negative sales.
- `unfilled_trade_values`: signed requests minus fills.
- `permission_unfilled`, `cap_unfilled`, `cash_unfilled`: sequential causes
  that add up to the signed unfilled amount.
- `buy_fill_scale`: the pro-rata cash scaling after permissions and caps.

Summary reports include requested/executed/unfilled gross notionals, their fill
ratio, cause totals, and maximum L1 distance between actual and target weights.
The CLI records the gate policy and SHA256 of the OOS permission arrays.
Zero requests have fill ratio one by convention. Monetary totals are not
annualized; they use account currency units.

The [synthetic audit](audits/constrained-execution-20261121/README.md) includes
independent 60-digit Decimal ledgers, cash conservation, future-input mutation,
and bit-identical array checks when all constraints are released. Locked exits
can raise or lower realized equity relative to complete fills; the audit retains
both cases.

This is historical research accounting with same-row sale proceeds and zero
cash interest. It does not model T+1 settlement, queue priority, price impact,
slippage, corporate actions or lot sizes. Input signals still need causal
validation. Synthetic checks establish the stated policy, not deployable alpha.
