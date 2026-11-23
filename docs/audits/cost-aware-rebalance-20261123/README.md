# Cost-aware actual-holdings rebalancing audit — 2026-11-23

This is **synthetic research-engineering evidence** for the planner and cash
ledger. [Raw report](report.json), about 150 KB, retains every control result,
decision/status, daily NAV, fees, oracle error, environment and command.
Implementation commit: `bffa8abf70c8227a07f0f2423a5c6d8968d3cd2e`.

## Equal-execution-fee controls

The synthetic panel contains 200 dates × 30 assets, seed `20261123`. Each account
starts with one million CNY at row 100 and evaluates the final 100 rows, using
the preceding 20 returns for covariance. The forecasts are constructed
independently of future realized returns; no intended alpha is present.

Both planners use the same risk aversion, position-value limit, cash-floor
constraint, permissions and optional monetary caps. Actual execution charges
**5 bps one-way in both cases**. The control sets the planning fee to zero;
the new planner includes the actual fee in both its objective and funding
constraint. Orders are executed through the same self-financing constrained
ledger, with feedback from actual holdings in both paths.

| Constructed case | Fees / initial NAV, aware | Fees / initial NAV, blind | Mean half-L1 turnover, aware | Mean half-L1 turnover, blind | Final NAV, aware | Final NAV, blind |
|---|---:|---:|---:|---:|---:|---:|
| Weak alternating 3 bps edge | 0.0000% | 2.0429% | 0.0000 | 0.2002 | 1.000000 | 1.038451 |
| Changing independent forecasts | 8.1730% | 8.7855% | 0.7453 | 0.8017 | 1.190584 | 1.206355 |
| Changing forecasts with prior-turnover caps | 6.7519% | 7.3752% | 0.6094 | 0.6858 | 1.216883 | 1.183172 |

Fees are cumulative currency fees divided by initial NAV, not annualized.
Turnover excludes entry and uses actual traded notional / pre-trade NAV.
The aware planner stays in cash for the weak edge rather than buying a forecast
below its entry fee. The control happens to earn positive realized returns in
that constructed path. Lower modeled costs **do not guarantee higher realized
returns**, as the first two rows show. All results are preserved.

Planning uses a 20% **pre-trade NAV monetary-value** position limit, with existing
drift allowed to remain without increasing that value, and a 5% pre-trade NAV
cash floor. Shrunk covariance and risk aversion one are shared. The third case
caps each asset's trade at 0.1% of strictly prior 20-row monetary turnover.
The blind planner has the same planning cash-floor constraint but can miss its
actual floor after fees it failed to model; that is part of the controlled
planning difference. This is not a calibrated market-capacity experiment.

All three aware paths optimized 100 rows with **zero failed holds**. Zero-failure
counts for the blind controls are also retained. Actual fills, cash, fees and
planning discrepancies are inspectable in the raw report.

## Independent checks

- Fifteen one-asset piecewise quadratic problems enumerate stationary points
  and boundaries on both absolute-cost regions. Maximum holding-fraction error
  versus the analytic global optimum: `8.59e-9`.
- Independent 60-digit Decimal breakpoint solving reconstructs full constrained
  cash ledgers from the selected targets. Maximum absolute currency error across
  NAV, cash, fees and trades: `1.31e-8`, with initial capital one million.
- Planning fees and executed fees reconcile. In the constrained CLI case,
  maximum holding-value discrepancy was `8.74e-11` currency units and maximum
  fee discrepancy `3.42e-13`.
- Future forecasts, returns, permissions and caps are rewritten from row 160;
  every retained earlier holding, trade, fee, NAV and plan remains byte-identical.
- The actual CLI saves 100 decisions and the complete account ledger, and its
  final NAV agrees with the direct API case.

The complete local suite passed **588 tests**. Tests also cover no-trade and
entry/exit thresholds, cash-paid entry, locked drifted holdings, cash floors,
notional caps, DPP caching, solver fallback, failed/warm-up exact holds, feedback
of actual marked positions, strictly prior covariance, monetary scaling,
input preservation and CLI artifact separation. Ruff and `git diff --check`
passed. CI runs the new command after its full synthetic pipeline.

## Reproduce and inspect

```bash
python scripts/audit_cost_aware_rebalance.py \
  --work-dir /tmp/mlquant-rebalance-reproduction \
  --json-out artifacts/cost-aware-rebalance-audit.json
python -m pytest
python -m ruff check src tests scripts/audit_cost_aware_rebalance.py
```

The task directory retains frozen inputs and complete `backtest.pkl` ledgers.
Those generated files are not committed; the repository contains the small raw
report and code to regenerate them.

Local environment: Python 3.9.6, NumPy 1.26.4, PyTorch 2.8.0, Apple M5 CPU,
macOS 26.6.2 arm64. No GPU, market downloads or model training occurred. Exact
audit and CLI commands are in the JSON. Tests/audit used:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-rebalance-20261123/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```

The same prefix was used for the recorded audit. Ruff used
`../.lint-venv/bin/python -m ruff check src tests scripts/audit_cost_aware_rebalance.py`.
Two existing final warnings remain: LibreSSL/urllib3 and CVXPY's large-parameter
DPP compilation notice.

Initial validation exposed two issues that were preserved rather than treated
as successes. Two analytic boundary checks first rejected tiny nonzero holdings
in flat quadratic-cost regions; evaluating feasible hold/max-sale baselines
eliminated these dominated solver candidates without loosening those tests.
Separately, a partially cleared old temporary dependency directory caused
AkShare namespace/import failures and fallback to NumPy 2.0.2's macOS matmul
warnings. The task rebuilt isolated NumPy 1.26.4, AkShare 1.18.88, Tabulate 0.9.0
and Mini-Racer 0.12.4 dependencies, leaving the shared environment untouched;
the full suite then passed.

Forecast calibration, source-specific missing histories/corporate actions and
market execution still require independent validation. Permitted fills use the
mark and same-row sale proceeds; there is no settlement, queue, impact, slippage
or lot-size model. See the [API/CLI contract](../../cost_aware_rebalance.md).
