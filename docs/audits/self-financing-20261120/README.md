# Self-financing cash-ledger audit

Implementation commit: `00bf65196c551d67aaa3273bdf7b50f360fe0b14`.
This is **synthetic historical-accounting engineering evidence**. Inputs are
constructed targets/returns and seeded NumPy arrays, not market observations.
All cases assume complete fills at the mark, one-way proportional fees, zero
cash interest and initial NAV one. No market data/API or GPU was used.

## Independent account reconstruction

[audit.json.gz](audit.json.gz) retains daily NAV, fees, cash and actual L1
turnover, the approximation's NAV path, full environment/commands, input/trade
fingerprints and all error measurements. [summary.json](summary.json) records
compressed/payload hashes and a compact case summary.

The independent oracle uses **50-digit Decimal arithmetic and exact buy/sell
breakpoints**. The production path uses FP64 bisection. Every post-trade NAV,
fee, cash balance and per-asset trade matches the oracle across 827 account rows
and 100,804 asset-row cells. Maximum absolute errors, in initial-NAV units:

| Case | Grid | One-way fee | NAV error | Fee error | Cash error | Trade error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 50/50 | 252 x 2 | 20 bps | 4.39e-14 | 1.84e-18 | 0 | 5.62e-16 |
| Changing targets and cash | 75 x 4 | 100 bps | 5.66e-15 | 6.77e-17 | 3.55e-15 | 3.28e-15 |
| Synthetic large panel | 500 x 200 | 5 bps | 5.04e-14 | 2.88e-19 | 5.25e-15 | 1.45e-17 |

For each case, changing future targets and returns preserves the earlier cash
and trade ledger exactly. Input weights still need an independent causality
check; this audit checks accounting timing, not strategy construction.

## How the conventions differ

| Case | Target-difference final NAV | Self-financing final NAV |
| --- | ---: | ---: |
| Fixed 50/50 | 0.998000 | 0.988034 |
| Changing targets and cash | 0.620930 | 0.622116 |
| Synthetic 500 x 200 | 0.982062 | 0.979428 |

The 50/50 case alternates +2%/−2% asset returns. At fixed post-trade fractions
the portfolio's gross return is zero, but restoring those fractions requires
actual purchases and sales. Mean half-L1 executed turnover is 0.01, while the
target-difference approximation reports zero ongoing turnover. Its missed fees
explain most of the endpoint difference. The entry fee also differs because
the new purchases and fees share the same cash budget.

Correction can raise or lower net NAV depending on price drift, cash and target
changes, as the complete table shows. These are convention comparisons, not
strategy-return improvements, calibrated market costs or an execution model.

## Tests and reproduction

[pytest.log](pytest.log) records **484 passing tests** (33 new cases). In
addition to ledger identities and timing, tests independently enumerate all
buy/sell partitions in 60-digit Decimal arithmetic. They check entry/exit,
drift at constant targets, partial cash, scale/permutation invariance, zero-fee
agreement, float32 budget rounding, asset/account total loss, CLI overrides,
saved ledgers and held-out window metadata. Ruff passes.

```bash
python scripts/audit_self_financing.py \
  --json-out artifacts/self-financing-audit.json
mlquant backtest --config configs/small.yaml --accounting self_financing
```

The calculation environment is Apple M5 CPU, macOS 26.6.2, Python 3.9.6,
PyTorch 2.8.0 and NumPy 1.26.4. Seed: `20261120`. Exact audit command and code
fingerprint remain in the compressed report. Exact full-suite command:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-hac-20261118/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```

See the [accounting contract](../../self_financing_backtest.md). The default
engine remains the archived target-difference approximation; this opt-in path
does not revalidate earlier public-data results. Fees require market-specific
calibration, and tradability, settlement, volume and impact are outside this
historical accounting model.
