# Constrained-fill ledger audit — 2026-11-21

This is **synthetic historical-execution engineering evidence**. It verifies
that blocked sales retain positions, replacement purchases use actual funds,
and charges follow simulated fills. It contains no market-data downloads,
strategy training or alpha claim. [Raw report](report.json) (about 87 KB) records
daily equity, fees and cash, plus detailed small-case fill ledgers.

Implementation commit: `f647a893f558b56e8b0a030a77e059de5a159c09`.
The report records hashes of both implementation files and each generated input.

## Results

| Constructed case | Shape | One-way fee | Constrained final NAV | Complete-fill final NAV | Maximum Decimal error* |
|---|---:|---:|---:|---:|---:|
| Locked exit, then held asset loses 30% | 4×2 | 20 bps | 0.695814 | 1.192824 | 3.33e-16 |
| Locked exit, then held asset gains 30% | 4×2 | 20 bps | 1.292226 | 0.795216 | 4.44e-16 |
| Mixed permissions, caps, changing targets and cash | 80×4 | 100 bps | 0.810643 | 0.550873 | 7.77e-16 |
| Random constraints on a synthetic panel | 500×200 | 5 bps | 1.023714 | 1.023524 | 6.66e-16 |

*Maximum absolute error across equity, fees, cash, trades, reference requests
and buy scaling. Currency quantities use initial NAV one; scaling is unitless.
These constructed outcomes compare assumptions, not strategy performance.
Both signs of the final-NAV difference are retained.

In the loss case, complete fills sell the original asset before its loss and
buy another asset that later gains 20%. The constrained policy blocks the
original sale, cannot fund the replacement and retains the subsequent loss.
The gain case reverses the returns and produces the opposite difference.

All four cases passed:

- independent scalar 60-digit Decimal breakpoint solving and fill ledgers;
- conservation of cash, marked holdings and actual fees;
- directional permission and absolute-notional-cap checks;
- sequential unfilled-cause reconciliation;
- unchanged prefix ledgers when future returns, targets, masks and caps change;
- bit-identical existing ledger arrays when all constraints are released.

The full local suite passed **512 tests** with two existing environment warnings
(LibreSSL/urllib3 and CVXPY DPP compilation). Ruff and `git diff --check` passed.
Tests also cover directional price bands, missing volume, pro-rata cash funding,
bankruptcy, scale/permutation controls, and saved OOS CLI ledgers across a halt
and resumption. CI exercises the constrained CLI after the synthetic pipeline.

## Reproduction and environment

From the implementation commit with development dependencies installed:

```bash
python scripts/audit_constrained_execution.py \
  --json-out artifacts/constrained-execution.json
python -m pytest
python -m ruff check src tests scripts/audit_constrained_execution.py
```

The local run used Python 3.9.6, NumPy 1.26.4, PyTorch 2.8.0, Apple M5 CPU,
macOS 26.6.2 arm64, CPU only, seed `20261121`. The JSON preserves the exact audit
command. To avoid modifying the shared development environment, audit/tests used:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-hac-20261118/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```

The same environment prefix was used for the recorded audit command. Ruff used
`../.lint-venv/bin/python -m ruff check src tests scripts/audit_constrained_execution.py`.

An initial local test run passed 59 checks and failed two oracle boundary
assertions: finite-precision Decimal cash could be slightly negative at roughly
1e-60, and independently reconstructed FP64 holdings could reach −8.7e-19 after
liquidation. The revised oracle floors only residual cash within 1e-45 and tests
use a 1e-15 reconstruction tolerance while still requiring **stored actual
weights and cash to be nonnegative**. The complete suite and audit then passed.
This numerical boundary handling is not an injected cash source.

## Interpretation limits

Costs are declared flat one-way rates on actual simulated notional. Cash earns
zero interest; net sale proceeds are usable on the same row. Permission and
notional constraints are constructed, not calibrated market-liquidity data.
Permitted trades assume fills at the mark. There is no settlement, queue,
slippage, price-impact, lot-size or corporate-action model. No historical
public-data result has been rerun under this policy.

See the [API and CLI contract](../../constrained_backtest.md) for target-reference
semantics, unfilled-order handling and the panel permission policy.
