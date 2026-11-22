# Capital-capacity sensitivity audit — 2026-11-22

This is **synthetic engineering evidence** for a frozen-portfolio capital
sweep. It tests the historical monetary-budget policy, coverage, cash and
fills; it does not estimate an investable strategy's market capacity or alpha.
[Raw report](report.json) is about 66 KB and retains every scenario's metrics,
daily normalized NAV/cash/target error, exact commands, environment, source
hashes and independent-oracle errors.

Implementation commit: `3f76580706e4c25692af0ae8274882ab2c01793f`.

## Frozen experiment

- 500 dates × 200 synthetic assets, `SyntheticConfig` seed `20261122`.
- Fixed random 20-name target group changed every seven rows, with equal
  weights totaling 90%. No alpha signal, model fitting or capital selection.
- Evaluation starts at row 300: 200 dates, 2021-02-25 to 2021-12-01. Each
  scenario starts in cash; the preceding history still informs liquidity.
- Shared per-asset cap: 1% of the prior 20 rows' monetary turnover, excluding
  the order row. All 40,000 evaluation cells had complete history in this run.
- One-way costs: 20 bps on fills. Synthetic amount and capital use declared
  CNY units, amount multiplier one. Cash interest is zero.
- Price/volume and supplied directional-band permissions apply to all accounts.
  The no-notional-cap reference uses the same permissions.

| Initial capital | Gross-notional fill ratio | Mean cash weight | Mean target L1 distance | Final NAV multiple |
|---|---:|---:|---:|---:|
| 1 million | 99.4459% | 10.0007% | 0.001398 | 0.904754 |
| 100 million | 25.3053% | 24.9155% | 0.562550 | 0.905685 |
| 10 billion | 0.3495% | 98.6193% | 0.898299 | 0.998007 |

The biggest account loses less largely because it barely builds the requested
portfolio and stays in cash. This is the diagnostic's point: read execution and
exposure alongside returns. These constructed numbers are not a profitable
strategy, a capacity estimate for real securities or an optimal capital rule.
Targets are shared; each account recomputes subsequent requests from its actual
holdings, so gross-request denominators differ across scenarios.

## Independent verification

The audit passed:

- Scalar 60-digit Decimal reconstruction of every evaluation-row cap. Maximum
  absolute cap error: `9.31e-10` currency units.
- Independent Decimal breakpoint target solving and full cash/fill ledgers
  at all three capitals. Maximum NAV error / initial capital: `1.56e-14`.
- Nonnegative stored holdings and cash, absolute-notional-cap enforcement.
- Cash/holding/fee conservation; maximum currency residual at 10 billion was
  `2.78e-6`, about `2.78e-16` of initial capital.
- Changing turnover from row 400 onward leaves orders and NAV through row 400
  **inclusive** byte-identical: its own observed turnover cannot set its cap.
- Consistent conversion of turnover and capital to thousands preserves actual
  weights and normalized economics.

The full local suite passed **548 tests** with the two existing LibreSSL/urllib3
and CVXPY DPP environment warnings. Ruff and `git diff --check` passed. Additional
tests cover hand-calculated fills, full pre-evaluation liquidity, missing versus
zero history, old-outlier removal, liberal-cap agreement with the same-gate
reference, unit/permutation/input preservation, bankruptcy, benchmark alignment,
invalid inputs and saved CLI reports without overwriting ordinary backtests.
CI also reuses its complete synthetic pipeline for a two-capital CLI sweep.

## Reproduce

From the implementation commit with development dependencies installed:

```bash
python scripts/audit_capital_capacity.py \
  --work-dir /tmp/mlquant-capacity-reproduction \
  --json-out artifacts/capital-capacity-audit.json
python -m pytest
python -m ruff check src tests scripts/audit_capital_capacity.py
```

The audit creates its own frozen synthetic files, calls the actual CLI in a
subprocess and verifies the saved full ledgers. Inputs and `capacity.pkl` remain
in the task directory; they are not committed. The repository retains only the
small inspectable report and reproduction code.

Local environment: Python 3.9.6, NumPy 1.26.4, PyTorch 2.8.0, Apple M5 CPU,
macOS 26.6.2 arm64; no GPU, paid APIs or market downloads. Exact audit/CLI commands
are in the JSON. The audit and tests used the existing isolated dependency
overlay, leaving the shared environment untouched:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-hac-20261118/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```

The same prefix was used for the recorded audit. Ruff used
`../.lint-venv/bin/python -m ruff check src tests scripts/audit_capital_capacity.py`.

## Interpretation

Participation here is a budget against **past average monetary turnover**, not
today's realized volume. Permitted orders assume mark-price fills; net sales can
fund same-row purchases. No calibrated impact, slippage, settlement, queue,
lot-size or real market capacity model is present. Vendor units must be checked
and converted explicitly. No archived public-data performance is revalidated.
See the [API/CLI contract](../../capital_capacity.md).
