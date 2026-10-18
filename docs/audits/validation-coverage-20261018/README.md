# Validation Execution Coverage — 2026-10-18

**A finite zero-return row can come from an MLP that never trained or invested.**
The full-pipeline report now displays factor coverage, training/prediction
samples and skip reasons, and portfolio cash/P&L exposure dates. The report
audit surfaces these warnings while preserving the failed run. This is evidence
about reporting quality, not investment performance or a paper-result update.

The signal-date causality correction was merged in
[PR #78](https://github.com/initial-d/ml-quant-trading/pull/78), main commit
`a0b33491266d578d63c300ad3d79832d237c40db`. The new coverage implementation and
replay script are committed at `fed2db53109cc0a419b6d02d0e4db6de308b5bdd`.
See [the protocol](../../public_data_validation.md#full-pipeline-execution-coverage).

## Synthetic engineering evidence

[compare_coverage.py](compare_coverage.py) runs the base and current full
pipeline on the same synthetic panel: 120 dates, 20 stocks, seed 7, no random
halts, all 213 registered factors. Actual synthetic dates are 2020-01-02 through
2020-06-17; config start/end are unused by the injected synthetic loader.
It compares the actual factor-library output, model training, portfolios,
summary metrics and cost sensitivity, without network access.

- Weights, metrics and cost sensitivity are identical before and after the
  reporting change: assertions and the input fingerprint are in
  [comparison.json](comparison.json).
- Joint factor-valid signal cells: **1,040 / 2,400**; forward-label cells:
  **1,020**. This short panel is deliberately inadequate for full-factor MLP
  training, not an estimate of real-data coverage.
- All three MLP splits are skipped. Training sample counts are **0, 240, 640**,
  below the unchanged minimum of **852** (`4 * 213`). Prediction samples existed,
  but no split trained, no finite prediction was produced, and all MLP portfolios
  stayed cash for all 120 dates.
- The equal-weight baseline has 118 P&L exposure dates; factor-mean variants
  have 51; MLP variants have zero. Full-panel metrics retain warm-up cash, and
  the report explicitly warns that these are not matched-window comparisons.
- The base report's existing audit passes. The new report has eight coverage
  warnings. Neither an audit pass nor these synthetic metrics establish alpha.

The generated [synthetic report](synthetic-summary.md) and
[raw report JSON](synthetic-summary.json) keep metrics beside the coverage
warnings. [coverage-run.txt](coverage-run.txt) preserves execution output.
Do not compare these results with the archived July market backtests.

## Verification and environment

- Full test suite: **146 passed**, [pytest.txt](pytest.txt). Tests cover empty
  factor masks and a normally trained MLP, with the existing causality tests.
- Ruff: passed, [ruff.txt](ruff.txt); `git diff --check`: passed.
- A regression test against the base fails because coverage is absent:
  [old-diagnostics-test.txt](old-diagnostics-test.txt).
- Exact commands: [commands.txt](commands.txt). Environment:
  [environment.json](environment.json), [pip-freeze.txt](pip-freeze.txt).
  Python 3.12.14, PyTorch 2.14.1, AkShare 1.19.1, macOS 26.6.2 arm64,
  Apple M5, 10 logical CPUs, 32 GiB RAM. CPU only; CUDA unavailable; MPS unused.
  Replay torch/interop/OMP/BLAS threads: 1. Costs: 5 bps plus 2 bps slippage;
  sensitivity: 0/7/15/30 effective bps. No bootstrap inference.
- Integrity: [checksums.txt](checksums.txt). Workspace paths in copied logs
  and reports are redacted to `<repo>`; original local artifacts remain under
  `artifacts/revalidation-20261018/`. Input market prices are not published.

## Public-data access failure, kept separately

The fixed probe from [the October 17 plan](../public-revalidation-20261017/plan.md)
was retried once per symbol on October 18 after merging the causality fix.
It again failed for both symbols, so the planned expanded run did not start.

| Symbol | Started, Asia/Shanghai | Outcome |
|---|---|---|
| `600519` | 2026-10-18 09:03:10 | Upstream disconnect; no data returned |
| `000001` | 2026-10-18 09:03:11 | Upstream disconnect; no data returned |

[provider-probe.json](provider-probe.json) records exact timestamps, durations,
source dates 2021-01-01 through 2025-01-01, daily `qfq`, and `retries=1`.
[provider-probe.txt](provider-probe.txt) preserves warnings and tracebacks;
[provider_probe.py](provider_probe.py) preserves the executed probe source,
copied from the October 17 probe with only its output directory changed.
Probe code commit: `a0b33491266d578d63c300ad3d79832d237c40db`.

The observed warning is `RemoteDisconnected('Remote end closed connection
without response')`; the loader raises `ValueError: No data returned`.
The underlying cause is unknown. No panel, coverage audit, market backtest,
new Sharpe result or proprietary paper experiment exists for this attempt.
Costs are not applicable to download-only probes. Synthetic evidence above
does not replace the blocked public-data run.

Coverage diagnostics reveal execution gaps; they do not repair evaluation
windows, add exchange execution modeling, or revalidate archived results.
