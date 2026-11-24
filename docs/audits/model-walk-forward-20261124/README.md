# Model-refitting audit — 2026-11-24

Evidence category: **synthetic engineering and planted-process diagnostics**.
This run verifies sequential forecast provenance and downstream accounting;
it provides no new market performance, investment or alpha evidence.

Source commit: `3fab92a5785c52b59147319428a227a0405b2c42`.
Python 3.9.6, PyTorch 2.8.0, NumPy 1.26.4, Apple M5,
macOS 26.6.2 arm64, CPU, GPU unused. The audit subprocess fixes Torch intra/inter-op
threads to one; BLAS/solver threading is not fixed and timings are run observations,
not a controlled performance benchmark.

## Full 213-factor path

[Raw manifest](full-pipeline.json), [complete output](full-pipeline.log) and
[exact runner snapshot](full-pipeline.py) preserve configuration, commands,
environment, input/checkpoint hashes and both backtest reports. The runner uses
the repository's complete `configs/small.yaml`, changing only artifact locations.
No provider data is fetched. Raw tensor inputs, predictions, model checkpoints
and complete account pickles remain under `/tmp/mlquant-walk-forward-20261124/`;
they are not committed to the repository.

Command used locally (the dependency overlay is isolated from the shared venv):

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-rebalance-20261123/check-deps:$PWD/src \
../bench-venv/bin/python /tmp/mlquant-walk-forward-20261124/full-pipeline.py
```

To replay with a normal `.[dev]` installation from the repository root, copy the
runner to its recorded location and execute it. Its fixed output directory will
be updated; use a separate checkout/environment for an independent reproduction.

```bash
mkdir -p /tmp/mlquant-walk-forward-20261124
cp docs/audits/model-walk-forward-20261124/full-pipeline.py \
  /tmp/mlquant-walk-forward-20261124/full-pipeline.py
python /tmp/mlquant-walk-forward-20261124/full-pipeline.py
```

| Check | Observed result |
|---|---:|
| Panel / factors | 500 dates × 200 assets × 213 factors |
| Fresh models | 4 |
| First OOS row / scheduled OOS dates | 300 / 200 |
| Signal dates with usable predictions | 200 |
| Usable prediction cells / scheduled cells | 29,759 / 40,000 |
| Forecast training elapsed | 3.57 s |
| Fixed-target portfolio construction elapsed | 73.15 s |
| Constrained fixed-target evaluation | 200 dates |
| Cost-aware planning/execution elapsed | 103.80 s |
| Optimized / failed-hold cost-aware dates | 200 / 0 |
| Maximum planned vs filled holding value difference | 4.16e-17 |
| Maximum planned vs executed fee difference | 3.25e-19 |

The label horizon and outer gap are one original row. Each rolling history has
299 original rows; each inner validation gap removes at least one observed
signal date. All four training and validation label endpoints precede their
corresponding test start. Prediction coverage uses current factors only; reduced
cell coverage is preserved rather than reported as 40,000 valid forecasts.
Both backtests use 5 bps one-way fees; account initial NAV is one unit. Mark-price
execution and same-row sell funding remain the existing research assumptions.
The slow portfolio/optimizer timings are retained alongside fast training.

## Frozen-checkpoint controls

Exact command:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-rebalance-20261123/check-deps:$PWD/src \
../bench-venv/bin/python scripts/audit_walk_forward_training.py \
  --output-dir /tmp/mlquant-walk-forward-20261124/controls
```

[Raw controls](controls.json) retain every fold, loss history, chosen epoch,
prediction/checkpoint hashes, elapsed time and outcome. Cases are fixed in code:
180 dates × 30 assets × 2 Gaussian features, a planted coefficient of +0.02,
a switch to -0.02 at row 90 in the sign-switch case, and zero coefficient in the
no-signal case, with 0.005 Gaussian noise. Seeds are 11 and 29. Histories are 60
rows, test windows 30, with both rolling and expanding refits. Every model uses
12 epochs, hidden size 8, and the same training settings. No winning seed,
window or case is selected for presentation.

The frozen control is **exactly the first refit's selected validation checkpoint**,
predicted one date at a time over the same available cells and evaluated on the
same label-eligible dates. It therefore shares the initial model with refitting
and does not benefit from a different hindsight checkpoint selection.

| Process | Seed | Window | Refit MSE | Frozen MSE | Refit mean IC | Frozen mean IC |
|---|---:|---|---:|---:|---:|---:|
| Stationary | 11 | rolling | 0.0014864 | 0.0022022 | 0.288 | 0.051 |
| Stationary | 11 | expanding | 0.0008881 | 0.0022022 | 0.555 | 0.051 |
| Stationary | 29 | rolling | 0.0058972 | 0.0008637 | 0.255 | 0.490 |
| Stationary | 29 | expanding | 0.0010943 | 0.0008637 | 0.709 | 0.490 |
| Sign switch | 11 | rolling | 0.0017683 | 0.0023605 | 0.120 | -0.027 |
| Sign switch | 11 | expanding | 0.0015891 | 0.0023605 | -0.171 | -0.027 |
| Sign switch | 29 | rolling | 0.0065170 | 0.0018586 | 0.151 | -0.231 |
| Sign switch | 29 | expanding | 0.0025952 | 0.0018586 | -0.014 | -0.231 |
| No signal | 11 | rolling | 0.0013237 | 0.0019474 | -0.001 | 0.007 |
| No signal | 11 | expanding | 0.0007288 | 0.0019474 | -0.004 | 0.007 |
| No signal | 29 | rolling | 0.0059399 | 0.0007183 | 0.002 | 0.016 |
| No signal | 29 | expanding | 0.0013281 | 0.0007183 | 0.002 | 0.016 |

Refitting is not uniformly better. Initialization and finite training can produce
large MSE; higher IC can coexist with worse MSE, and no-signal IC stays near zero.
These small, deliberately planted controls establish inspectable behavior, not
a tuned forecast model, automatic drift adaptation, or profitable strategy.
There is no trading/cost model in this supervised-quality comparison.

## Independent and regression checks

The full local suite passes: **611 tests**, two existing LibreSSL/CVXPY warnings.
Ruff on `src`, `tests` and the audit script, and `git diff --check`, pass.

- An independent one-step Adam gradient calculation matches the first fold's
  masked-training forecast; validation labels do not enter its gradient.
- A controlled three-epoch validation sequence proves prediction reloads the
  best checkpoint rather than the last epoch.
- Original-row gaps cover one- and three-period horizons, sparse dates, rolling
  and expanding histories, and the final partial test window.
- Changing future labels/features/masks preserves earlier predictions and the
  first checkpoint hash exactly. Outer purge and dataset padding are unused.
- Current predictions remain available when future label availability is false;
  invalid factors remain explicitly uncovered.
- IC/RankIC training uses complete date batches. Torch RNG states restore and
  same-environment seeded runs reproduce predictions and checkpoint hashes.
- Insufficient data, invalid split settings and nonfinite losses fail explicitly.
  Failed CLI refitting preserves previously published predictions.
- The actual CLI connects refits to both backtest paths and rejects mismatched
  forecast/portfolio artifacts after retraining.

Test command used:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-rebalance-20261123/check-deps:$PWD/src \
../bench-venv/bin/python -m pytest
```

The implementation does not correct a retrospectively chosen universe, future
preprocessing, misdeclared labels, or hyperparameters selected on combined OOS
outcomes. Earlier test dates legitimately enter later histories; an independent
outer holdout is needed after retrospective tuning.
