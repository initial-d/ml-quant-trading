# Purged rolling factor-screening audit

Implementation commit: `c3ffea5` (complete SHA and file fingerprints in each
report). This is synthetic engineering evidence and planted IC timing controls.
No market data was downloaded, no factors were recomputed and no GPU was used.

The existing [500-date/200-stock/213-factor report](../../benchmarks/daily-factor-ic-20261026/README.md)
is screened with 200 original training rows, a one-row label-horizon gap and
50-row test windows. Six disjoint windows cover 299 test rows; the last contains
49 rows and is retained. Each fold corrects all 426 IC/RankIC hypotheses using
BY. **All six folds have no eligible factor**, so no OOS performance is inferred
from this frozen synthetic source. This is different from choosing the best
factor regardless of evidence.

## Independent timing and arithmetic checks

- All **2,556 training series** match an independently constructed dense
  Bartlett covariance matrix. Maximum standard-error difference: `3.47e-18`;
  maximum p-value difference: `3.89e-16`. All six full-family corrections and
  resulting selections match Statsmodels 0.14.4.
- Each fold's future values, including the purged boundary, were changed while
  preserving missing positions. All **six** training reports and frozen
  selections stayed identical. This checks the screening layer; the input
  factors, universe and preprocessing still require their own causality audit.
- A separate [planted control](control-source.json), seed `20261119`, has three
  constructed IC sequences on 305 rows with a five-row horizon. Four folds
  cover 200 test rows. All 24 training-series checks and four future-mutation
  checks pass. These sequences are not correlations computed from market data.
- A future-only winner is excluded from the initial selection and can qualify
  later after accumulating past evidence. A training winner that reverses keeps
  its training sign and produces first-test mean signed RankIC **−0.343761**.
  The evaluator does not reverse the sign using test outcomes.

[The retained full-suite log](pytest.log) records **451 passing tests**. The
28 new cases include label-overlap poisoning, future-only winners, reversals,
deterministic ties, partial windows, missing-date gaps, empty selections and
the distinction between selecting a factor and observing its future IC. Ruff
also passes. Calculation and testing used Apple M5 CPU, macOS 26.6.2, Python
3.9.6, PyTorch 2.8.0 and NumPy 1.26.4. IC contains no execution-cost model and
the checks incur no paid data/API or GPU cost.

## Retained evidence and reproduction

[summary.json](summary.json) records source, compressed-report, report-payload
and check hashes. The large original source stays in its existing location.

| Case | Full decisions and OOS records | Independent checks |
| --- | --- | --- |
| Frozen 213-factor synthetic source | [frozen-report.json.gz](frozen-report.json.gz) | [frozen-check.json](frozen-check.json) |
| Planted timing control | [control-report.json.gz](control-report.json.gz) | [control-check.json](control-check.json) |

From a source checkout:

```bash
mlquant factor-walk-forward \
  --input-report docs/benchmarks/daily-factor-ic-20261026/report.json.gz \
  --json-out artifacts/factor-walk-forward.json \
  --train-dates 200 --test-dates 50 --max-factors 10
python docs/audits/factor-walk-forward-20261119/check.py \
  --source docs/benchmarks/daily-factor-ic-20261026/report.json.gz \
  --report artifacts/factor-walk-forward.json --json-out artifacts/factor-walk-forward-check.json
python docs/audits/factor-walk-forward-20261119/make_controls.py \
  --json-out artifacts/factor-control-source.json
mlquant factor-walk-forward --input-report artifacts/factor-control-source.json \
  --json-out artifacts/factor-control.json --train-dates 100 --test-dates 50
```

`check.py` also accepts the retained `.json.gz` reports and requires audit-only
`statsmodels==0.14.4`. The exact calculation/check commands, Python and library
versions, CPU/OS, commit and code fingerprints are retained inside the reports.
Exact local full-suite command:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-hac-20261118/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```

Test IC summaries are descriptive per-factor measurements. They are not the IC
of a combined score or portfolio P&L. Each training fold's correction does not
control the entire sequence of folds, retrospective factor-family choice or
unreported parameter searches. These checks establish implementation and timing
consistency, not finite-sample calibration or market alpha.
