# Mean IC inference and factor search correction

`mlquant factor-inference` consumes a frozen daily factor report and adds
Newey–West mean standard errors, two-sided p-values, pointwise confidence
intervals and joint IC/RankIC multiple-testing correction. It reuses saved
series, so inference does not require recomputing the 213-factor tensor or
fetching market data.

```bash
python scripts/eval_factor_ic.py --config configs/small.yaml \
  --json-out artifacts/daily-ic.json
mlquant factor-inference --input-report artifacts/daily-ic.json \
  --json-out artifacts/factor-inference.json
```

The input can also be `.json.gz`. Every factor needs a unique `name`,
`daily_ic` and `daily_rank_ic` on the same original panel-row grid. Missing dates
use `null`. Optional `total_dates` and `valid_dates` are checked for consistency;
IC and RankIC share the original eligibility mask. `label_horizon` defaults to
one panel row and should be included for multi-row labels.

## Time dependence

For observed daily IC values, let `n` be the observed count, `mu` their mean,
and `u[t] = IC[t] - mu` on observed dates and zero on missing dates. The zero is
a missing score contribution, not an imputed zero IC. Missing dates stay in the
original grid, so observations separated by missing dates remain separated when
computing lag terms.

The mean variance is `u' K u / n²`, multiplied by `n/(n-1)`, where
`K[i,j] = max(1 - |i-j|/(L+1), 0)`. A moving-sum quadratic form evaluates this
positive-semidefinite Bartlett kernel without subtracting nearly cancelling
positive and negative covariance terms. The mean and unannualized IC IR keep
their descriptive interpretation. Serial dependence can increase or decrease
the standard error.

Default `L = floor(4*(n/100)^(2/9))`, at least `label_horizon-1`, capped at grid
length minus one. Specify a lag and inspect sensitivity when longer dependence
is plausible:

```bash
mlquant factor-inference --input-report artifacts/daily-ic.json \
  --json-out artifacts/factor-inference-lag20.json --max-lag 20
```

An explicit lag shorter than `label_horizon-1` is rejected. This lower bound
covers label overlap; longer dependence can still require a larger lag.
Default minimum observed dates is 20 (`--min-dates`). Short or constant series
retain their mean/status and have no inferential p-value. Confidence intervals
are pointwise asymptotic-normal intervals, not simultaneous intervals or proof
of finite-sample calibration.

## One declared search family

All supplied factor names and both metrics enter one family: 213 factors produce
426 hypotheses. Untestable series remain in its size with an effective p-value
of one; their displayed adjusted p-values stay `null`. Duplicate names and
inconsistent series lengths are rejected rather than quietly reducing the
declared search.

| `--method` | Error criterion | Dependence requirement |
| --- | --- | --- |
| `by` (default) | Benjamini–Yekutieli false discovery rate | Arbitrary dependence, conditional on valid marginal p-values |
| `bh` | Benjamini–Hochberg false discovery rate | Independence or appropriate positive dependence |
| `holm` | Holm family-wise error | Arbitrary dependence, conditional on valid marginal p-values |

`--alpha` defaults to 0.05. HAC p-values are asymptotic approximations, so these
conditional guarantees do not turn short, misspecified or informatively missing
series into valid tests. Correction covers the supplied family; researchers
must also account for unreported factor variants, repeated datasets, tuning and
later selection. A rejection supports further held-out investigation rather
than automatic inclusion in a portfolio.

## Python API

```python
from mlquant.backtest import ic_mean_inference, adjust_pvalues

result = ic_mean_inference(daily_rank_ic, max_lag=20, horizon=5)
adjusted = adjust_pvalues(all_predeclared_p_values, method="by")
```

`infer_factor_ic_report(report, ...)` performs the full report-family operation.
The CLI records the source report/payload checksums, original metadata, code
fingerprint, source commit and calculation environment. The frozen input is
preserved; output cannot replace it.

## Checked on the existing full synthetic report

The [November 18 audit](audits/factor-inference-20261118/README.md) reuses the
existing 500-date/200-stock/213-factor report. All 426 mean tests match an
independent Statsmodels HAC calculation; multiple-test formulas also match
Statsmodels. At the default lag, 19 unadjusted tests reach p<=0.05 and zero
remain after BY correction. This is synthetic statistical/engineering
evidence, with no new market return or strategy-performance claim.

## Purged rolling selection and future evaluation

```bash
mlquant factor-walk-forward --input-report artifacts/daily-ic.json \
  --json-out artifacts/factor-walk-forward.json \
  --train-dates 200 --test-dates 50 --max-factors 10
```

```python
from mlquant.backtest import walk_forward_factor_ic
result = walk_forward_factor_ic(report, train_dates=200, test_dates=50)
```

This command chooses factors using each trailing training window's joint
IC/RankIC correction, then freezes their names and signs for the next disjoint
test window. The default selection metric is RankIC (`--metric ic` also works).
Only factors whose chosen metric rejects after correction qualify. Order is
adjusted p-value, descending absolute HAC statistic, then name; `--max-factors`
caps the list. A negative training mean freezes a negative direction. A factor
that reverses in the test period retains its disappointing signed IC. Empty
selections remain empty; there is no best-looking-factor fallback.

For a label on signal row `t` ending at `t + label_horizon`, the default purge
gap is the full horizon. With test start `s`, training ends exclusively at
`s - purge_dates`, so its last label ends strictly before `s`. Training windows
contain the requested number of original rows; missing dates stay gaps.
`--purge-dates` can increase the gap but cannot reduce it below the horizon.
The final shorter test window is retained and marked `partial_test`.

Every fold saves all training hypotheses, selection decisions, exact ranges,
label endpoints and signed per-factor test IC series. `oos_by_factor` stitches
each named factor's selected windows on the original grid, leaving other rows
missing. Selection coverage and observed IC coverage are separate: choosing a
factor does not imply its future cross-section is observable. Future missingness
does not remove it from the frozen selection.

Each fold's correction applies to its training family, **not to repeated folds
or parameter searches as a whole**. Test summaries are descriptive and contain
no post-selection p-values. Signed IC is per factor, not portfolio returns or
the IC of a combined score. The input family, universe and preprocessing still
need to be causal and fixed independently of held-out outcomes; the declared
horizon must accurately describe when labels become known.
