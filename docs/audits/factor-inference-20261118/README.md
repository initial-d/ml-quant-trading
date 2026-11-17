# Frozen full-factor inference audit

Source implementation: `be9afec413bc68598cd917015f524772cef721f3`.
Input: the existing [full daily factor report](../../benchmarks/daily-factor-ic-20261026/README.md),
500 dates x 200 synthetic stocks, all 213 registered factors. The saved source
commit, input fingerprints and original environment remain in each inference
report's `source_metadata`. No factor recomputation, market-data fetch or GPU
work was needed.

Every supplied factor contributes two hypotheses (IC and RankIC). All 426 are
testable in this source, with 431–498 observed dates and gaps preserved on the
500-row grid. The default automatic Bartlett lag is five rows for every series.
Pointwise mean intervals/p-values use an asymptotic normal reference and
finite-sample multiplier n/(n-1). This is full-sample synthetic inference,
not held-out selection or alpha evidence.

| Bartlett lag | Tests | Unadjusted p<=0.05 | BY adjusted p<=0.05 |
| --- | ---: | ---: | ---: |
| 0 (IID mean reference) | 426 | 14 | 0 |
| Automatic (5) | 426 | 19 | 0 |
| 20 (sensitivity) | 426 | 21 | 0 |

HAC accommodates both positive and negative dependence: its standard errors
need not always be larger than IID. The 19 unadjusted tests were calculated in
this audit; the original report published means/IC IR without p-values. BY
corrects this declared 426-test family, not unreported experiments or future
tuning. A synthetic source does not establish calibration on real markets.

## Independent checks

[oracle.json](oracle.json) compares **all 426 series** with Statsmodels 0.14.4:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Mean IC | 5.20e-18 |
| HAC standard error | 6.51e-18 |
| Two-sided p-value | 8.88e-16 |

For missing dates the oracle fits OLS on the original grid with zero design
rows at missing positions. This preserves the score lags and yields X'X=n.
Statsmodels' uncorrected HAC covariance is multiplied by the specified
observed-date n/(n-1), rather than its built-in total-design-row correction.
BY/BH/Holm match Statsmodels on the full family and nontrivial controls including
ties, zero/unit p-values and untestable hypotheses. Test regressions independently
construct the dense Bartlett matrix, check calendar gaps and label-overlap lag
requirements, and exercise CLI/gzip/provenance and input-validation contracts.

Calculation environment: Apple M5 CPU, macOS 26.6.2, Python 3.9.6, PyTorch 2.8.0
installed (GPU unused), NumPy 2.0.2, SciPy 1.13.1, Statsmodels 0.14.4. No
transactions or paid data; IC excludes execution costs. This audit checks
implementation consistency, not predictive quality, finite-sample statistical
calibration or an engineering speedup.

## Evidence and reproduction

[summary.json](summary.json) records source/output hashes and the sensitivity
counts. Compressed inference records keep complete per-factor means, standard
errors, p-values, intervals, lag-pair counts, statuses and adjusted decisions:

- [default automatic BY](inference-auto-by.json.gz)
- [lag-zero BY](inference-lag0-by.json.gz)
- [lag-20 BY](inference-lag20-by.json.gz)

The original 2.1 MB source remains at its existing path; this audit does not
duplicate it or introduce market data/model weights into the repository.

Check the retained source/output hashes, all joint-family decision counts and
the oracle's connection to the default report with the standard library only:

```bash
python docs/audits/factor-inference-20261118/verify.py
```

This checks saved-evidence consistency; it does not rerun inference. To rerun:

```bash
mlquant factor-inference \
  --input-report docs/benchmarks/daily-factor-ic-20261026/report.json.gz \
  --json-out artifacts/factor-inference.json
python docs/audits/factor-inference-20261118/check_oracle.py \
  --source docs/benchmarks/daily-factor-ic-20261026/report.json.gz \
  --inference artifacts/factor-inference.json --output artifacts/factor-oracle.json
```

The oracle needs audit-only `statsmodels==0.14.4` plus Patsy and SciPy; they are
not new runtime requirements. Original exact commands and hashes are retained
in each report and in `oracle.json`.

[pytest-clean.log](pytest-clean.log): **423 tests passed** on the complete source
suite. The local check overlay uses NumPy 1.26.4 and AkShare 1.18.88 while
retaining PyTorch 2.8.0; market-provider tests use mocks. The initial old shared
environment omitted AkShare, imported stale code in subprocesses and emitted a
NumPy-2/Accelerate matmul warning as an error. Its failed log is preserved in
`pytest-old-environment.log.gz`. Setting the checkout's absolute PYTHONPATH and
using task-local dependency overlays resolved all six failures without changing
those tests or the portfolio implementation. Ruff also passes.

Exact full-suite command for this retained local log (the temporary dependency
overlay is task-local, not a repository or runtime requirement):

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib \
PYTHONPATH=/tmp/mlquant-hac-20261118/check-deps:/Users/duyimin/Documents/Codex/2026-07-04/ban/work/ml-quant-trading/src \
../bench-venv/bin/python -m pytest
```
