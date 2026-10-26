# Daily factor IC diagnostic

The earlier `eval_factor_ic.py` pooled stock-days into a single correlation and
computed factors before applying the limit-move mask. The corrected script
streams all 213 factors with mask-first construction and calculates Pearson IC
and average-tie Spearman RankIC across assets separately on each date.

Independent SciPy tests cover pairwise masks, ties and nonfinite inputs. A
regression shows two daily cross-sections with IC -1 each producing pooled IC
above +0.99 because the levels change across dates. Constant or undersized
cross-sections are missing observations, not zero IC. Label eligibility uses
adjacent available finite positive raw closes, separate from signal availability;
halt-crossing labels are not imputed, and the final day is unavailable.

The complete synthetic 500-date / 200-stock run completed all 213 factors at
source `410f6a6`, with 431–498 eligible dates per factor. This is full-sample
synthetic screening, not held-out factor selection or investment evidence.
Daily summaries weight each eligible date equally; IC IR is unannualized mean /
sample standard deviation, with no temporal-dependence adjustment or p-values.
IC excludes execution costs and does not measure portfolio profitability.

Apple M5 CPU / macOS 26.6.2, Python 3.12.14 / PyTorch 2.14.1, threads 1/1,
seeded SyntheticConfig GBM, no GPU, external-data fetch or paid services.
[summary.json](summary.json) retains source SHA, exact command, config/input
fingerprints, environment, limitations and per-factor summaries.
[report.json.gz](report.json.gz) contains the complete daily IC/RankIC and pair
count series; gzip is used to avoid committing a 7.4 MB formatted JSON file.
Inspect it with `gzip -dc report.json.gz` or Python's `gzip.open` + `json.load`.
[run.txt](run.txt) contains all 213 console summaries.

[local-tests.txt](local-tests.txt): 240 tests passed, including nine new daily-IC
regressions. Commands and tested code provenance are in
[local-checks.json](local-checks.json). This extends
[PR #81](https://github.com/initial-d/ml-quant-trading/pull/81).
