# `legacy/` — original research scripts (archival, **unsupported**)

The files under this directory are the *raw research scripts* that
produced the numbers in the paper before the codebase was cleaned up.
They are kept for archival reference and to make ablation reproduction
easier; they are **not** part of the supported `mlquant` package.

If you are looking for clean, tested, runnable code:
**use the `mlquant` package under `src/mlquant/`.** The `Makefile`
targets (`make features`, `make train`, `make portfolio`, `make
backtest`) wire it together for you.

## What's here

| Path                          | Original purpose                                         | Replacement in `mlquant`                                |
|-------------------------------|----------------------------------------------------------|---------------------------------------------------------|
| `cuda_features.py`            | PyTorch GPU factor engine (single huge file)             | `mlquant.features.tensor_factors`                       |
| `features/Feature.py`         | Pandas-based alpha-101++ library                         | `mlquant.features.alpha101`                             |
| `features/generate_feature_*` | OCHL → factor pickle dump                                | `mlquant cli features`                                  |
| `features/sigGen.cpp`         | C++ window-sample generator                              | (removed; the GPU path subsumes this)                   |
| `portfolio/portfolio_v2.py`   | MOSEK rotated-quadratic-cone Markowitz with α-sweep      | `mlquant.portfolio.markowitz` + `portfolio.frontier`    |
| `portfolio/*` (other variants)| Per-experiment forks                                     | (one canonical implementation in `mlquant.portfolio`)   |
| `train_inday/*`               | Training, prediction, eval scripts (multiple variants)   | `mlquant.training.{trainer,dataset,augment}`            |
| `backtracking/*`              | Awk + shell evaluation pipelines                         | `mlquant.backtest.{engine,metrics}`                     |

## Why keep them?

* **Provenance.** Several numbers in the paper depend on idiosyncratic
  choices (label horizon, masking thresholds, neutralisation
  ordering) that are easier to audit by reading the original script
  than by inferring them from the cleaned-up code.
* **Ablations.** The Alpha101++ extension (~600 factors) and the
  alternative loss variants (`AdjMSELoss1/2/3`) live here verbatim and
  can be ported one-by-one as the test suite grows.

## What got removed

* `mosek.lic` — a commercial licence accidentally committed to the
  original repo. **Removed from `HEAD`. If you forked the original
  repo, please rotate that licence.**
* `__pycache__/`, `nohup.out`, `Untitled.ipynb`, `*.bak*`, the stray
  `'` file, and `libxcb.so.1` — research detritus.
* The compiled `sigGen` binary; rebuild from `sigGen.cpp` if you need
  it.

## Will the legacy scripts run?

They depend on absolute paths (`/da1/public/duyimin/...`,
`/home/guochenglin/...`) and on commercial / paywalled data feeds
(Wind, Tushare). They were never intended for external execution and
are not part of CI. If you want to re-run a specific ablation, copy
the relevant script into a notebook and adapt the paths.
