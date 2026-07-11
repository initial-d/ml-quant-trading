# Public-Data Mini Reproduction

This note records a small public-data factor IC run using `yfinance`. It is meant
to verify the public API path and document a reproducible workflow without
proprietary data. It is not a trading strategy, investment advice, or evidence
of live profitability.

## Setup

- Commit: `c16501c`
- Data source: `yfinance`
- Universe: `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `META`, `JPM`, `XOM`, `UNH`, `LLY`
- Date range requested: `2021-01-01` to `2025-01-01`
- Date range returned: `2021-01-04` to `2024-12-31`
- Observations: 1005 dates x 10 stocks
- Device: CPU
- Random seed: not used for the yfinance path; the synthetic fallback in the notebook uses `seed=7`

Factor subset:

- `best_001`
- `best_002`
- `original_001`
- `stock_001`
- `add_015`
- `old_042`

## Reproduction Snippet

This follows the same pattern as `notebooks/public_factor_ic.ipynb`.

```python
from mlquant.data import make_panel
from mlquant.features import compute_legacy_set

tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "JPM", "XOM", "UNH", "LLY",
]

panel = make_panel(
    source="yfinance",
    tickers=tickers,
    start="2021-01-01",
    end="2025-01-01",
    device="cpu",
)

factor_names = (
    "best_001",
    "best_002",
    "original_001",
    "stock_001",
    "add_015",
    "old_042",
)

factors, factor_mask, names = compute_legacy_set(panel, names=factor_names)
```

## One-Day Forward Rank IC Summary

The table reports daily cross-sectional Spearman rank IC between each factor and
one-day forward returns. With only 10 liquid US equities this is intentionally a
small smoke test, not a statistically strong research claim.

| Factor | Mean IC | Median IC | IC Std | Positive Rate | Observations |
|---|---:|---:|---:|---:|---:|
| `old_042` | 0.0150 | 0.0303 | 0.3625 | 0.5224 | 985 |
| `add_015` | 0.0139 | 0.0061 | 0.3953 | 0.4985 | 985 |
| `stock_001` | 0.0078 | 0.0182 | 0.3497 | 0.5015 | 985 |
| `best_002` | -0.0007 | -0.0061 | 0.3572 | 0.4796 | 983 |
| `original_001` | -0.0013 | 0.0061 | 0.4113 | 0.4935 | 985 |
| `best_001` | -0.0050 | -0.0066 | 0.3687 | 0.4677 | 975 |

Best mean IC in this small run: `old_042`.

## Interpretation

- The workflow successfully downloads public OHLCV data, builds a `Panel`, computes a factor subset, and calculates one-day forward rank IC.
- The universe is deliberately tiny, so cross-sectional IC is noisy.
- `yfinance` adjusts and backfills data differently from institutional market data vendors; treat this as an API/reproducibility check rather than a paper replication.
- No transaction costs, survivorship-bias controls, portfolio construction, or slippage model are included in this mini note.

## Expected Output Check

Use the table above as a setup check. A successful run should produce:

- a `Panel` with 10 stocks and roughly 1000 trading dates for the requested range
- one-day rank IC rows for all six factors
- `old_042` near the top of the small-universe mean IC ranking in the maintainer run

Small numerical differences are expected when yfinance revises data or when pandas,
PyTorch, or BLAS versions change. Treat large differences as a prompt to check the
downloaded ticker data, missing values, and dependency versions.

## Second Case: ETF Cross-Asset Universe

This second public-data case expands the cross-section from 10 individual US
stocks to 20 liquid ETFs spanning broad equity benchmarks, US sectors, fixed
income, gold, and international equity markets. The purpose remains an API and
reproducibility check; the results are not a trading recommendation or evidence
of investable performance.

### Setup

- Commit used as the repository base: `52c894e`
- Data source: `yfinance` 0.2.40
- Broad market ETFs: `SPY`, `QQQ`, `IWM`, `DIA`
- Sector ETFs: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLI`, `XLB`, `XLU`
- Cross-asset ETFs: `AGG`, `TLT`, `GLD`, `HYG`
- International ETFs: `EFA`, `EEM`, `FXI`
- Date range requested: `2021-01-01` to `2025-01-01` (end exclusive)
- Date range returned: `2021-01-04` to `2024-12-31`
- Observations: 1005 dates x 20 ETFs
- Per-ticker coverage: all 20 ETFs returned 1005 valid OHLCV rows
- Failed or partial tickers: none
- Device: CPU
- Random seed: not used

The factor subset is unchanged from the first case: `best_001`, `best_002`,
`original_001`, `stock_001`, `add_015`, and `old_042`.

### Reproduction Snippet

The complete reproduction, including a bulk download followed by one-ticker
retries for any missing columns, is checked in as `scripts/etf_factor_ic.py`.
From the repository root, run:

```bash
PYTHONIOENCODING=utf-8 python scripts/etf_factor_ic.py \
  | tee scripts/etf_factor_ic_output.txt
```

The factor and IC path used by that script is:

```python
from scripts.etf_factor_ic import (
    ETF_UNIVERSE,
    FACTOR_NAMES,
    configure_yfinance_user_agent,
    download_with_fallback,
    one_day_forward_returns,
    rank_ic_by_date,
    summarize_ic,
)
from mlquant.features import compute_legacy_set

configure_yfinance_user_agent()
result = download_with_fallback(ETF_UNIVERSE)
panel = result.panel
panel.assert_consistent()

factors, factor_mask, names = compute_legacy_set(panel, names=FACTOR_NAMES)
fwd_returns, fwd_mask = one_day_forward_returns(panel)
ic = rank_ic_by_date(
    factors,
    fwd_returns,
    factor_mask & fwd_mask,
    panel.dates,
    names,
)
summary = summarize_ic(ic)
print(summary)
```

The compatibility helper only replaces yfinance's obsolete Chrome 39 user
agent when that exact default is present. The loader call itself remains
`make_panel(source="yfinance", ...)`, and newer yfinance versions without that
legacy default are left unchanged.

### One-Day Forward Rank IC Summary

As in the first case, the table reports daily cross-sectional Spearman rank IC
between each factor and one-day forward returns. The calculations match
`notebooks/public_factor_ic.ipynb`, including its warm-up masks and summary
expressions.

| Factor | Mean IC | Median IC | IC Std | Positive Rate | Observations |
|---|---:|---:|---:|---:|---:|
| `original_001` | 0.0131 | 0.0090 | 0.3782 | 0.4965 | 985 |
| `add_015` | -0.0031 | -0.0060 | 0.3595 | 0.4846 | 985 |
| `stock_001` | -0.0044 | 0.0030 | 0.2618 | 0.4915 | 985 |
| `old_042` | -0.0057 | -0.0045 | 0.3108 | 0.4806 | 985 |
| `best_002` | -0.0061 | -0.0108 | 0.3174 | 0.4746 | 985 |
| `best_001` | -0.0136 | -0.0030 | 0.3150 | 0.4856 | 982 |

Best mean IC in this ETF run: `original_001`.

### Interpretation

- Expanding from 10 stocks to 20 ETFs changes the factor ordering: `old_042`
  led the first mini case, while `original_001` has the highest mean IC here.
  All mean values remain close to zero, so the ranking should be treated as a
  smoke-test result rather than stable evidence of predictive power.
- ETFs compress many company-specific effects into portfolio-level returns.
  The nine sector funds still provide cross-sectional differentiation through
  sector rotation, but they also share substantial broad-market exposure; that
  combination can change how stock-oriented legacy factors rank the universe.
- The bond, gold, and international funds add return drivers that differ from
  US equity sectors. This broadens the API check, but the aggregate summary does
  not by itself establish that any factor is stable within each asset subgroup.
- The two repeated maintainer runs produced byte-identical standard output. That
  checks deterministic computation against the same downloaded data, not future
  stability of the upstream public data.

### Caveats

- `yfinance` is a convenient public-data source, not an institutional data
  contract. Historical adjustments, missing values, endpoint behavior, and
  rate limits can change after this run.
- ETF liquidity, trading hours, premiums or discounts, and underlying-market
  closures differ across the universe. This mini reproduction does not model
  those differences.
- ETF survivorship bias is generally less visible than in a stock universe
  because these selected funds all survived the requested period. The selection
  is retrospective and does not constitute a point-in-time membership rule.
- No transaction costs, slippage, portfolio construction, multiple-testing
  controls, or subgroup significance tests are included.
- This project and reproduction are for research and engineering experimentation
  only. They are not financial advice, investment advice, or a trading
  recommendation.

### Expected Output Check

Use the committed `scripts/etf_factor_ic_output.txt` as the detailed reference.
A successful rerun should satisfy these checks:

| Check | Maintainer Run |
|---|---:|
| Requested ETFs | 20 |
| Successfully loaded ETFs | 20 |
| Failed or partial ETFs | 0 |
| Returned panel | 1005 dates x 20 ETFs |
| Returned date range | 2021-01-04 to 2024-12-31 |
| Factor summary rows | 6 |
| IC observations per factor | 982 to 985 |
| Highest mean IC | `original_001` at 0.0131 |

Small numerical differences are expected if yfinance revises adjusted history or
if pandas, PyTorch, SciPy, or yfinance behavior changes. Missing tickers should be
reported rather than silently replaced, and large IC differences should prompt a
review of the coverage table, dependency versions, and factor masks.

## Next Useful Reproductions

- Repeat the workflow on ETFs or sector-balanced universes.
- Add a larger public universe with clear membership rules.
- Compare US equities with A-share symbols supported by public providers.
- Run the larger walk-forward validation harness in [`public_data_validation.md`](public_data_validation.md), which includes portfolio construction, transaction costs, slippage, turnover, drawdown, and baseline comparisons.
