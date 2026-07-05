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

## Next Useful Reproductions

- Repeat the workflow on ETFs or sector-balanced universes.
- Add a larger public universe with clear membership rules.
- Compare US equities with A-share symbols supported by public providers.
- Turn this IC diagnostic into a small portfolio backtest with explicit transaction costs and leakage controls.
