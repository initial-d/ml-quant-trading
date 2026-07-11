"""Reproduce one-day legacy-factor rank IC on a public ETF universe.

The script first requests the complete universe through ``make_panel``. If the
bulk request fails or returns empty columns, it retries the affected tickers
one at a time and merges every successful result into one Panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from yfinance import data as yfinance_data

from mlquant.data import Panel, make_panel
from mlquant.features import compute_legacy_set

ETF_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLB",
    "XLU",
    "AGG",
    "TLT",
    "GLD",
    "HYG",
    "EFA",
    "EEM",
    "FXI",
)

FACTOR_NAMES = (
    "best_001",
    "best_002",
    "original_001",
    "stock_001",
    "add_015",
    "old_042",
)

START = "2021-01-01"
END = "2025-01-01"
DEVICE = "cpu"
PANEL_FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@dataclass(frozen=True)
class DownloadResult:
    panel: Panel
    failed_tickers: tuple[str, ...]
    bulk_error: str | None


def configure_yfinance_user_agent() -> None:
    """Replace yfinance's obsolete Chrome 39 user agent when present.

    Some Yahoo endpoints reject that legacy user agent with HTTP 429 while
    accepting the same public chart request from a current generic browser
    user agent. Newer yfinance releases may not expose either class attribute,
    in which case this compatibility step is a no-op.
    """
    for class_name in ("YfData", "TickerData"):
        data_class = getattr(yfinance_data, class_name, None)
        headers = getattr(data_class, "user_agent_headers", None)
        if isinstance(headers, dict) and "Chrome/39" in headers.get("User-Agent", ""):
            data_class.user_agent_headers = {"User-Agent": "Mozilla/5.0"}


def one_day_forward_returns(panel: Panel) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-day forward returns and the corresponding tradability mask."""
    fwd = torch.roll(panel.returns, shifts=-1, dims=0)
    fwd[-1] = 0.0
    tradable_next = panel.mask & torch.roll(panel.mask, shifts=-1, dims=0)
    tradable_next[-1] = False
    return fwd, tradable_next


def rank_ic_by_date(
    factor_values: torch.Tensor,
    returns: torch.Tensor,
    valid_mask: torch.Tensor,
    dates: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Calculate daily cross-sectional Spearman rank IC for each factor."""
    rows: list[dict[str, object]] = []
    values_np = factor_values.detach().cpu().numpy()
    returns_np = returns.detach().cpu().numpy()
    mask_np = valid_mask.detach().cpu().numpy()

    for date_index, date in enumerate(dates):
        row: dict[str, object] = {"date": pd.Timestamp(date)}
        for factor_index, name in enumerate(names):
            mask = mask_np[date_index]
            values = values_np[date_index, :, factor_index]
            forward_returns = returns_np[date_index]
            valid = mask & np.isfinite(values) & np.isfinite(forward_returns)
            row[name] = (
                np.nan
                if valid.sum() < 3
                else pd.Series(values[valid]).corr(
                    pd.Series(forward_returns[valid]), method="spearman"
                )
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def _ticker_slice(panel: Panel, ticker_index: int) -> Panel:
    fields = {
        field: getattr(panel, field)[:, ticker_index : ticker_index + 1] for field in PANEL_FIELDS
    }
    return Panel.from_tensors(
        dates=panel.dates,
        stocks=np.asarray([panel.stocks[ticker_index]]),
        fields=fields,
        mask=panel.mask[:, ticker_index : ticker_index + 1],
    )


def _covered_ticker_panels(panel: Panel) -> dict[str, Panel]:
    covered: dict[str, Panel] = {}
    for ticker_index, ticker_value in enumerate(panel.stocks):
        ticker = str(ticker_value)
        if bool(panel.mask[:, ticker_index].any().item()):
            covered[ticker] = _ticker_slice(panel, ticker_index)
    return covered


def _merge_ticker_panels(
    ticker_panels: dict[str, Panel], requested_tickers: Sequence[str]
) -> Panel:
    successful_tickers = [ticker for ticker in requested_tickers if ticker in ticker_panels]
    if not successful_tickers:
        raise RuntimeError("yfinance returned no usable ETF data")

    dates = np.unique(
        np.concatenate([ticker_panels[ticker].dates for ticker in successful_tickers])
    )
    shape = (len(dates), len(successful_tickers))
    fields = {
        field: torch.zeros(shape, dtype=torch.float32, device=DEVICE) for field in PANEL_FIELDS
    }
    mask = torch.zeros(shape, dtype=torch.bool, device=DEVICE)

    for ticker_index, ticker in enumerate(successful_tickers):
        source = ticker_panels[ticker]
        date_indices = np.searchsorted(dates, source.dates)
        target_indices = torch.as_tensor(date_indices, dtype=torch.long, device=DEVICE)
        for field in PANEL_FIELDS:
            fields[field][target_indices, ticker_index] = getattr(source, field)[:, 0]
        mask[target_indices, ticker_index] = source.mask[:, 0]

    return Panel.from_tensors(
        dates=dates,
        stocks=np.asarray(successful_tickers),
        fields=fields,
        mask=mask,
    )


def download_with_fallback(tickers: Sequence[str]) -> DownloadResult:
    """Try one bulk yfinance request, then retry absent tickers individually."""
    ticker_panels: dict[str, Panel] = {}
    bulk_error: str | None = None

    try:
        bulk_panel = make_panel(
            source="yfinance",
            tickers=tuple(tickers),
            start=START,
            end=END,
            device=DEVICE,
        )
        ticker_panels.update(_covered_ticker_panels(bulk_panel))
    except Exception as exc:  # yfinance raises several provider-specific errors
        bulk_error = f"{type(exc).__name__}: {exc}"

    retry_tickers = [ticker for ticker in tickers if ticker not in ticker_panels]
    for ticker in retry_tickers:
        try:
            single_panel = make_panel(
                source="yfinance",
                tickers=(ticker,),
                start=START,
                end=END,
                device=DEVICE,
            )
        except Exception:
            continue
        if bool(single_panel.mask.any().item()):
            ticker_panels[ticker] = _ticker_slice(single_panel, 0)

    failed_tickers = tuple(ticker for ticker in tickers if ticker not in ticker_panels)
    panel = _merge_ticker_panels(ticker_panels, tickers)
    panel.assert_consistent()
    return DownloadResult(panel, failed_tickers, bulk_error)


def coverage_table(
    panel: Panel, requested_tickers: Sequence[str], failed_tickers: Sequence[str]
) -> pd.DataFrame:
    """Summarize valid OHLCV coverage for every requested ticker."""
    panel_lookup = {str(ticker): index for index, ticker in enumerate(panel.stocks)}
    failed = set(failed_tickers)
    rows = []

    for ticker in requested_tickers:
        if ticker in failed:
            rows.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "observations": 0,
                    "first_date": "-",
                    "last_date": "-",
                    "coverage_rate": 0.0,
                }
            )
            continue

        ticker_index = panel_lookup[ticker]
        valid_indices = panel.mask[:, ticker_index].detach().cpu().numpy().nonzero()[0]
        observations = int(valid_indices.size)
        rows.append(
            {
                "ticker": ticker,
                "status": "full" if observations == panel.n_dates else "partial",
                "observations": observations,
                "first_date": str(pd.Timestamp(panel.dates[valid_indices[0]]).date()),
                "last_date": str(pd.Timestamp(panel.dates[valid_indices[-1]]).date()),
                "coverage_rate": observations / panel.n_dates,
            }
        )

    return pd.DataFrame(rows)


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    """Match the summary calculation in public_factor_ic.ipynb."""
    return pd.DataFrame(
        {
            "mean_ic": ic.mean(skipna=True),
            "median_ic": ic.median(skipna=True),
            "ic_std": ic.std(skipna=True),
            "positive_rate": (ic > 0).mean(skipna=True),
            "observations": ic.count(),
        }
    ).sort_values("mean_ic", ascending=False)


def print_coverage(coverage: pd.DataFrame) -> None:
    print("Per-ticker data coverage:")
    print(
        f"{'Ticker':<8} {'Status':<8} {'Observations':>12} "
        f"{'First Date':>12} {'Last Date':>12} {'Coverage':>10}"
    )
    print("-" * 69)
    for row in coverage.itertuples(index=False):
        print(
            f"{row.ticker:<8} {row.status:<8} {row.observations:>12d} "
            f"{row.first_date:>12} {row.last_date:>12} {row.coverage_rate:>9.2%}"
        )


def print_ic_summary(summary: pd.DataFrame) -> None:
    print("One-day forward Spearman rank IC summary:")
    print(
        f"{'Factor':<16} {'Mean IC':>9} {'Median IC':>11} "
        f"{'IC Std':>9} {'Positive Rate':>14} {'Observations':>13}"
    )
    print("-" * 78)
    for factor, row in summary.iterrows():
        print(
            f"{factor:<16} {row['mean_ic']:>9.4f} {row['median_ic']:>11.4f} "
            f"{row['ic_std']:>9.4f} {row['positive_rate']:>13.2%} "
            f"{int(row['observations']):>13d}"
        )


def main() -> None:
    configure_yfinance_user_agent()
    result = download_with_fallback(ETF_UNIVERSE)
    panel = result.panel

    print("ETF cross-asset public-data reproduction")
    print(f"yfinance version: {yf.__version__}")
    print(f"Requested date range: {START} to {END} (end exclusive)")
    print(
        "Returned panel: "
        f"{panel.n_dates} dates x {panel.n_stocks} ETFs, "
        f"{pd.Timestamp(panel.dates[0]).date()} to "
        f"{pd.Timestamp(panel.dates[-1]).date()}"
    )
    print(f"Bulk download error: {result.bulk_error or 'none'}")
    print(
        "Failed tickers: " + (", ".join(result.failed_tickers) if result.failed_tickers else "none")
    )
    print()

    coverage = coverage_table(panel, ETF_UNIVERSE, result.failed_tickers)
    print_coverage(coverage)
    print()

    factors, factor_mask, names = compute_legacy_set(panel, names=FACTOR_NAMES)
    forward_returns, forward_mask = one_day_forward_returns(panel)
    valid_mask = factor_mask & forward_mask
    ic = rank_ic_by_date(factors, forward_returns, valid_mask, panel.dates, names)
    summary = summarize_ic(ic)
    print_ic_summary(summary)


if __name__ == "__main__":
    main()
