from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Sequence, Optional, Union
import yfinance as yf

from .panel import Panel

def load_yfinance_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    device: Union[str, torch.device] = "cpu",
    proxy_vwap: bool = True
) -> Panel:
    """Download data from yfinance and return a Panel.

    Parameters
    ----------
    tickers : Sequence[str]
        List of tickers to download. For A-shares, use standard suffixes (e.g., '000001.SZ', '600000.SS').
    start : str
        Start date (e.g. "2020-01-01").
    end : str
        End date (e.g. "2023-12-31").
    device : str or torch.device
        Where to allocate the resulting tensors.
    proxy_vwap : bool
        If True, estimates VWAP as (Open + Close + High + Low) / 4.
        yfinance doesn't provide VWAP natively.
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    df = yf.download(list(tickers), start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for tickers {tickers} from {start} to {end}")

    # Standardize column naming
    # For single ticker, yfinance returns flat columns. For multiple, it returns MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex columns: level 0 is Price (e.g. 'Close', 'Open'), level 1 is Ticker
        pass
    else:
        # Flat index: this means a single ticker was given to yf.download
        if len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])
        else:
            raise ValueError("Unexpected flat columns for multiple tickers")

    # Extract wide DataFrames
    fields_wide = {}

    def get_wide_df(price_col):
        if price_col in df.columns.get_level_values(0):
            # Extract cross-section and sort columns by ticker order
            wide = df[price_col].copy()
            # If some tickers missing, add them
            for t in tickers:
                if t not in wide.columns:
                    wide[t] = np.nan
            return wide[list(tickers)]
        return None

    open_df = get_wide_df("Open")
    high_df = get_wide_df("High")
    low_df = get_wide_df("Low")
    close_df = get_wide_df("Close")
    volume_df = get_wide_df("Volume")

    if any(x is None for x in [open_df, high_df, low_df, close_df, volume_df]):
        raise ValueError("Missing required OHLCV columns from yfinance data")

    fields_wide["open"] = open_df
    fields_wide["high"] = high_df
    fields_wide["low"] = low_df
    fields_wide["close"] = close_df
    fields_wide["volume"] = volume_df

    if proxy_vwap:
        fields_wide["vwap"] = (open_df + close_df + high_df + low_df) / 4.0

    dates = df.index.to_numpy()
    stocks = np.array(list(tickers))

    # Mask where close is not NaN
    mask = (~open_df.isna() & ~high_df.isna() & ~low_df.isna() & ~close_df.isna() & ~volume_df.isna()).to_numpy()

    tensors = {
        name: torch.from_numpy(df_.fillna(0.0).to_numpy(dtype=np.float32).copy()).to(device)
        for name, df_ in fields_wide.items()
    }

    return Panel.from_tensors(
        dates=dates,
        stocks=stocks,
        fields=tensors,
        mask=torch.from_numpy(mask.copy()).to(device),
    )
