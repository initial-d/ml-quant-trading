from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Sequence, Optional, Union

from .panel import Panel

def load_baostock_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    device: Union[str, torch.device] = "cpu",
    proxy_vwap: bool = True
) -> Panel:
    """Download data from baostock and return a Panel.

    Parameters
    ----------
    tickers : Sequence[str]
        List of tickers to download. For A-shares, use baostock format (e.g., 'sh.600000', 'sz.000001').
    start : str
        Start date (e.g. "2020-01-01").
    end : str
        End date (e.g. "2023-12-31").
    device : str or torch.device
        Where to allocate the resulting tensors.
    proxy_vwap : bool
        If True, estimates VWAP as (Open + Close + High + Low) / 4.
    """
    import baostock as bs

    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

    all_data = []

    for ticker in tickers:
        rs = bs.query_history_k_data_plus(
            ticker,
            "date,code,open,high,low,close,preclose,volume,amount,tradestatus",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3" # 3 for forward adjust
        )
        if rs.error_code != '0':
            print(f"Warning: Failed to fetch {ticker}: {rs.error_msg}")
            continue

        while (rs.error_code == '0') & rs.next():
            all_data.append(rs.get_row_data())

    bs.logout()

    if not all_data:
        raise ValueError(f"No data returned for tickers {tickers} from {start} to {end}")

    df = pd.DataFrame(all_data, columns=["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus"])

    # Convert numerical columns
    num_cols = ["open", "high", "low", "close", "preclose", "volume", "amount"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["date"] = pd.to_datetime(df["date"])

    # Extract wide DataFrames
    fields_wide = {}

    def get_wide_df(col_name):
        wide = df.pivot(index="date", columns="code", values=col_name)
        # Ensure all requested tickers are present
        for t in tickers:
            if t not in wide.columns:
                wide[t] = np.nan
        return wide[list(tickers)]

    open_df = get_wide_df("open")
    high_df = get_wide_df("high")
    low_df = get_wide_df("low")
    close_df = get_wide_df("close")
    volume_df = get_wide_df("volume")
    amount_df = get_wide_df("amount")
    preclose_df = get_wide_df("preclose")
    status_df = get_wide_df("tradestatus") # 1 means tradable, 0 means halt

    fields_wide["open"] = open_df
    fields_wide["high"] = high_df
    fields_wide["low"] = low_df
    fields_wide["close"] = close_df
    fields_wide["volume"] = volume_df
    fields_wide["amount"] = amount_df
    fields_wide["last_close"] = preclose_df

    if proxy_vwap:
        # Avoid division by zero by filling NaNs or 0s
        # Only compute vwap where volume > 0, otherwise use proxy
        vwap_actual = amount_df / volume_df
        # If amount or volume is 0 or NaN, fallback to typical price
        vwap_proxy = (open_df + close_df + high_df + low_df) / 4.0
        fields_wide["vwap"] = vwap_actual.fillna(vwap_proxy).replace([np.inf, -np.inf], np.nan).fillna(vwap_proxy)

    dates = open_df.index.to_numpy()
    stocks = np.array(list(tickers))

    # status_df is string '1' or '0' natively before we pivoted, so convert it properly
    # If a stock was missing on a day, it's NaN. Let's make tradable mask: status == '1'
    status_df = status_df.fillna('0')
    is_tradable = (status_df == '1') | (status_df == 1)

    # Mask where close is not NaN and tradable
    mask = (~open_df.isna() & ~high_df.isna() & ~low_df.isna() & ~close_df.isna() & ~volume_df.isna() & is_tradable).to_numpy()

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
