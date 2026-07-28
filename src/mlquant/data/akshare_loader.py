from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch

from .panel import Panel


def load_akshare_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    device: Union[str, torch.device] = "cpu",
    adjust: str = "qfq",
    proxy_vwap: bool = True,
) -> Panel:
    """Download A-share daily data from akshare and return a Panel.

    Parameters
    ----------
    tickers : Sequence[str]
        A-share ticker codes without exchange prefixes (for example, "600519").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.
    device : str or torch.device
        Where to allocate the resulting tensors.
    adjust : str
        Price adjustment mode: "qfq", "hfq", or an empty string.
    proxy_vwap : bool
        If True, estimate VWAP as (Open + Close + High + Low) / 4.
    """
    import akshare

    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    # 按输入顺序去重，避免下载和透视时产生重复列
    unique_tickers = list(dict.fromkeys(tickers))

    if not proxy_vwap:
        raise ValueError(
            "akshare cannot provide an adjustment-consistent raw VWAP; use proxy_vwap=True"
        )

    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    data_columns = ["date", "open", "high", "low", "close", "volume", "amount"]

    all_data = []
    for ticker in unique_tickers:
        try:
            ticker_df = akshare.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust,
            )
            if ticker_df is None or ticker_df.empty:
                continue

            ticker_df = ticker_df.rename(columns=column_map)
            missing_columns = [column for column in data_columns if column not in ticker_df.columns]
            if missing_columns:
                raise ValueError(f"missing columns: {missing_columns}")

            ticker_df = ticker_df[data_columns].copy()
            ticker_df["code"] = ticker
            all_data.append(ticker_df)
        except Exception as e:
            print(f"Warning: Failed to fetch {ticker}: {e}")
            continue

    if not all_data:
        raise ValueError(
            f"No data returned for tickers {unique_tickers} from {start} to {end}"
        )

    df = pd.concat(all_data, ignore_index=True)

    # 将行情字段统一转为数值类型
    num_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])

    # 将长表转换为日期乘股票的宽表
    fields_wide = {}

    def get_wide_df(col_name):
        wide = df.pivot(index="date", columns="code", values=col_name)
        # 为所有请求的股票补齐列，并保持原始顺序
        for ticker in unique_tickers:
            if ticker not in wide.columns:
                wide[ticker] = np.nan
        return wide[list(unique_tickers)]

    open_df = get_wide_df("open")
    high_df = get_wide_df("high")
    low_df = get_wide_df("low")
    close_df = get_wide_df("close")
    volume_df = get_wide_df("volume")
    amount_df = get_wide_df("amount")

    fields_wide["open"] = open_df
    fields_wide["high"] = high_df
    fields_wide["low"] = low_df
    fields_wide["close"] = close_df
    fields_wide["volume"] = volume_df
    fields_wide["amount"] = amount_df
    fields_wide["vwap"] = (open_df + close_df + high_df + low_df) / 4.0

    dates = open_df.index.to_numpy()
    stocks = list(unique_tickers)

    mask = (
        ~open_df.isna()
        & ~high_df.isna()
        & ~low_df.isna()
        & ~close_df.isna()
        & ~volume_df.isna()
    ).to_numpy()

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
