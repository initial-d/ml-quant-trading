"""Loaders for real OCHLV CSV / parquet files.

The paper uses Wind exports; this module is intentionally generic so
you can plug in any tab-separated OCHLV dump that has at least these
columns:

    TRADE_DT  S_INFO_WINDCODE  open  close  high  low  vwap  volume

Any extra columns — ``S_DQ_AMOUNT`` (turnover), ``LIMIT_UP``,
``LIMIT_DOWN``, ``LAST_CLOSE``, ``FHZS_FLAG`` (ex-dividend / split
flag) — are picked up automatically when present and surfaced on the
:class:`Panel` so the bias-correction and market-breadth pipelines can
use the *real* exchange figures rather than ±10 % proxies.

The loader is the single entry point for any external file path; the
synthetic generator in :mod:`mlquant.data.synthetic` produces the same
shape, so downstream code never branches on data source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .panel import Panel


# Each entry maps the canonical Panel field name to a tuple of column
# aliases we accept on disk. The first match wins, so the most specific
# Wind name should come first.
_REQUIRED_ALIASES: Mapping[str, tuple[str, ...]] = {
    "open":   ("open",   "S_FWDS_ADJOPEN",  "OPEN"),
    "high":   ("high",   "S_FWDS_ADJHIGH",  "HIGH"),
    "low":    ("low",    "S_FWDS_ADJLOW",   "LOW"),
    "close":  ("close",  "S_FWDS_ADJCLOSE", "CLOSE"),
    "volume": ("volume", "S_DQ_VOLUME"),
    "vwap":   ("vwap",   "S_DQ_AVGPRICE"),
}

_OPTIONAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "amount":     ("amount",     "S_DQ_AMOUNT",   "AMOUNT"),
    "limit_up":   ("limit_up",   "LIMIT_UP",      "S_DQ_LIMIT"),
    "limit_down": ("limit_down", "LIMIT_DOWN",    "S_DQ_STOPPING"),
    "last_close": ("last_close", "LAST_CLOSE",    "S_DQ_PRECLOSE"),
}


def _resolve(df: pd.DataFrame, name: str, aliases: tuple[str, ...]) -> Optional[pd.Series]:
    for alias in aliases:
        if alias in df.columns:
            return df[alias].astype(np.float32)
    return None


def load_ochlv_csv(
    path: str | Path,
    *,
    sep: str = "\t",
    universe: Optional[Sequence[str]] = None,
    device: str | torch.device = "cpu",
) -> Panel:
    """Load a long-format OCHLV file into a :class:`Panel`.

    Parameters
    ----------
    path : str | Path
        Path to a tab-separated (or ``sep`` separated) file with at least
        ``TRADE_DT`` and ``S_INFO_WINDCODE`` columns plus OCHLV fields.
    universe : sequence of str, optional
        If given, restrict to this set of tickers before pivoting.
    device : torch device or str
        Where to allocate the resulting tensors.
    """
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if "TRADE_DT" not in df.columns or "S_INFO_WINDCODE" not in df.columns:
        raise KeyError("file must contain TRADE_DT and S_INFO_WINDCODE columns")

    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str))
    df["S_INFO_WINDCODE"] = df["S_INFO_WINDCODE"].astype(str).str[:6]
    if universe is not None:
        df = df[df["S_INFO_WINDCODE"].isin(set(universe))]
    df = df.drop_duplicates(subset=["TRADE_DT", "S_INFO_WINDCODE"]).sort_values(
        ["TRADE_DT", "S_INFO_WINDCODE"]
    )

    # Build a wide [date, stock] frame for every field we recognise.
    fields_long: dict[str, pd.Series] = {}
    for name, aliases in {**_REQUIRED_ALIASES, **_OPTIONAL_ALIASES}.items():
        ser = _resolve(df, name, aliases)
        if ser is not None:
            fields_long[name] = ser
        elif name in _REQUIRED_ALIASES:
            raise KeyError(
                f"None of {aliases} present in dataframe columns {list(df.columns)}"
            )

    fields_wide: dict[str, pd.DataFrame] = {}
    for fld, ser in fields_long.items():
        wide = (
            df.assign(**{fld: ser})
              .pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values=fld)
              .sort_index()
        )
        fields_wide[fld] = wide

    base = fields_wide["close"]
    dates = base.index.to_numpy()
    stocks = base.columns.to_numpy()

    mask = (~base.isna()).to_numpy()
    tensors = {
        name: torch.from_numpy(df_.fillna(0.0).to_numpy(dtype=np.float32)).to(device)
        for name, df_ in fields_wide.items()
    }

    panel = Panel.from_tensors(
        dates=dates,
        stocks=stocks,
        fields=tensors,
        mask=torch.from_numpy(mask).to(device),
    )
    return panel
