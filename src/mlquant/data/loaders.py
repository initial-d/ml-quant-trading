"""Loaders for real OCHLV CSV / parquet files.

The paper uses Wind exports; this module is intentionally generic so
you can plug in any tab-separated OCHLV dump that has at least these
columns:

    TRADE_DT  S_INFO_WINDCODE  open  close  high  low  vwap  volume

Any extra columns (limit_up, limit_down, dividend flags, etc.) are
preserved and respected by the bias-correction pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .panel import Panel


_FIELD_ALIASES = {
    "open":   ("open", "S_FWDS_ADJOPEN", "OPEN"),
    "high":   ("high", "S_FWDS_ADJHIGH", "HIGH"),
    "low":    ("low",  "S_FWDS_ADJLOW",  "LOW"),
    "close":  ("close","S_FWDS_ADJCLOSE","CLOSE"),
    "volume": ("volume","S_DQ_VOLUME"),
    "vwap":   ("vwap", "S_DQ_AVGPRICE"),
}


def _resolve(df: pd.DataFrame, name: str) -> pd.Series:
    for alias in _FIELD_ALIASES[name]:
        if alias in df.columns:
            return df[alias].astype(np.float32)
    raise KeyError(f"None of {_FIELD_ALIASES[name]} present in dataframe columns {list(df.columns)}")


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
    """
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if "TRADE_DT" not in df.columns or "S_INFO_WINDCODE" not in df.columns:
        raise KeyError("file must contain TRADE_DT and S_INFO_WINDCODE columns")

    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str))
    df["S_INFO_WINDCODE"] = df["S_INFO_WINDCODE"].astype(str).str[:6]
    if universe is not None:
        df = df[df["S_INFO_WINDCODE"].isin(set(universe))]
    df = df.drop_duplicates(subset=["TRADE_DT", "S_INFO_WINDCODE"]).sort_values(["TRADE_DT", "S_INFO_WINDCODE"])

    fields = {}
    for fld in ("open", "high", "low", "close", "volume", "vwap"):
        wide = (
            df.assign(**{fld: _resolve(df, fld)})
              .pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values=fld)
              .sort_index()
        )
        fields[fld] = wide

    base = fields["close"]
    dates = base.index.to_numpy()
    stocks = base.columns.to_numpy()

    mask = (~base.isna()).to_numpy()
    tensors = {
        name: torch.from_numpy(df_.fillna(0.0).to_numpy(dtype=np.float32)).to(device)
        for name, df_ in fields.items()
    }

    return Panel.from_tensors(
        dates=dates,
        stocks=stocks,
        fields=tensors,
        mask=torch.from_numpy(mask).to(device),
    )
