"""Panel-data loaders + synthetic generator.

The package treats market data as a pair of ``[date, stock]`` torch
tensors plus a boolean mask telling us which cells are tradable. All
downstream code (factors, training, portfolio) operates on this layout.

Two data sources are wired up out of the box:

* ``"synthetic"`` — :func:`make_synthetic_panel`, deterministic GBM panel
  for tests, demos and CI.
* ``"csv"`` — :func:`load_ochlv_csv`, generic tab-separated Wind / Tushare
  dump with optional ``LIMIT_UP`` / ``LIMIT_DOWN`` / ``LAST_CLOSE`` /
  ``S_DQ_AMOUNT`` columns.

Both return the same :class:`Panel`, so consumers should never branch
on the source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .panel import Panel
from .synthetic import SyntheticConfig, make_synthetic_panel
from .loaders import load_ochlv_csv

__all__ = [
    "Panel",
    "SyntheticConfig",
    "make_synthetic_panel",
    "load_ochlv_csv",
    "make_panel",
]


def make_panel(source: str = "synthetic", **kwargs: Any) -> Panel:
    """Unified factory: ``make_panel("synthetic", ...)`` or ``make_panel("csv", path=...)``.

    Parameters
    ----------
    source : str
        Either ``"synthetic"`` (forwarded to :class:`SyntheticConfig`) or
        ``"csv"`` (forwarded to :func:`load_ochlv_csv`).
    **kwargs
        Source-specific keyword arguments.
    """
    if source == "synthetic":
        cfg = SyntheticConfig(**kwargs) if kwargs else None
        return make_synthetic_panel(cfg)
    if source == "csv":
        path: Optional[str | Path] = kwargs.pop("path", None)
        if path is None:
            raise TypeError("make_panel(source='csv', ...) requires a `path=` kwarg")
        return load_ochlv_csv(path, **kwargs)
    raise ValueError(f"unknown panel source: {source!r}")
