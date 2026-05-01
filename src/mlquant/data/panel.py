"""The :class:`Panel` — a thin, mask-aware container for OCHLV panels.

Design rules
------------
* Every field has shape ``[T, N]`` (date × stock) and dtype ``float32``.
* ``mask[t, i] == True`` iff stock *i* was tradable on date *t*. Limit-up,
  limit-down, halted, and pre-IPO cells are masked out — this is the
  "bias correction" plumbing the paper relies on.
* Panels are *immutable*: factor pipelines return new tensors rather
  than mutating in place. This keeps reasoning straightforward and is
  a hard requirement for autograd-backed factors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class Panel:
    """OCHLV panel + tradability mask, all on the same device."""

    dates:  np.ndarray            # shape [T], dtype object/datetime64
    stocks: np.ndarray            # shape [N], dtype object (ticker str)

    open:   torch.Tensor          # [T, N]
    high:   torch.Tensor
    low:    torch.Tensor
    close:  torch.Tensor
    volume: torch.Tensor
    vwap:   torch.Tensor
    mask:   torch.Tensor          # [T, N] bool

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_tensors(
        cls,
        dates: Sequence,
        stocks: Sequence[str],
        fields: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> "Panel":
        required = {"open", "high", "low", "close", "volume", "vwap"}
        missing = required - set(fields)
        if missing:
            raise KeyError(f"Panel.from_tensors missing fields: {sorted(missing)}")
        T, N = mask.shape
        for k, v in fields.items():
            if tuple(v.shape) != (T, N):
                raise ValueError(f"field {k!r} has shape {tuple(v.shape)}, expected {(T, N)}")
            if v.dtype != torch.float32:
                fields = {**fields, k: v.float()}
        return cls(
            dates=np.asarray(dates),
            stocks=np.asarray(stocks),
            open=fields["open"],
            high=fields["high"],
            low=fields["low"],
            close=fields["close"],
            volume=fields["volume"],
            vwap=fields["vwap"],
            mask=mask.bool(),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def n_dates(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_stocks(self) -> int:
        return int(self.mask.shape[1])

    @property
    def device(self) -> torch.device:
        return self.close.device

    @property
    def returns(self) -> torch.Tensor:
        """Daily simple returns, masked at non-tradable cells."""
        prev = torch.roll(self.close, shifts=1, dims=0)
        prev[0] = self.close[0]                     # avoid div-by-zero on day 0
        ret = self.close / prev - 1.0
        ret[0] = 0.0
        valid = self.mask & torch.roll(self.mask, shifts=1, dims=0)
        valid[0] = False
        return ret.masked_fill(~valid, 0.0)

    def to(self, device: torch.device | str) -> "Panel":
        return Panel(
            dates=self.dates,
            stocks=self.stocks,
            open=self.open.to(device),
            high=self.high.to(device),
            low=self.low.to(device),
            close=self.close.to(device),
            volume=self.volume.to(device),
            vwap=self.vwap.to(device),
            mask=self.mask.to(device),
        )

    def slice_dates(self, start: int, end: int) -> "Panel":
        return Panel(
            dates=self.dates[start:end],
            stocks=self.stocks,
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
            vwap=self.vwap[start:end],
            mask=self.mask[start:end],
        )

    # ------------------------------------------------------------------
    # Sanity checks (used by tests + CLI ``mlquant check``)
    # ------------------------------------------------------------------
    def assert_consistent(self) -> None:
        T, N = self.mask.shape
        assert self.dates.shape == (T,), f"dates {self.dates.shape}"
        assert self.stocks.shape == (N,), f"stocks {self.stocks.shape}"
        for name in ("open", "high", "low", "close", "volume", "vwap"):
            field = getattr(self, name)
            assert field.shape == (T, N), f"{name} {field.shape}"
        # H >= O,C,L and L <= O,C,H on tradable cells
        m = self.mask
        ohlc = torch.stack([self.open, self.high, self.low, self.close], dim=0)
        assert ((self.high >= ohlc[[0, 2, 3]].max(dim=0).values) | ~m).all(), "high < o/l/c"
        assert ((self.low  <= ohlc[[0, 1, 3]].min(dim=0).values) | ~m).all(), "low  > o/h/c"


def fields_iter(panel: Panel) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every numeric panel field."""
    for name in ("open", "high", "low", "close", "volume", "vwap"):
        yield name, getattr(panel, name)
