"""The :class:`Panel` — a thin, mask-aware container for OCHLV panels.

Design rules
------------
* Every numeric field has shape ``[T, N]`` (date × stock) and dtype
  ``float32``.
* ``mask[t, i] == True`` iff stock *i* was tradable on date *t*. Limit-up,
  limit-down, halted, and pre-IPO cells are masked out — this is the
  "bias correction" plumbing the paper relies on.
* Panels are *immutable*: factor pipelines return new tensors rather
  than mutating in place. This keeps reasoning straightforward and is
  a hard requirement for autograd-backed factors.

Optional A-share microstructure fields
--------------------------------------
The legacy ``cuda_features.py`` operates on Wind dumps that carry a few
extra columns beyond OCHLV: traded ``amount`` (turnover in CNY), the
exchange-published ``limit_up`` / ``limit_down`` prices and the previous
session's ``last_close``. They are *optional* on this dataclass — the
synthetic generator emits them, real-data loaders fill them when the
file has the columns, and downstream code falls back to derived
quantities when they are absent (e.g. ±10 % return proxy for limit
moves).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


_REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "vwap")
_OPTIONAL_FIELDS = ("amount", "limit_up", "limit_down", "last_close")


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

    # ------- optional A-share microstructure fields -----------------------
    # All [T, N] float32 when present; ``None`` means "data source did not
    # supply this column — fall back to derived defaults".
    amount:     Optional[torch.Tensor] = field(default=None)
    limit_up:   Optional[torch.Tensor] = field(default=None)
    limit_down: Optional[torch.Tensor] = field(default=None)
    last_close: Optional[torch.Tensor] = field(default=None)

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
        missing = set(_REQUIRED_FIELDS) - set(fields)
        if missing:
            raise KeyError(f"Panel.from_tensors missing fields: {sorted(missing)}")
        T, N = mask.shape
        promoted = {}
        for k, v in fields.items():
            if k not in _REQUIRED_FIELDS and k not in _OPTIONAL_FIELDS:
                # Allow forward-compat fields to round-trip silently.
                continue
            if tuple(v.shape) != (T, N):
                raise ValueError(f"field {k!r} has shape {tuple(v.shape)}, expected {(T, N)}")
            promoted[k] = v.float() if v.dtype != torch.float32 else v
        kwargs = {k: promoted[k] for k in _REQUIRED_FIELDS}
        for k in _OPTIONAL_FIELDS:
            if k in promoted:
                kwargs[k] = promoted[k]
        return cls(
            dates=np.asarray(dates),
            stocks=np.asarray(stocks),
            mask=mask.bool(),
            **kwargs,
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
        """Daily simple returns, masked at non-tradable cells.

        When :attr:`last_close` is present we use it directly so the day-0
        return is meaningful and adjusted-vs-unadjusted close mismatches
        do not inject NaNs. Otherwise we fall back to the previous-row
        close.
        """
        if self.last_close is not None:
            prev = self.last_close.clamp_min(1e-9)
            ret = self.close / prev - 1.0
            valid = self.mask & (self.last_close > 0)
            return ret.masked_fill(~valid, 0.0)
        prev = torch.roll(self.close, shifts=1, dims=0)
        prev[0] = self.close[0]                     # avoid div-by-zero on day 0
        ret = self.close / prev - 1.0
        ret[0] = 0.0
        valid = self.mask & torch.roll(self.mask, shifts=1, dims=0)
        valid[0] = False
        return ret.masked_fill(~valid, 0.0)

    @property
    def has_real_limits(self) -> bool:
        """True iff exchange-published limit-up / limit-down prices are present."""
        return self.limit_up is not None and self.limit_down is not None

    def field(self, name: str) -> Optional[torch.Tensor]:
        """Generic field accessor — returns ``None`` if the field is absent."""
        return getattr(self, name, None)

    def to(self, device: torch.device | str) -> "Panel":
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.to(device)
        return replace(
            self,
            open=self.open.to(device),
            high=self.high.to(device),
            low=self.low.to(device),
            close=self.close.to(device),
            volume=self.volume.to(device),
            vwap=self.vwap.to(device),
            mask=self.mask.to(device),
            amount=_mv(self.amount),
            limit_up=_mv(self.limit_up),
            limit_down=_mv(self.limit_down),
            last_close=_mv(self.last_close),
        )

    def slice_dates(self, start: int, end: int) -> "Panel":
        def _sl(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t[start:end]
        return replace(
            self,
            dates=self.dates[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
            vwap=self.vwap[start:end],
            mask=self.mask[start:end],
            amount=_sl(self.amount),
            limit_up=_sl(self.limit_up),
            limit_down=_sl(self.limit_down),
            last_close=_sl(self.last_close),
        )

    # ------------------------------------------------------------------
    # Sanity checks (used by tests + CLI ``mlquant check``)
    # ------------------------------------------------------------------
    def assert_consistent(self) -> None:
        T, N = self.mask.shape
        assert self.dates.shape == (T,), f"dates {self.dates.shape}"
        assert self.stocks.shape == (N,), f"stocks {self.stocks.shape}"
        for name in _REQUIRED_FIELDS:
            f_ = getattr(self, name)
            assert f_.shape == (T, N), f"{name} {f_.shape}"
        for name in _OPTIONAL_FIELDS:
            f_ = getattr(self, name)
            if f_ is not None:
                assert f_.shape == (T, N), f"{name} {f_.shape}"
        # H >= O,C,L and L <= O,C,H on tradable cells
        m = self.mask
        ohlc = torch.stack([self.open, self.high, self.low, self.close], dim=0)
        assert ((self.high >= ohlc[[0, 2, 3]].max(dim=0).values) | ~m).all(), "high < o/l/c"
        assert ((self.low  <= ohlc[[0, 1, 3]].min(dim=0).values) | ~m).all(), "low  > o/h/c"


def fields_iter(panel: Panel) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every numeric panel field that is set."""
    for name in _REQUIRED_FIELDS:
        yield name, getattr(panel, name)
    for name in _OPTIONAL_FIELDS:
        v = getattr(panel, name)
        if v is not None:
            yield name, v
