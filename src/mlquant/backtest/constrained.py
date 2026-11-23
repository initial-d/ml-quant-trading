"""Row-local historical fills with directional gates and cash-funded buys."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .self_financing import SelfFinancingResult, _run_cash_ledger


@dataclass(repr=False)
class ConstrainedResult(SelfFinancingResult):
    requested_trade_values: np.ndarray   # Complete-fill reference order, currency units
    unfilled_trade_values: np.ndarray    # Signed requested minus executed, currency units
    permission_unfilled: np.ndarray      # First remove forbidden order directions
    cap_unfilled: np.ndarray             # Then clip absolute permitted notional
    cash_unfilled: np.ndarray            # Finally scale permitted purchases for cash
    buy_fill_scale: np.ndarray           # [T] pro-rata scaling after gates/caps
    buy_allowed: np.ndarray
    sell_allowed: np.ndarray
    max_trade_values: np.ndarray | None


def _permission(value, shape, name):
    if value is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean grid matching weights; no broadcasting")
    return result.copy()


def run_constrained_backtest(
    weights, returns, *, buy_allowed=None, sell_allowed=None, max_trade_values=None,
    costs_bps: float = 5.0, initial_equity: float = 1.0, benchmark=None,
) -> ConstrainedResult:
    """Execute target requests without fabricating sales or borrowing cash.

    First mark yesterday's ACTUAL holdings with today's simple returns. Compute
    the hypothetical complete-fill target order using the self-financing NAV
    solve. Gate each direction, optionally cap each absolute trade in currency
    units, execute permitted sales, then scale permitted buys pro rata so buys
    AND their fees fit existing cash plus net sale proceeds. Actual fees apply
    only to fills. Unfilled requests expire; tomorrow's requests are recomputed
    from actual holdings, not queued orders. Blocked assets remain held.

    Targets retain their complete-fill post-fee interpretation as a reference;
    constrained fills can miss both those targets and their cash allocation.
    There is no reoptimization around locked positions. Gates must use only
    current/past information. All arrays share [T,N]; caps are finite,
    nonnegative currency values in the same units as initial_equity (default 1).
    None means unrestricted. Other input/timing rules match self_financing.

    This is a historical mark-price policy with same-row sale proceeds, zero
    cash interest and no settlement, queue, impact, lot-size or broker model.
    """
    return _run_constrained_ledger(weights, returns, buy_allowed=buy_allowed, sell_allowed=sell_allowed,
                                    max_trade_values=max_trade_values, costs_bps=costs_bps,
                                    initial_equity=initial_equity, benchmark=benchmark)


def _run_constrained_ledger(weights, returns, *, buy_allowed=None, sell_allowed=None,
                            max_trade_values=None, costs_bps=5., initial_equity=1.,
                            benchmark=None, decision=None):
    shape = np.asarray(weights).shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("weights must be a nonempty [T,N] grid")
    buy = _permission(buy_allowed, shape, "buy_allowed")
    sell = _permission(sell_allowed, shape, "sell_allowed")
    caps = None
    if max_trade_values is not None:
        caps = np.asarray(max_trade_values, dtype=np.float64).copy()
        if caps.shape != shape or not np.isfinite(caps).all() or np.any(caps < 0):
            raise ValueError("max_trade_values must be a finite nonnegative grid matching weights")
    requested, permission_loss, cap_loss, cash_loss = (np.zeros(shape) for _ in range(4))
    scales = np.ones(shape[0])

    def execute(t, order, marked, cash, rate):
        requested[t] = order
        permitted = np.where(np.where(order > 0, buy[t], sell[t]), order, 0.0)
        permission_loss[t] = order - permitted
        limited = permitted if caps is None else np.clip(permitted, -caps[t], caps[t])
        cap_loss[t] = permitted - limited
        if np.array_equal(limited, order):
            return None  # Exact historical compatibility with the complete-fill ledger.
        sales = -np.minimum(limited, 0.0)
        buys = np.maximum(limited, 0.0)
        available = cash + float(sales.sum()) * (1 - rate)
        buy_total = float(buys.sum())
        scales[t] = min(1.0, available / (1 + rate) / buy_total) if buy_total else 1.0
        filled_buys = buys * scales[t]
        cash_loss[t] = buys - filled_buys
        remaining = max(0.0, available - float(filled_buys.sum()) * (1 + rate))
        return filled_buys - sales, remaining

    base = _run_cash_ledger(weights, returns, costs_bps=costs_bps,
                            initial_equity=initial_equity, benchmark=benchmark, execution=execute,
                            decision=decision)
    unfilled = requested - base.trade_values
    total_requested = float(np.abs(requested).sum())
    total_filled = float(np.abs(base.trade_values).sum())
    base.metrics.update(
        requested_notional_total=total_requested,
        executed_notional_total=total_filled,
        unfilled_notional_total=float(np.abs(unfilled).sum()),
        notional_fill_ratio=total_filled / total_requested if total_requested else 1.0,
        permission_unfilled_notional=float(np.abs(permission_loss).sum()),
        cap_unfilled_notional=float(np.abs(cap_loss).sum()),
        cash_unfilled_notional=float(np.abs(cash_loss).sum()),
        max_target_weight_l1=float(np.abs(base.weights - base.target_weights).sum(axis=1).max()),
    )
    return ConstrainedResult(**vars(base), requested_trade_values=requested,
                             unfilled_trade_values=unfilled, permission_unfilled=permission_loss,
                             cap_unfilled=cap_loss, cash_unfilled=cash_loss, buy_fill_scale=scales,
                             buy_allowed=buy, sell_allowed=sell, max_trade_values=caps)


def panel_trade_permissions(panel):
    """Conservative close-mark gates from observed price, volume and price bands.

    Missing/masked prices or nonpositive/missing volume block both directions.
    A supplied limit-up blocks buys at/above the band; a limit-down blocks sells
    at/below it. Invalid cells in a supplied band block that direction. Absent
    bands impose no inferred +/-10% restriction. Uses no forward label mask.
    A price at a band does not reveal queue fills: this is a declared policy.
    """
    to_array = lambda value: value.detach().cpu().numpy()  # noqa: E731
    close, volume, mask = (to_array(value) for value in (panel.close, panel.volume, panel.mask))
    available = mask & np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
    buy, sell = available.copy(), available.copy()
    for name, direction, compare in (("limit_up", buy, np.less), ("limit_down", sell, np.greater)):
        band = getattr(panel, name)
        if band is not None:
            price = to_array(band)
            direction &= np.isfinite(price) & (price > 0) & compare(close, price)
    metadata = {
        "policy": "observed positive close and volume; directional supplied close-price bands",
        "limit_up_supplied": panel.limit_up is not None,
        "limit_down_supplied": panel.limit_down is not None,
        "missing_band_policy": "absent columns impose no limit proxy; invalid supplied cells block direction",
        "fill_assumption": "permitted orders fill at close mark; same-row net sale proceeds fund buys",
    }
    return buy, sell, metadata
