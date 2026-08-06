"""`cost_drag_cumulative` means what its name says.

The field was called `cost_drag` and sat in a summary row beside `ann_return`,
`gross_ann_return` and `ann_vol`, which are annualised. It is not: it is the
cost paid over the whole backtest. Nothing in the output said so, and
`docs/public_data_validation.md` asks the reader to compare it with those
neighbours.

These tests pin the distinction so the name and the quantity cannot drift apart
again.
"""
from __future__ import annotations

import numpy as np

from mlquant.backtest.engine import run_backtest


def _alternating_book(periods: int, n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A book that fully rebalances every period, so cost accrues steadily."""
    weights = np.zeros((periods, n))
    weights[0::2, 0] = 1.0
    weights[1::2, 1] = 1.0
    returns = np.zeros((periods, n))
    return weights, returns


def test_cost_drag_is_cumulative_not_per_period():
    """Doubling the backtest doubles the cost, because the field is a total.

    This is the check that distinguishes a cumulative quantity from an
    annualised one: an annualised figure would be unchanged.
    """
    short = run_backtest(*_alternating_book(200), costs_bps=5.0)
    long = run_backtest(*_alternating_book(400), costs_bps=5.0)

    assert short.cost_drag_cumulative > 0
    ratio = long.cost_drag_cumulative / short.cost_drag_cumulative
    assert 1.95 < ratio < 2.05, f"expected ~2x over twice the periods, got {ratio:.3f}"

    # Turnover is a rate and must not move with the length of the run.
    assert np.isclose(
        short.metrics["turnover"], long.metrics["turnover"], rtol=0.05
    ), "turnover moved with run length; the fixture is not comparable"


def test_cost_drag_scales_with_the_fee():
    cheap = run_backtest(*_alternating_book(200), costs_bps=1.0)
    dear = run_backtest(*_alternating_book(200), costs_bps=10.0)
    assert np.isclose(dear.cost_drag_cumulative, 10 * cheap.cost_drag_cumulative)


def test_legacy_alias_still_reads():
    """Archived reports and older scripts use `cost_drag`; both must work."""
    res = run_backtest(*_alternating_book(120), costs_bps=5.0)

    assert res.cost_drag == res.cost_drag_cumulative
    assert res.metrics["cost_drag"] == res.metrics["cost_drag_cumulative"]
    assert "cost_drag_cumulative" in res.metrics
