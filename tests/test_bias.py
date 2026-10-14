"""Bias-correction limit-day mask."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.bias import limit_move_mask


def test_limit_mask_excludes_extreme_moves():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=80, seed=11))
    # Manually inject a +20% jump on day 5 stock 0; should be masked out.
    p.close[5, 0] = p.close[4, 0] * 1.20
    m = limit_move_mask(p, limit_pct=0.10)
    assert m[5, 0].item() is False or not bool(m[5, 0])

    # Day-0 must be False (no previous close).
    assert not m[0].any()


def test_limit_mask_subset_of_panel_mask():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=80, seed=12))
    m = limit_move_mask(p)
    # halted cells (panel.mask = False) should also be False here
    assert (m & ~p.mask).sum().item() == 0


def test_board_limit_pct_classifies_a_share_boards():
    from mlquant.features.bias import board_limit_pct

    stocks = ["600519", "sz.300750", "688981.SS", "bj.430047", "000001.SZ", "AAPL", "SYN00001"]
    pct = board_limit_pct(stocks, default=0.5)
    assert pct.tolist() == pytest.approx([0.098, 0.198, 0.198, 0.298, 0.098, 0.5, 0.5])


def test_board_limit_pct_applies_chinext_rule_change_by_date():
    import numpy as np

    from mlquant.features.bias import board_limit_pct

    dates = np.array(["2020-08-21", "2020-08-24"], dtype="datetime64[D]")
    pct = board_limit_pct(["300750", "688981"], dates)
    assert pct.shape == (2, 2)
    assert pct[:, 0].tolist() == pytest.approx([0.098, 0.198])   # ChiNext switches
    assert pct[:, 1].tolist() == pytest.approx([0.198, 0.198])   # STAR was always 20 %


def test_limit_mask_accepts_per_stock_thresholds():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=2, n_dates=20, seed=13))
    p = replace(p, limit_up=None, limit_down=None, last_close=None)
    p.close[5, :] = p.close[4, :] * 1.15
    # Stock 0 on a 10 % board (masked), stock 1 on a 20 % board (tradable).
    m = limit_move_mask(p, limit_pct=torch.tensor([0.098, 0.198]))
    assert not bool(m[5, 0])
    assert bool(m[5, 1]) == bool(p.mask[5, 1])
