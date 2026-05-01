"""Bias-correction limit-day mask."""
from __future__ import annotations

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
