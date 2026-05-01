"""Sanity properties of the synthetic OCHLV panel."""
from __future__ import annotations

import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel


def test_panel_shape_and_dtypes():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=20, n_dates=80, seed=1))
    assert p.close.shape == (80, 20)
    assert p.close.dtype == torch.float32
    p.assert_consistent()


def test_no_negative_prices():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=50, n_dates=200, seed=2))
    valid = p.mask
    assert (p.close[valid] > 0).all()
    assert (p.open[valid]  > 0).all()
    assert (p.high[valid]  > 0).all()
    assert (p.low[valid]   > 0).all()


def test_returns_inside_limit():
    p = make_synthetic_panel(SyntheticConfig(n_stocks=50, n_dates=200, limit_pct=0.10, seed=3))
    r = p.returns
    valid = p.mask & torch.roll(p.mask, 1, 0)
    valid[0] = False
    assert r[valid].abs().max().item() <= 0.105      # tiny tolerance for float


def test_seed_determinism():
    a = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=30, seed=7))
    b = make_synthetic_panel(SyntheticConfig(n_stocks=10, n_dates=30, seed=7))
    assert torch.equal(a.close, b.close)
    assert torch.equal(a.mask,  b.mask)
