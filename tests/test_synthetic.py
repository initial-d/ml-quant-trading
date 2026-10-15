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


def test_returns_bridge_zero_filled_halts():
    """Loaders zero-fill halted prices; returns must not book -100 % / +1e9."""
    from dataclasses import replace

    p = make_synthetic_panel(SyntheticConfig(n_stocks=3, n_dates=20, halt_prob=0.0, seed=4))
    close = p.close.clone()
    mask = p.mask.clone()
    close[5:7, 0] = 0.0                       # two halted days, zero-filled
    mask[5:7, 0] = False
    p = replace(p, close=close, mask=mask, last_close=None, limit_up=None, limit_down=None)

    r = p.returns
    assert r[5, 0].item() == 0.0 and r[6, 0].item() == 0.0
    expected = (close[7, 0] / close[4, 0] - 1.0).item()
    assert abs(r[7, 0].item() - expected) < 1e-6
    assert r.abs().max().item() < 0.5
    assert r[0].abs().sum().item() == 0.0
