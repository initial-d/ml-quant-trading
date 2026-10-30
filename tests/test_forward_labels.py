"""Forward labels reject unusable prices without confusing zero with validity."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.target import forward_return, target_01
from mlquant.training.dataset import FactorDataset


def panel_with_close(close):
    close = torch.tensor(close, dtype=torch.float32)
    panel = make_synthetic_panel(SyntheticConfig(n_dates=6, n_stocks=2, seed=4))
    return replace(panel, close=close, mask=torch.ones_like(close, dtype=torch.bool))


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_unusable_endpoint_is_not_a_valid_label(bad, endpoint):
    p = panel_with_close([[10., 10.], [11., 10.], [12., 10.]])
    p.close[endpoint, 0] = bad
    values, mask = target_01(p)
    assert not mask[0, 0]
    assert values[0, 0] == 0
    assert torch.isfinite(values).all()
    # A genuine zero return in the control stock still has a valid mask.
    assert values[0, 1] == 0 and mask[0, 1]
    ds = FactorDataset(torch.ones(3, 2, 1), p.mask, values, label_mask=mask)
    assert not ((ds.dates == 0) & (ds.stocks == 0)).any()


def test_masked_nan_does_not_escape_as_nan_times_zero():
    p = panel_with_close([[10., 10.], [float("nan"), 11.], [12., 12.]])
    p.mask[1, 0] = False
    values, mask = target_01(p)
    assert torch.equal(values[:, 0], torch.zeros(3))
    assert not mask[:, 0].any()


def test_positive_finite_prices_with_overflowing_return_are_invalid():
    p = panel_with_close([[torch.finfo(torch.float32).tiny, 10.],
                          [torch.finfo(torch.float32).max, 11.]])
    values, mask = target_01(p)
    assert not mask[0, 0] and values[0, 0] == 0
    assert mask[0, 1] and torch.isfinite(values).all()


def test_multi_row_horizon_uses_endpoint_observations_and_alignment():
    p = panel_with_close([[10., 20.], [11., 22.], [12., 24.], [13., 26.]])
    p.mask[1, 0] = False
    values, mask = forward_return(p, np.int64(2))
    assert mask[0, 0]  # endpoints are valid despite an unavailable middle row
    assert not mask[1, 0]
    assert torch.allclose(values[0], torch.tensor([0.2, 0.2]))
    assert not mask[-2:].any() and not values[-2:].any()


@pytest.mark.parametrize("horizon", [3, 4, 100])
def test_horizon_without_future_rows_returns_empty_validity(horizon):
    p = panel_with_close([[10., 10.], [11., 11.], [12., 12.]])
    values, mask = forward_return(p, horizon)
    assert values.shape == p.close.shape and mask.shape == p.mask.shape
    assert not values.any() and not mask.any()


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, float("nan")])
def test_invalid_horizon_is_rejected(horizon):
    p = panel_with_close([[10., 10.], [11., 11.]])
    with pytest.raises(ValueError, match="positive integer"):
        forward_return(p, horizon)
