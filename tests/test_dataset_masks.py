"""Offline label validity is independent of next-day factor availability."""
from dataclasses import replace

import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.bias import limit_move_mask
from mlquant.features.target import target_01
from mlquant.training.dataset import FactorDataset


def test_observed_limit_close_remains_a_valid_label():
    p = make_synthetic_panel(SyntheticConfig(n_dates=6, n_stocks=2, seed=4))
    close = torch.tensor([[10., 10.], [10., 10.], [11., 10.],
                          [11., 10.], [11., 10.], [11., 10.]])
    p = replace(p, close=close, mask=torch.ones_like(close, dtype=torch.bool),
                last_close=None, limit_up=None, limit_down=None)
    factors = close.unsqueeze(-1)
    factor_mask = limit_move_mask(p)
    labels, label_mask = target_01(p)
    legacy = FactorDataset(factors, factor_mask, labels)
    explicit = FactorDataset(factors, factor_mask, labels, label_mask=label_mask)
    assert p.mask[2, 0] and not factor_mask[2, 0]
    assert not ((legacy.dates == 1) & (legacy.stocks == 0)).any()
    sample = (explicit.dates == 1) & (explicit.stocks == 0)
    assert sample.sum() == 1
    assert torch.allclose(explicit.labels[sample], torch.tensor([0.1]))
    # The limit-day feature itself remains excluded by the research policy.
    assert not ((explicit.dates == 2) & (explicit.stocks == 0)).any()


def test_label_mask_is_aligned_and_rejects_zero_filled_missing_labels():
    factors = torch.arange(8.).reshape(4, 2, 1)
    mask = torch.ones(4, 2, dtype=torch.bool)
    labels = torch.zeros(4, 2)
    label_mask = mask.clone()
    label_mask[1, 0] = False
    ds = FactorDataset(factors, mask, labels, label_mask=label_mask)
    assert len(ds) == 5
    assert not ((ds.dates == 1) & (ds.stocks == 0)).any()
    assert ((ds.dates == 0) & (ds.stocks == 0)).any()
    assert (ds.dates < 3).all()


def test_nonfinite_features_and_labels_never_reach_training():
    factors = torch.ones(4, 2, 2)
    labels = torch.zeros(4, 2)
    factors[0, 0, 1] = float("nan")
    factors[1, 0, 0] = float("inf")
    labels[2, 0] = float("-inf")
    ds = FactorDataset(factors, torch.ones(4, 2, dtype=torch.bool), labels)
    assert ds.stocks.tolist() == [1, 1, 1]
    assert torch.isfinite(ds.features).all() and torch.isfinite(ds.labels).all()


@pytest.mark.parametrize("field", ["mask", "forward_returns", "label_mask"])
def test_mismatched_shapes_are_rejected(field):
    args = dict(factors=torch.ones(4, 2, 1), mask=torch.ones(4, 2, dtype=torch.bool),
                forward_returns=torch.zeros(4, 2), label_mask=torch.ones(4, 2, dtype=torch.bool))
    args[field] = args[field][:, :1]
    with pytest.raises(ValueError, match=field):
        FactorDataset(**args)
