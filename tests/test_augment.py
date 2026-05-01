"""Properties of the GBM data-augmentation pass."""
from __future__ import annotations

import torch

from mlquant.training.augment import GBMAugmentConfig, gbm_augment


def test_shape_and_finiteness():
    rng = torch.Generator().manual_seed(0)
    log_r = torch.randn((300, 50), generator=rng) * 0.02
    mask  = torch.ones_like(log_r, dtype=torch.bool)
    aug, am = gbm_augment(log_r, mask, GBMAugmentConfig(n_synthetic=2, block_size=20, seed=1))
    assert aug.shape == (600, 50)
    assert am.shape == aug.shape
    assert torch.isfinite(aug).all()


def test_per_stock_moments_close_to_history():
    rng = torch.Generator().manual_seed(0)
    log_r = torch.randn((1000, 10), generator=rng) * 0.02 + 0.001
    mask  = torch.ones_like(log_r, dtype=torch.bool)
    aug, _ = gbm_augment(log_r, mask, GBMAugmentConfig(n_synthetic=5, block_size=20, seed=2))
    real_mu = log_r.mean(dim=0)
    aug_mu  = aug.mean(dim=0)
    real_sd = log_r.std(dim=0)
    aug_sd  = aug.std(dim=0)
    # Loose tolerances: 5 000 sample bootstrap is noisy by design.
    assert (aug_mu - real_mu).abs().max().item() < 5e-3
    assert (aug_sd / real_sd - 1).abs().max().item() < 0.1


def test_invalid_block_raises():
    log_r = torch.zeros((10, 4))
    mask  = torch.ones_like(log_r, dtype=torch.bool)
    try:
        gbm_augment(log_r, mask, GBMAugmentConfig(block_size=20))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for block_size > T")
