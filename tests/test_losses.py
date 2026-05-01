"""Sign-aware losses + IC objectives."""
from __future__ import annotations

import torch

from mlquant.models.losses import AdjMSELoss, ICLoss, RankICLoss


def test_adj_mse_punishes_wrong_sign():
    pred  = torch.tensor([0.5,  0.5])
    label = torch.tensor([0.5, -0.5])      # 1st: same sign,  2nd: opposite
    loss = AdjMSELoss(gamma=0.1)(pred, label)
    # Same-sign squared error is 0; opposite-sign is 1.0 with weight 1.1 -> 1.1
    # Mean across two samples: (0 + 1.1) / 2 = 0.55
    assert abs(loss.item() - 0.55) < 1e-5


def test_ic_loss_perfect_correlation():
    x = torch.linspace(0.0, 1.0, 10)
    loss = ICLoss()(x, x)            # perfect correlation -> -1
    assert abs(loss.item() + 1.0) < 1e-5


def test_rank_ic_loss_negative_correlation_sign():
    x = torch.linspace(0.0, 1.0, 10)
    y = -x
    loss = RankICLoss(temperature=0.05)(x, y)
    # Loss is -(rank corr) ≈ -(-1) = +1
    assert loss.item() > 0.5
