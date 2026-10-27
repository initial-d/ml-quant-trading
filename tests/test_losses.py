"""Sign-aware losses + IC objectives."""
from __future__ import annotations

import torch
import pytest

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


@pytest.mark.parametrize("loss_type", [ICLoss, RankICLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["prediction", "label", "both", "singleton"])
def test_constant_correlation_batches_have_zero_finite_gradients(loss_type, dtype, case):
    p = [1.] if case == "singleton" else ([1.] * 4 if case in ("prediction", "both") else [1., 2., 3., 4.])
    t = [2.] if case == "singleton" else ([2.] * 4 if case in ("label", "both") else [4., 1., 3., 2.])
    pred = torch.tensor(p, dtype=dtype, requires_grad=True)
    target = torch.tensor(t, dtype=dtype, requires_grad=True)
    loss = loss_type()(pred, target)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(pred.grad, torch.zeros_like(pred))
    assert torch.equal(target.grad, torch.zeros_like(target))


@pytest.mark.parametrize("loss_type", [ICLoss, RankICLoss])
def test_nondegenerate_correlation_values_and_gradients_match_previous_formula(loss_type):
    pred = torch.tensor([.7, -1.2, .3, 2.4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.1, -.8, 1.5, .9], dtype=torch.float64, requires_grad=True)
    loss_fn = loss_type()
    x, y = (pred, target) if loss_type is ICLoss else (loss_fn._soft_rank(pred), loss_fn._soft_rank(target))
    a, b = x - x.mean(), y - y.mean()
    reference = -(a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-12)
    actual = loss_fn(pred, target)
    expected_grad = torch.autograd.grad(reference, (pred, target), retain_graph=True)
    actual_grad = torch.autograd.grad(actual, (pred, target))
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for before, after in zip(expected_grad, actual_grad):
        torch.testing.assert_close(after, before, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("loss_type,option", [(AdjMSELoss, "gamma"), (RankICLoss, "temperature")])
def test_loss_hyperparameters_are_finite_positive(loss_type, option, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        loss_type(**{option: bad})


@pytest.mark.parametrize("loss_type", [ICLoss, RankICLoss])
def test_correlation_losses_reject_empty_or_broadcast_shapes(loss_type):
    with pytest.raises(ValueError, match="same nonempty shape"):
        loss_type()(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError, match="same nonempty shape"):
        loss_type()(torch.ones(3, 1), torch.ones(3))


@pytest.mark.parametrize("loss_type", [ICLoss, RankICLoss])
def test_constant_batch_does_not_poison_the_next_optimizer_step(loss_type):
    weights = torch.nn.Parameter(torch.tensor([1., 1., 1.]))
    optimizer = torch.optim.Adam([weights], lr=.01)
    loss_fn = loss_type()
    for target in [torch.tensor([1., 2., 3.]), torch.ones(3)]:
        optimizer.zero_grad()
        loss_fn(weights, target).backward()
        optimizer.step()
        assert weights.isfinite().all()
    # An unrelated informative step must not inherit NaN Adam moments.
    optimizer.zero_grad()
    (weights - torch.tensor([0., 2., 3.])).square().mean().backward()
    optimizer.step()
    assert weights.isfinite().all()
    assert not torch.equal(weights.detach(), torch.ones(3))
    assert optimizer.state[weights]["exp_avg"].isfinite().all()
    assert optimizer.state[weights]["exp_avg_sq"].isfinite().all()
