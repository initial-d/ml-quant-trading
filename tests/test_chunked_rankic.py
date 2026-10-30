"""Chunked soft rank keeps its objective/derivatives with bounded activations."""
from unittest.mock import patch

import pytest
import torch

from mlquant.models.losses import RankICLoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [.05, 1., 7.])
@pytest.mark.parametrize("chunk", [1, 4, 9])
def test_chunked_loss_and_both_gradients_match_dense_with_ties_and_strides(dtype, temperature, chunk):
    values = torch.tensor([.1, 9., -.7, 9., .1, 9., .9, 9., 1.2, 9., -.4, 9., .5, 9., -.2, 9., 2.1, 9.], dtype=dtype)
    pred = values[::2].requires_grad_()
    target = torch.tensor([.6, -.1, .6, .3, -.7, .2, .9, -.4, .8], dtype=dtype, requires_grad=True)
    assert not pred.is_contiguous()
    dense = RankICLoss(temperature, chunk_size=None)(pred, target)
    actual = RankICLoss(temperature, chunk_size=chunk)(pred, target)
    expected_grads = torch.autograd.grad(dense, (pred, target), retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (pred, target))
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, dense, atol=tolerance, rtol=tolerance)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("constant", ["prediction", "target", "both"])
def test_constant_chunked_batches_keep_zero_gradients(dtype, constant):
    pred = torch.ones(17, dtype=dtype) if constant != "target" else torch.arange(17, dtype=dtype)
    target = torch.ones(17, dtype=dtype) if constant != "prediction" else torch.arange(17, dtype=dtype)
    pred.requires_grad_()
    target.requires_grad_()
    loss = RankICLoss(chunk_size=4)(pred, target)
    loss.backward()
    assert loss.item() == 0
    assert pred.grad.eq(0).all() and target.grad.eq(0).all()


def test_checkpointed_rankic_supports_first_and_second_derivatives():
    pred = torch.tensor([.4, -.2, .7, 1.1], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([-.1, .3, .6, .8], dtype=torch.float64, requires_grad=True)
    loss = RankICLoss(.8, chunk_size=2)
    assert torch.autograd.gradcheck(loss, (pred, target))
    assert torch.autograd.gradgradcheck(loss, (pred, target))


def test_large_cross_section_never_builds_a_dense_pairwise_matrix_or_saves_quadratic_activations():
    n, chunk = 513, 32
    pred = torch.linspace(-2, 2, n, requires_grad=True)
    target = torch.sin(torch.arange(n).float())
    saved, shapes = {}, []
    real_sigmoid = torch.sigmoid

    def keep(tensor):
        storage = tensor.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    def observe(values):
        shapes.append(tuple(values.shape))
        return real_sigmoid(values)

    with patch("torch.sigmoid", side_effect=observe):
        with torch.autograd.graph.saved_tensors_hooks(keep, lambda tensor: tensor):
            loss = RankICLoss(chunk_size=chunk)(pred, target)
        retained_bytes = sum(saved.values())
        loss.backward()
    assert all(rows == n and columns <= chunk for rows, columns in shapes)
    assert len(shapes) > 2 * ((n + chunk - 1) // chunk)  # Backward recomputes blocks.
    assert retained_bytes < 64 * n * pred.element_size()
    assert pred.grad.isfinite().all()
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(keep, lambda tensor: tensor):
        dense = RankICLoss(chunk_size=None)(pred, target)
    assert sum(saved.values()) >= n * n * pred.element_size()
    torch.testing.assert_close(loss, dense, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_chunk_size_is_rejected(bad):
    with pytest.raises(ValueError, match="positive integer or None"):
        RankICLoss(chunk_size=bad)


def test_rankic_rejects_multidimensional_cross_sections():
    with pytest.raises(ValueError, match="one-dimensional"):
        RankICLoss()(torch.ones(2, 3), torch.ones(2, 3))
