import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_alpha_set, compute_legacy_set, using_rolling_backend
from mlquant.features.legacy_factors import LEGACY_REGISTRY
from mlquant.features.rolling_backend import get_rolling_backend
from mlquant.features.tensor_factors import ts_corr, ts_cov, ts_mean, ts_std
from tests.test_rolling_moments import oracle


@pytest.mark.parametrize("kind,function", [("mean", ts_mean), ("std", ts_std),
                                           ("cov", ts_cov), ("corr", ts_corr)])
def test_scoped_primitives_use_finite_window_oracle(kind, function):
    # Nonfinite observations distinguish actual dispatch from an ignored option.
    x = torch.tensor([[1., 4.], [2., 3.], [float("nan"), 7.], [8., 5.], [4., 6.]])
    y = torch.tensor([[3., 2.], [4., 6.], [1., 8.], [9., 4.], [7., 3.]])
    mask = torch.ones_like(x, dtype=torch.bool)
    with using_rolling_backend("linear"):
        actual, valid = (function(x, y, mask, 3) if kind in ("corr", "cov")
                         else function(x, mask, 3))
    expected, expected_mask = oracle(x.numpy(), y.numpy(), mask.numpy(), 3, kind)
    np.testing.assert_array_equal(valid.numpy(), expected_mask)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7)
    assert get_rolling_backend() == "unfold"


def test_registered_imported_alias_and_factor_engine_selection():
    panel = make_synthetic_panel(SyntheticConfig(n_dates=30, n_stocks=4, seed=7))
    panel.volume[9, 1] = float("nan")
    log_volume = np.log1p(panel.volume.numpy())
    dv = np.zeros_like(log_volume)
    dv[1:] = log_volume[1:] - log_volume[:-1]
    returns = ((panel.close - panel.open) / panel.open).numpy()
    mask = panel.mask.numpy().copy()
    mask[1:] &= panel.mask[:-1].numpy()
    mask[0] = False
    expected, expected_mask = oracle(dv, returns, mask, 4, "corr")
    actual, valid, names = compute_legacy_set(
        panel, names=("stock_001",), neutralize=False, rolling_backend="linear",
    )
    assert names == ["stock_001"]
    np.testing.assert_array_equal(valid.numpy(), expected_mask)
    np.testing.assert_allclose(actual[:, :, 0].numpy(), expected, atol=1e-5)
    with using_rolling_backend("linear"):
        direct, direct_mask = LEGACY_REGISTRY["stock_001"](panel)
    torch.testing.assert_close(actual[:, :, 0], direct)
    assert torch.equal(valid, direct_mask)


def test_constant_masked_rank_series_has_zero_correlation():
    # Three identical FP32 ranks can acquire a nonzero centred residual under
    # direct FP32 reduction. A nested rank can magnify that tiny correlation.
    x = torch.full((4, 1), 0.45)
    y = torch.tensor([[0.1], [0.2], [0.7], [0.9]])
    mask = torch.tensor([[False], [True], [True], [True]])
    with using_rolling_backend("linear"):
        actual, valid = ts_corr(x, y, mask, 4)
    expected, expected_mask = oracle(x.numpy(), y.numpy(), mask.numpy(), 4, "corr")
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(valid.numpy(), expected_mask)


def test_factor_apis_default_to_unfold_and_restore_callers_scope():
    panel = make_synthetic_panel(SyntheticConfig(n_dates=70, n_stocks=5))
    expected = compute_legacy_set(panel, names=("alpha_002", "stock_001"))
    with using_rolling_backend("linear"):
        actual = compute_legacy_set(panel, names=("alpha_002", "stock_001"))
        assert get_rolling_backend() == "linear"
        alpha = compute_alpha_set(panel, names=("alpha002",), rolling_backend="linear")
        legacy = compute_legacy_set(panel, names=("alpha_002",), rolling_backend="linear")
        torch.testing.assert_close(alpha[0], legacy[0], atol=0, rtol=0)
        assert torch.equal(alpha[1], legacy[1])
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    assert torch.equal(actual[1], expected[1])
    assert get_rolling_backend() == "unfold"


def test_nested_selection_exception_and_engine_failure_restore():
    panel = make_synthetic_panel(SyntheticConfig(n_dates=30, n_stocks=2))
    with using_rolling_backend("linear"):
        with pytest.raises(RuntimeError):
            with using_rolling_backend("unfold"):
                assert get_rolling_backend() == "unfold"
                raise RuntimeError("failed operation")
        assert get_rolling_backend() == "linear"
        with pytest.raises(KeyError):
            compute_legacy_set(panel, names=("does_not_exist",), rolling_backend="unfold")
        assert get_rolling_backend() == "linear"
    assert get_rolling_backend() == "unfold"


@pytest.mark.parametrize("backend", [None, "auto", 0, ["linear"]])
def test_invalid_selection(backend):
    with pytest.raises(ValueError, match="rolling_backend"):
        with using_rolling_backend(backend):
            pass
    assert get_rolling_backend() == "unfold"


def test_thread_contexts_are_isolated():
    barrier = Barrier(2)

    def run(backend):
        with using_rolling_backend(backend):
            barrier.wait(timeout=10)
            return get_rolling_backend()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, backend) for backend in ("linear", "unfold")]
        assert [future.result(timeout=15) for future in futures] == ["linear", "unfold"]
    assert get_rolling_backend() == "unfold"


def test_async_contexts_are_isolated():
    async def run(backend):
        with using_rolling_backend(backend):
            await asyncio.sleep(0)
            return get_rolling_backend()

    async def together():
        return await asyncio.gather(run("linear"), run("unfold"))

    assert asyncio.run(together()) == ["linear", "unfold"]
    assert get_rolling_backend() == "unfold"
