import weakref

import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import compute_legacy_set, iter_legacy_factors, using_rolling_backend
from mlquant.features.legacy_factors import LEGACY_REGISTRY
from mlquant.features.rolling_backend import get_rolling_backend


@pytest.mark.parametrize("backend", ["unfold", "linear"])
@pytest.mark.parametrize("neutralize", [False, True])
def test_stream_preserves_all_factor_values_masks_and_order(backend, neutralize):
    panel = make_synthetic_panel(SyntheticConfig(n_dates=100, n_stocks=8))
    values, joint, names = compute_legacy_set(panel, neutralize=neutralize, rolling_backend=backend)
    masks = []
    for index, (name, actual, mask) in enumerate(iter_legacy_factors(
        panel, neutralize=neutralize, rolling_backend=backend,
    )):
        assert name == names[index]
        torch.testing.assert_close(actual, values[:, :, index], atol=0, rtol=0)
        masks.append(mask)
    assert len(masks) == 213
    assert torch.equal(torch.stack(masks).all(0), joint)


def test_paused_abandoned_and_failed_streams_restore_context(monkeypatch):
    panel = make_synthetic_panel(SyntheticConfig(n_dates=40, n_stocks=3))
    with using_rolling_backend("linear"):
        iterator = iter_legacy_factors(panel, names=("stock_001", "stock_002"), rolling_backend="unfold")
        next(iterator)
        assert get_rolling_backend() == "linear"
        iterator.close()
        assert get_rolling_backend() == "linear"
        def broken(panel):
            raise RuntimeError("factor failed")
        monkeypatch.setitem(LEGACY_REGISTRY, "stock_001", broken)
        with pytest.raises(RuntimeError, match="factor failed"):
            next(iter_legacy_factors(panel, names=("stock_001",), rolling_backend="unfold"))
        assert get_rolling_backend() == "linear"


def test_stream_is_lazy_and_does_not_retain_previous_output(monkeypatch):
    panel = make_synthetic_panel(SyntheticConfig(n_dates=40, n_stocks=3))
    calls, references = [], []
    def factor(panel):
        calls.append(len(calls))
        if references:
            assert references[-1]() is None
        return panel.close + 1, panel.mask
    monkeypatch.setitem(LEGACY_REGISTRY, "stock_001", factor)
    iterator = iter_legacy_factors(panel, names=("stock_001",) * 3, neutralize=False)
    assert not calls
    for _ in range(3):
        _, values, _ = next(iterator)
        references.append(weakref.ref(values))
        del values
    assert len(calls) == 3
    iterator.close()


def test_invalid_stream_request_fails_before_any_factor_computation():
    panel = make_synthetic_panel(SyntheticConfig(n_dates=40, n_stocks=3))
    with pytest.raises(ValueError, match="rolling_backend"):
        iter_legacy_factors(panel, rolling_backend="auto")
    with pytest.raises(KeyError):
        iter_legacy_factors(panel, names=("stock_001", "absent"))
