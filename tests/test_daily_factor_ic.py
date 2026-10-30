import pytest
import torch
from scipy.stats import pearsonr, spearmanr

from mlquant.backtest.factor_ic import daily_factor_ic
from scripts.eval_factor_ic import close_to_close_labels, summarize
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from dataclasses import replace


def test_daily_coefficients_match_independent_scipy_with_ties_and_pair_masks():
    torch.manual_seed(26)
    values = torch.randint(-3, 4, (9, 31)).double()
    targets = torch.randn(9, 31, dtype=torch.float64)
    factor_mask, label_mask = torch.rand(9, 31) > .2, torch.rand(9, 31) > .3
    values[0, 0], targets[1, 1] = float("nan"), float("inf")
    result = daily_factor_ic(values, targets, factor_mask, label_mask)
    for t in range(9):
        pair = factor_mask[t] & label_mask[t] & values[t].isfinite() & targets[t].isfinite()
        x, y = values[t, pair].numpy(), targets[t, pair].numpy()
        assert result.count[t] == len(x)
        assert result.valid[t]
        assert float(result.ic[t]) == pytest.approx(pearsonr(x, y).statistic, abs=1e-12)
        assert float(result.rank_ic[t]) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    poisoned = values.clone()
    poisoned[~factor_mask | ~label_mask] = 1e30
    other = daily_factor_ic(poisoned, targets, factor_mask, label_mask)
    torch.testing.assert_close(result.ic, other.ic, atol=0, rtol=0)
    torch.testing.assert_close(result.rank_ic, other.rank_ic, atol=0, rtol=0)


def test_pooled_dates_can_reverse_the_cross_sectional_sign():
    values = torch.tensor([[1., 2., 3.], [101., 102., 103.]])
    targets = torch.tensor([[-1., -2., -3.], [99., 98., 97.]])
    mask = torch.ones_like(values, dtype=torch.bool)
    result = daily_factor_ic(values, targets, mask, mask)
    assert pearsonr(values.flatten().numpy(), targets.flatten().numpy()).statistic > .99
    torch.testing.assert_close(result.ic, torch.tensor([-1., -1.], dtype=torch.float64))
    torch.testing.assert_close(result.rank_ic, result.ic)


def test_constant_and_too_small_dates_are_missing_not_zero():
    values = torch.tensor([[1., 1., 1.], [1., 2., 3.], [1., 2., 3.]])
    targets = torch.tensor([[1., 2., 3.], [1., 2., 3.], [3., 2., 1.]])
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[1, 2] = False
    result = daily_factor_ic(values, targets, mask, mask)
    assert result.valid.tolist() == [False, False, True]
    assert result.ic[:2].isnan().all()
    report = summarize(result)
    assert report["daily_ic"] == [None, None, -1.]
    assert report["ic"]["mean"] == -1.
    assert report["ic"]["std"] is None


def test_unavailable_close_endpoints_do_not_create_false_returns():
    panel = make_synthetic_panel(SyntheticConfig(n_dates=100, n_stocks=3))
    close, mask = panel.close.clone(), panel.mask.clone()
    close[20, 0] = 0
    close[21, 1] = float("inf")
    mask[30, 2] = False
    target, valid = close_to_close_labels(replace(panel, close=close, mask=mask))
    for t, stock in [(19, 0), (20, 0), (20, 1), (21, 1), (29, 2), (30, 2)]:
        assert not valid[t, stock]
        assert target[t, stock].isnan()
    assert not valid[-1].any()
    assert target[-1].isnan().all()


def test_no_gradients_and_extreme_finite_inputs():
    values = torch.tensor([[-1e300, 0, 1e300]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    result = daily_factor_ic(values, -values, mask, mask)
    assert not result.ic.requires_grad
    assert float(result.ic[0]) == pytest.approx(-1.)
    assert float(result.rank_ic[0]) == pytest.approx(-1.)


@pytest.mark.parametrize("bad", [1, True, 2.5])
def test_min_assets_validation(bad):
    values = torch.ones(2, 3)
    with pytest.raises(ValueError):
        daily_factor_ic(values, values, values.bool(), values.bool(), min_assets=bad)


def test_mask_shape_and_dtype_validation():
    values = torch.ones(2, 3)
    with pytest.raises(TypeError):
        daily_factor_ic(values, values, values, values.bool())
    with pytest.raises(ValueError):
        daily_factor_ic(values, values, values.bool(), torch.ones(2, 2, dtype=torch.bool))
