import numpy as np
import pytest
import torch

from mlquant.features import ts_corr_linear, ts_cov_linear, ts_mean_linear, ts_std_linear


def oracle(x, y, mask, window, kind):
    out = np.zeros_like(x)
    valid = np.zeros_like(mask)
    for t in range(window - 1, len(x)):
        for col in range(x.shape[1]):
            sl = slice(t - window + 1, t + 1)
            good = mask[sl, col] & np.isfinite(x[sl, col])
            if kind in ("cov", "corr"):
                good &= np.isfinite(y[sl, col])
            a = x[sl, col][good].astype(np.float64)
            if len(a) < (1 if kind == "mean" else 2):
                continue
            valid[t, col] = True
            if kind == "mean":
                out[t, col] = np.mean(a)
            elif kind == "std":
                out[t, col] = np.std(a, ddof=1)
            else:
                b = y[sl, col][good].astype(np.float64)
                ca, cb = a - np.mean(a), b - np.mean(b)
                cross = np.dot(ca, cb)
                denom = np.linalg.norm(ca) * np.linalg.norm(cb)
                out[t, col] = (cross / (len(a) - 1) if kind == "cov"
                               else cross / max(denom, 1e-12))
    return out, valid


FUNCTIONS = {"mean": ts_mean_linear, "std": ts_std_linear,
             "cov": ts_cov_linear, "corr": ts_corr_linear}


def call(kind, x, y, mask, window):
    return (FUNCTIONS[kind](x, y, mask, window) if kind in ("cov", "corr")
            else FUNCTIONS[kind](x, mask, window))


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("window", [1, 2, 11, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_window_oracle(kind, window, dtype):
    rng = np.random.default_rng(77)
    x = torch.tensor(rng.normal(size=(35, 5)), dtype=dtype)
    y = 0.4 * x + torch.tensor(rng.normal(size=x.shape), dtype=dtype)
    mask = torch.tensor(rng.random(x.shape) > 0.2)
    mask[:, 0] = False
    x[~mask] = float("nan")
    y[~mask] = float("inf")
    x[12, 3], y[9, 2] = float("nan"), float("inf")
    actual, valid = call(kind, x, y, mask, window)
    expected, expected_mask = oracle(x.numpy(), y.numpy(), mask.numpy(), window, kind)
    np.testing.assert_array_equal(valid.numpy(), expected_mask)
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("kind", FUNCTIONS)
def test_large_levels_missing_first_observation_and_causality(kind):
    rng = np.random.default_rng(9)
    x = torch.tensor(1e9 + rng.normal(size=(90, 3)), dtype=torch.float64)
    y = torch.tensor(-2e9 + rng.normal(size=(90, 3)), dtype=torch.float64)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:14, 0] = False
    mask[::7, 1] = False
    mask[:, 2] = False
    mask[70:, 2] = True
    actual, valid = call(kind, x, y, mask, 12)
    expected, _ = oracle(x.numpy(), y.numpy(), mask.numpy(), 12, kind)
    np.testing.assert_allclose(actual.numpy(), expected, atol=5e-7, rtol=1e-6)
    x2, y2, mask2 = x.clone(), y.clone(), mask.clone()
    x2[50:], y2[50:], mask2[50:] = 1e14, -1e14, True
    later, later_mask = call(kind, x2, y2, mask2, 12)
    torch.testing.assert_close(actual[:50], later[:50], atol=0, rtol=0)
    assert torch.equal(valid[:50], later_mask[:50])


def test_constant_windows_and_empty_panels():
    x = torch.ones(16, 3)
    m = torch.ones_like(x, dtype=torch.bool)
    out, valid = ts_corr_linear(x, x, m, 4)
    assert out.eq(0).all() and valid[3:].all()
    for shape in [(0, 3), (4, 0)]:
        x = torch.empty(shape)
        for kind in FUNCTIONS:
            out, mask = call(kind, x, x, torch.ones_like(x, dtype=torch.bool), 3)
            assert out.shape == x.shape and not mask.any()


@pytest.mark.parametrize("kind", FUNCTIONS)
@pytest.mark.parametrize("window", [7, 10, 31])
def test_tiny_windows_after_large_outliers(kind, window):
    # A global first-observation anchor loses tiny post-outlier variances via
    # subtraction of O(1e-2) cumulative squares. Local disjoint summaries must
    # agree with direct windows across block boundaries, holes and later spikes.
    rng = np.random.default_rng(101)
    x = torch.tensor(rng.uniform(1e-10, 8e-9, (127, 3)), dtype=torch.float32)
    y = torch.tensor(rng.uniform(1e-10, 8e-9, (127, 3)), dtype=torch.float32)
    x[0, 0], x[53, 1], y[18, 2] = 0.03, 0.04, 0.02
    mask = torch.tensor(rng.random(x.shape) > 0.15)
    mask[0, 0] = True
    actual, valid = call(kind, x, y, mask, window)
    expected, expected_mask = oracle(x.numpy(), y.numpy(), mask.numpy(), window, kind)
    np.testing.assert_array_equal(valid.numpy(), expected_mask)
    # Corr retains the existing 1e-12 denominator floor for tiny signals.
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-15, rtol=1e-6)


def test_gradients_against_direct_windows():
    torch.manual_seed(14)
    x = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    m = torch.ones_like(x, dtype=torch.bool)
    m[0, 0] = False
    assert torch.autograd.gradcheck(lambda a, b: ts_corr_linear(a, b, m, 5)[0], (x, y))


@pytest.mark.parametrize("window", [0, -1, True, 2.5])
def test_invalid_windows(window):
    with pytest.raises(ValueError, match="positive integer"):
        ts_mean_linear(torch.ones(3, 2), torch.ones(3, 2, dtype=torch.bool), window)


def test_invalid_shapes_dtype():
    with pytest.raises(ValueError, match="boolean mask"):
        ts_std_linear(torch.ones(3, 2), torch.ones(3, 2), 2)
    with pytest.raises(ValueError, match="match x"):
        ts_corr_linear(torch.ones(3, 2), torch.ones(3, 1), torch.ones(3, 2, dtype=torch.bool), 2)
