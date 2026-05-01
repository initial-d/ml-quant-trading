"""Cross-check tensor primitives against pandas / numpy reference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mlquant.features.tensor_factors import (
    cs_rank, cs_zscore,
    delay, delta,
    ewma,
    ts_corr, ts_max, ts_mean, ts_min, ts_std, ts_sum,
)


def _to_df(x: torch.Tensor, mask: torch.Tensor) -> pd.DataFrame:
    arr = x.cpu().numpy().astype(float)
    arr[~mask.cpu().numpy()] = np.nan
    return pd.DataFrame(arr)


def test_cs_rank_matches_pandas(random_panel_tensors):
    x, mask = random_panel_tensors
    out, _ = cs_rank(x, mask)

    df = _to_df(x, mask)
    expected = df.rank(axis=1, pct=True, method="average")
    got = out.cpu().numpy()
    valid = mask.cpu().numpy()

    np.testing.assert_allclose(got[valid], expected.to_numpy()[valid], rtol=1e-4, atol=1e-5)


def test_cs_zscore_zero_mean_unit_var(random_panel_tensors):
    x, mask = random_panel_tensors
    out, _ = cs_zscore(x, mask)
    arr = out.cpu().numpy()
    valid = mask.cpu().numpy()
    for t in range(arr.shape[0]):
        row = arr[t][valid[t]]
        if row.size > 1:
            assert abs(row.mean()) < 1e-5
            assert abs(row.std(ddof=0) - 1.0) < 5e-2     # n vs n-1 differences


def test_ts_sum_mean_std_match_pandas(random_panel_tensors):
    x, mask = random_panel_tensors
    window = 5
    s, _ = ts_sum(x, mask, window)
    m, _ = ts_mean(x, mask, window)
    d, _ = ts_std(x, mask, window)

    df = _to_df(x, mask)
    pd_sum  = df.rolling(window, min_periods=window).sum()
    pd_mean = df.rolling(window, min_periods=window).mean()
    pd_std  = df.rolling(window, min_periods=window).std()

    valid_sum  = ~np.isnan(pd_sum.to_numpy())
    np.testing.assert_allclose(s.cpu().numpy()[valid_sum],  pd_sum.to_numpy()[valid_sum],  rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(m.cpu().numpy()[valid_sum],  pd_mean.to_numpy()[valid_sum], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(d.cpu().numpy()[valid_sum],  pd_std.to_numpy()[valid_sum],  rtol=1e-2, atol=1e-3)


def test_ts_min_max_match_pandas(random_panel_tensors):
    x, mask = random_panel_tensors
    lo, _ = ts_min(x, mask, 7)
    hi, _ = ts_max(x, mask, 7)

    df = _to_df(x, mask)
    valid = ~np.isnan(df.rolling(7, min_periods=7).min().to_numpy())
    np.testing.assert_allclose(lo.cpu().numpy()[valid], df.rolling(7, min_periods=7).min().to_numpy()[valid], rtol=1e-4)
    np.testing.assert_allclose(hi.cpu().numpy()[valid], df.rolling(7, min_periods=7).max().to_numpy()[valid], rtol=1e-4)


def test_ts_corr_matches_pandas(random_panel_tensors):
    x, mask = random_panel_tensors
    rng = np.random.default_rng(1)
    y = torch.tensor(rng.normal(size=x.shape), dtype=torch.float32)
    out, _ = ts_corr(x, y, mask, 10)

    dfx = _to_df(x, mask)
    dfy = _to_df(y, mask)
    expected = dfx.rolling(10, min_periods=10).corr(dfy)
    valid = ~np.isnan(expected.to_numpy())
    np.testing.assert_allclose(out.cpu().numpy()[valid], expected.to_numpy()[valid], rtol=1e-2, atol=1e-3)


def test_delay_and_delta(random_panel_tensors):
    x, mask = random_panel_tensors
    d, _ = delay(x, mask, 3)
    assert torch.equal(d[3:], x[:-3] * mask[:-3].float())
    diff, _ = delta(x, mask, 1)
    expected = (x[1:] - x[:-1])
    valid = (mask[1:] & mask[:-1])
    np.testing.assert_allclose(
        diff[1:].cpu().numpy()[valid.cpu().numpy()],
        expected.cpu().numpy()[valid.cpu().numpy()],
        rtol=1e-5, atol=1e-6,
    )


def test_ewma_recurrence():
    x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(x, dtype=torch.bool)
    out, _ = ewma(x, mask, alpha=0.5)
    # Closed form for alpha=0.5: y_t = 0.5 * x_t + 0.5 * y_{t-1}
    expected = []
    last = 0.0
    for v in x.squeeze().tolist():
        last = 0.5 * v + 0.5 * last
        expected.append(last)
    np.testing.assert_allclose(out.squeeze().cpu().numpy(), expected, rtol=1e-5)
