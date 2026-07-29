from __future__ import annotations

import numpy as np

from scripts.akshare_csi300_full_pipeline import (
    _buffered_top_quantile_weights,
    _optimized_candidate_weights,
    _neutralize_score_matrix,
    _rebalanced_equal_weight,
    _rebalanced_top_quantile_weights,
    _rolling_ic_weighted_scores,
)


def test_neutralize_score_matrix_is_cross_sectional_mean_zero():
    scores = np.array(
        [
            [1.0, 2.0, 3.0, np.nan],
            [2.0, 2.0, 2.0, 5.0],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
        ]
    )

    out = _neutralize_score_matrix(scores, valid)

    assert np.isfinite(out[valid]).all()
    assert np.isnan(out[~valid]).all()
    assert abs(float(out[0, valid[0]].mean())) < 1e-6
    assert abs(float(out[1, valid[1]].mean())) < 1e-6


def test_optimized_candidate_weights_shape_and_budget():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=(40, 8)).astype(np.float32)
    scores = rng.normal(size=(40, 8)).astype(np.float32)
    valid = np.ones((40, 8), dtype=bool)

    weights = _optimized_candidate_weights(
        scores,
        returns,
        valid,
        covariance_window=10,
        candidates=5,
        rebalance_step=5,
        risk_aversion=1.0,
        weight_cap=0.25,
    )

    assert weights.shape == returns.shape
    assert np.all(weights >= -1e-7)
    invested = weights[10:].sum(axis=1)
    assert np.nanmax(invested) <= 1.0001
    assert np.nanmax(invested) > 0.99


def test_rebalanced_equal_weight_holds_between_rebalances():
    valid = np.ones((8, 4), dtype=bool)
    valid[5:, 0] = False

    weights = _rebalanced_equal_weight(valid, rebalance_step=5)

    assert np.allclose(weights[0], [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(weights[1], weights[0])
    assert np.allclose(weights[4], weights[0])
    assert np.allclose(weights[5], [0.0, 1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(weights[7], weights[5])


def test_rebalanced_top_quantile_holds_between_rebalances():
    scores = np.array(
        [
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(scores, dtype=bool)

    weights = _rebalanced_top_quantile_weights(
        scores,
        valid,
        top_quantile=0.25,
        rebalance_step=5,
    )

    assert np.allclose(weights[0], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(weights[4], weights[0])
    assert np.allclose(weights[5], [1.0, 0.0, 0.0, 0.0])


def test_buffered_top_quantile_reduces_churn_inside_exit_band():
    scores = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 2.0, 1.0],
            [2.0, 4.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(scores, dtype=bool)

    weights = _buffered_top_quantile_weights(
        scores,
        valid,
        target_quantile=0.25,
        exit_quantile=0.5,
    )

    assert np.allclose(weights[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(weights[1], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(weights[2], [0.0, 1.0, 0.0, 0.0])


def test_rolling_ic_weighted_scores_do_not_use_future_targets():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(80, 6, 3)).astype(np.float32)
    target = (0.1 * features[:, :, 0] - 0.2 * features[:, :, 1]).astype(np.float32)
    valid = np.ones((80, 6), dtype=bool)

    scores = _rolling_ic_weighted_scores(
        features,
        target,
        valid,
        lookback=30,
        rebalance_step=5,
    )
    changed_future = target.copy()
    changed_future[61:] *= -100.0
    changed_scores = _rolling_ic_weighted_scores(
        features,
        changed_future,
        valid,
        lookback=30,
        rebalance_step=5,
    )

    assert np.allclose(scores[:60], changed_scores[:60], equal_nan=True)
