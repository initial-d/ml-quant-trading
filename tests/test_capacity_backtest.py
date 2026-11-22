"""Capital-pressure, strictly lagged liquidity and frozen-input CLI controls."""
import json
import pickle

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from mlquant.backtest import lagged_turnover_caps, run_capacity_backtest, run_constrained_backtest
from mlquant.cli.main import cli


def test_caps_use_past_window_exclude_current_and_future_turnover():
    turnover = np.array([[100.], [200.], [9999.], [300.]])
    caps, ready = lagged_turnover_caps(turnover, participation_rate=.1, lookback=2)
    np.testing.assert_allclose(caps[:, 0], [0, 0, 15, 509.95])
    np.testing.assert_array_equal(ready[:, 0], [False, False, True, True])
    changed = turnover.copy(); changed[2:] *= 1e5
    changed_caps, _ = lagged_turnover_caps(changed, participation_rate=.1, lookback=2)
    np.testing.assert_array_equal(caps[:3], changed_caps[:3])


def test_missing_history_differs_from_observed_zero_and_old_outlier_leaves_window():
    values = np.array([[1e30, 0], [10., np.nan], [20., 0], [30., 0], [40., 1]])
    caps, ready = lagged_turnover_caps(values, participation_rate=.1, lookback=2)
    assert caps[3, 0] == pytest.approx(1.5)  # Old 1e30 is outside this window.
    assert not ready[3, 1] and caps[3, 1] == 0
    assert ready[4, 1] and caps[4, 1] == 0
    huge, _ = lagged_turnover_caps(np.full((4, 1), 1e308), participation_rate=1, lookback=3)
    assert np.isfinite(huge[3, 0]) and huge[3, 0] == pytest.approx(1e308)


def test_hand_computed_capital_pressure_preserves_all_scenarios_and_pre_oos_liquidity():
    weights = np.ones((4, 1)); returns = np.zeros_like(weights); turnover = np.full_like(weights, 100)
    result = run_capacity_backtest(weights, returns, turnover, [5, 100, 1000],
                                   start_index=2, participation_rate=.1, lookback=2, costs_bps=0)
    assert result.trade_caps[0, 0] == 10 and result.liquidity_ready.all()
    np.testing.assert_array_equal(result.capitals, [5, 100, 1000])
    np.testing.assert_allclose(result.scenarios[0].trade_values[:, 0], [5, 0])
    np.testing.assert_allclose(result.scenarios[1].trade_values[:, 0], [10, 10])
    np.testing.assert_allclose(result.scenarios[2].trade_values[:, 0], [10, 10])
    assert result.rows[0]['notional_fill_ratio'] == 1 and result.rows[0]['mean_cash_weight'] == 0
    assert result.rows[1]['notional_fill_ratio'] == pytest.approx(20 / 190)
    assert result.rows[2]['notional_fill_ratio'] == pytest.approx(20 / 1990)
    assert result.rows[1]['mean_cash_weight'] == pytest.approx(.85)
    assert result.rows[2]['mean_cash_weight'] == pytest.approx(.985)
    assert result.rows[1]['mean_target_weight_l1'] == pytest.approx(.85)


def test_reference_uses_same_permission_gates_without_caps():
    weights = np.array([[1., 0.], [0., 1.], [0., 1.]])
    returns = np.array([[0., 0.], [0., 0.], [-.2, .1]])
    sell = np.ones_like(weights, dtype=bool); sell[1, 0] = False
    result = run_capacity_backtest(weights, returns, np.full_like(weights, 1e9), [10, 100],
                                   sell_allowed=sell, lookback=1, participation_rate=1, costs_bps=0)
    reference = run_constrained_backtest(weights, returns, sell_allowed=sell, costs_bps=0)
    np.testing.assert_array_equal(result.reference.cumulative_equity, reference.cumulative_equity)
    assert result.reference.trade_values[1, 0] == 0
    # The sweep has an initial warm-up cap of zero; its reference has no caps.
    assert result.scenarios[0].trade_values[0].sum() == 0


def test_liberal_caps_match_reference_when_pre_oos_history_is_available():
    rng = np.random.default_rng(4)
    weights = rng.dirichlet(np.ones(4), size=35)[:, :3]
    returns = rng.uniform(-.1, .1, weights.shape)
    buy, sell = rng.random(weights.shape) > .1, rng.random(weights.shape) > .1
    result = run_capacity_backtest(weights, returns, np.full_like(weights, 1e10), [1, 1e5],
                                   start_index=5, lookback=5, buy_allowed=buy, sell_allowed=sell)
    for capital, scenario in zip(result.capitals, result.scenarios):
        np.testing.assert_allclose(scenario.cumulative_equity / capital, result.reference.cumulative_equity, atol=1e-14)
        np.testing.assert_allclose(scenario.weights, result.reference.weights, atol=1e-14)
        assert scenario.metrics['cap_unfilled_notional'] == 0


def test_monetary_unit_scaling_asset_permutation_and_inputs_unchanged():
    rng = np.random.default_rng(3)
    weights = rng.dirichlet(np.ones(4), size=30)[:, :3]
    returns = rng.uniform(-.1, .1, weights.shape)
    turnover = rng.uniform(100, 2000, weights.shape)
    buy, sell = rng.random(weights.shape) > .2, rng.random(weights.shape) > .2
    saved = [value.copy() for value in (weights, returns, turnover, buy, sell)]
    options = dict(start_index=3, lookback=3, participation_rate=.1, buy_allowed=buy, sell_allowed=sell)
    base = run_capacity_backtest(weights, returns, turnover, [100, 1000], **options)
    scaled = run_capacity_backtest(weights, returns, turnover * 1e3, [1e5, 1e6], **options)
    permuted = run_capacity_backtest(weights[:, ::-1], returns[:, ::-1], turnover[:, ::-1], [100, 1000],
                                     start_index=3, lookback=3, participation_rate=.1,
                                     buy_allowed=buy[:, ::-1], sell_allowed=sell[:, ::-1])
    for scenario, large, reordered in zip(base.scenarios, scaled.scenarios, permuted.scenarios):
        np.testing.assert_allclose(scenario.weights, large.weights, atol=1e-14)
        np.testing.assert_allclose(scenario.cumulative_equity, large.cumulative_equity / 1e3, atol=1e-11)
        np.testing.assert_allclose(scenario.weights[:, ::-1], reordered.weights, atol=1e-14)
    for value, previous in zip((weights, returns, turnover, buy, sell), saved):
        np.testing.assert_array_equal(value, previous)


def test_future_changes_do_not_rewrite_scenario_prefix_and_no_current_turnover_lookahead():
    weights = np.tile([.4, .5], (20, 1)); returns = np.full_like(weights, .01)
    turnover = np.full_like(weights, 1000); buy = np.ones_like(weights, dtype=bool)
    original = run_capacity_backtest(weights, returns, turnover, [100, 10000], lookback=3,
                                     start_index=3, buy_allowed=buy)
    turnover[10:] *= 1e5
    changed_turnover = run_capacity_backtest(weights, returns, turnover, [100, 10000], lookback=3,
                                             start_index=3, buy_allowed=buy)
    for actual, changed in zip(original.scenarios, changed_turnover.scenarios):
        # Changes at row 10 cannot affect its own cap/order (OOS row 7).
        np.testing.assert_array_equal(actual.trade_values[:8], changed.trade_values[:8])
    weights[10:] = 0; returns[10:] = -.5; buy[10:] = False
    changed_all = run_capacity_backtest(weights, returns, turnover, [100, 10000], lookback=3,
                                        start_index=3, buy_allowed=buy)
    for actual, changed in zip(original.scenarios, changed_all.scenarios):
        np.testing.assert_array_equal(actual.cumulative_equity[:7], changed.cumulative_equity[:7])


def test_zero_rate_warmup_bankruptcy_and_benchmark_are_explicit():
    weights = np.ones((4, 1)); returns = np.zeros_like(weights)
    empty = run_capacity_backtest(weights, returns, weights * 100, [1], participation_rate=0,
                                  start_index=2, lookback=2)
    assert empty.rows[0]['mean_cash_weight'] == 1 and empty.rows[0]['notional_fill_ratio'] == 0
    cold = run_capacity_backtest(weights, returns, weights * 100, [1], lookback=10)
    assert not cold.liquidity_ready.any()
    returns[3] = -1; weights[3] = 0
    result = run_capacity_backtest(weights, returns, weights * 100, [1], start_index=2, lookback=1,
                                   participation_rate=1, costs_bps=0, benchmark=np.zeros(4))
    assert result.rows[0]['bankrupt_index'] == 1
    assert result.rows[0]['n_positive_nav_dates'] == 1
    assert result.rows[0]['final_equity_multiple'] == 0
    assert 'info_ratio' in result.rows[0]['metrics']


@pytest.mark.parametrize('options', [
    {'participation_rate': -1}, {'participation_rate': 1.1}, {'participation_rate': float('nan')},
    {'participation_rate': True}, {'lookback': 0}, {'lookback': True}, {'lookback': 1.5},
])
def test_invalid_liquidity_options_rejected(options):
    with pytest.raises(ValueError):
        lagged_turnover_caps([[100]], **options)


@pytest.mark.parametrize('turnover', [[], [100], [[-1]], [[float('inf')]]])
def test_invalid_turnover_rejected(turnover):
    with pytest.raises(ValueError):
        lagged_turnover_caps(turnover)


@pytest.mark.parametrize('capitals', [[], [0], [-1], [float('nan')], [float('inf')], [1, 1], [[1]], 1])
def test_invalid_capital_grid_rejected(capitals):
    with pytest.raises(ValueError):
        run_capacity_backtest([[1]], [[0]], [[100]], capitals)


@pytest.mark.parametrize('options', [
    {'start_index': -1}, {'start_index': 1}, {'start_index': True}, {'start_index': .1},
    {'benchmark': [0, 0]}, {'benchmark': [float('nan')]}, {'costs_bps': -1},
])
def test_invalid_replay_options_rejected(options):
    with pytest.raises(ValueError):
        run_capacity_backtest([[1]], [[0]], [[100]], [1], **options)


def prepare_cli(tmp_path, amount=True):
    art = tmp_path / 'art'; art.mkdir()
    close = torch.full((6, 2), 10.)
    blob = {'dates': np.arange('2026-01-01', '2026-01-07', dtype='datetime64[D]'),
            'stocks': ['a', 'b'], 'open': close, 'high': close, 'low': close, 'close': close,
            'vwap': close, 'volume': torch.ones_like(close), 'mask': torch.ones_like(close, dtype=torch.bool)}
    if amount:
        blob['amount'] = torch.full_like(close, 100.)
    torch.save(blob, art / 'panel.pt')
    torch.save({'weights': torch.full_like(close, .5)}, art / 'weights.pt')
    torch.save({'oos_start': 2}, art / 'predictions.pt')
    (art / 'summary.json').write_text('ordinary backtest sentinel')
    (art / 'backtest.pkl').write_text('ordinary ledger sentinel')
    cfg = tmp_path / 'config.yaml'; cfg.write_text(f'artifacts_dir: {art}\ncosts_bps: 0\n')
    args = ['capacity', '--config', str(cfg), '--capital', '100', '--capital', '1000',
            '--lookback', '2', '--participation-rate', '.1', '--currency', 'CNY',
            '--data-category', 'synthetic', '--data-source', 'constructed test panel']
    return art, args


def test_cli_frozen_artifacts_unit_conversion_oos_reports_and_full_ledgers(tmp_path):
    art, args = prepare_cli(tmp_path)
    outcome = CliRunner().invoke(cli, args)
    assert outcome.exit_code == 0, outcome.output
    report = json.loads((art / 'capacity/summary.json').read_text())
    assert report['category'] == 'synthetic' and report['currency'] == 'CNY'
    assert report['evaluation']['start_index'] == report['evaluation']['n_pre_evaluation_rows'] == 2
    assert report['evaluation']['n_dates'] == 4
    assert report['liquidity_coverage']['ready_cells'] == 8
    assert [row['initial_capital'] for row in report['scenarios']] == [100, 1000]
    assert set(report['input_sha256']) == {'panel.pt', 'weights.pt', 'predictions.pt'}
    assert (art / 'summary.json').read_text() == 'ordinary backtest sentinel'
    assert (art / 'backtest.pkl').read_text() == 'ordinary ledger sentinel'
    with (art / 'capacity/capacity.pkl').open('rb') as handle:
        ledger = pickle.load(handle)
    assert ledger.trade_caps.shape == (4, 2) and ledger.trade_caps[0, 0] == 10
    assert 'Mean cash weight' in (art / 'capacity/summary.md').read_text()
    scaled_out = tmp_path / 'scaled'
    scaled = CliRunner().invoke(cli, args + ['--amount-scale', '1000', '--output-dir', str(scaled_out)])
    assert scaled.exit_code == 0, scaled.output
    with (scaled_out / 'capacity.pkl').open('rb') as handle:
        changed = pickle.load(handle)
    np.testing.assert_allclose(changed.trade_caps, ledger.trade_caps * 1000)


def test_cli_refuses_share_volume_fallback_and_pipeline_output_overwrite(tmp_path):
    art, args = prepare_cli(tmp_path, amount=False)
    missing = CliRunner().invoke(cli, args)
    assert missing.exit_code != 0 and 'share volume is not a substitute' in missing.output
    blob = torch.load(art / 'panel.pt', weights_only=False); blob['amount'] = torch.full((6, 2), 100.)
    torch.save(blob, art / 'panel.pt')
    collision = CliRunner().invoke(cli, args + ['--output-dir', str(art)])
    assert collision.exit_code != 0 and 'separate directory' in collision.output
    assert (art / 'summary.json').read_text() == 'ordinary backtest sentinel'
