"""Independent analytic optima, actual-holding feedback and cash-paid plans."""
import json
import pickle

import cvxpy as cp
import numpy as np
import pytest
import torch
from click.testing import CliRunner

from mlquant.portfolio import RebalanceConfig, RebalanceOptimizer
from mlquant.backtest import run_rebalance_backtest
from mlquant.cli.main import cli


def profile(**options):
    return RebalanceConfig(**dict(costs_bps=10., position_value_cap=1., risk_aversion=0., **options))


def test_no_trade_region_and_cash_paid_entry_exit_thresholds():
    optimizer = RebalanceOptimizer(profile())
    history = np.zeros((5, 2))
    weak = optimizer.solve([.0005, 0], history, [0, 0])
    np.testing.assert_array_equal(weak.holding_values_fraction, 0)
    assert weak.fees_fraction == 0 and weak.cash_fraction == 1
    entry = optimizer.solve([.003, 0], history, [0, 0])
    assert entry.holding_values_fraction[0] == pytest.approx(1 / 1.001, abs=1e-7)
    assert entry.holding_values_fraction.sum() + entry.fees_fraction + entry.cash_fraction == pytest.approx(1)
    weak_switch = optimizer.solve([0, .0015], history, [1, 0])
    np.testing.assert_array_equal(weak_switch.holding_values_fraction, [1, 0])
    switch = optimizer.solve([0, .003], history, [1, 0])
    assert switch.holding_values_fraction[1] == pytest.approx(.999 / 1.001, abs=1e-7)
    assert switch.fees_fraction == pytest.approx(.001 * (1 + switch.holding_values_fraction[1]), abs=1e-8)


@pytest.mark.parametrize('mu', [-.004, -.001, 0., .001, .004])
@pytest.mark.parametrize('costs_bps', [0, 10, 100])
def test_one_asset_global_optimum_matches_independent_piecewise_quadratic(mu, costs_bps):
    history = np.tile([[-.01], [.01]], (20, 1))
    variance = float(np.var(history[:, 0], ddof=1))
    risk, current, rate, cash_floor = 10., .4, costs_bps * 1e-4, .1
    optimum = RebalanceOptimizer(RebalanceConfig(risk_aversion=risk, costs_bps=costs_bps,
        position_value_cap=.8, cash_floor=cash_floor, shrinkage=False)).solve([mu], history, [current])
    upper = min(.8, (1 + rate * current - cash_floor) / (1 + rate))
    # Enumerate each affine cost region's stationary point plus all boundaries.
    candidates = [0., current, upper]
    candidates += [np.clip((mu + rate) / (2 * risk * variance), 0, current),
                   np.clip((mu - rate) / (2 * risk * variance), current, upper)]
    utility = lambda h: mu * h - risk * variance * h * h - rate * abs(h - current)  # noqa: E731
    expected = max(candidates, key=utility)
    assert optimum.holding_values_fraction[0] == pytest.approx(expected, abs=2e-6)
    assert optimum.objective_value == pytest.approx(utility(expected), abs=1e-10)


def test_permissions_caps_drift_and_cash_floor_are_planned_together():
    optimizer = RebalanceOptimizer(RebalanceConfig(costs_bps=100, risk_aversion=0,
                                                   position_value_cap=.2, cash_floor=.1))
    current = np.array([.5, 0.])
    plan = optimizer.solve([-.01, .05], np.zeros((5, 2)), current,
                            sell_allowed=np.array([False, True]), max_trade_weights=[.1, .1])
    assert plan.holding_values_fraction[0] == .5  # Over-cap locked holding is retained, not increased.
    assert plan.holding_values_fraction[1] == pytest.approx(.1, abs=1e-7)
    assert plan.cash_fraction >= .1 - 1e-12
    assert plan.fees_fraction == pytest.approx(.001, abs=1e-8)
    with pytest.raises(ValueError, match='infeasible cash floor'):
        RebalanceOptimizer(RebalanceConfig(cash_floor=.9)).solve([1.], np.zeros((5, 1)), [.8],
                                                                  sell_allowed=np.array([False]))
    flat = RebalanceOptimizer(profile()).solve([.1], np.zeros((5, 1)), [0.], buy_allowed=np.array([False]))
    np.testing.assert_array_equal(flat.holding_values_fraction, 0)


def test_cash_floor_funded_after_fees_and_no_forced_full_investment():
    optimizer = RebalanceOptimizer(RebalanceConfig(costs_bps=100, risk_aversion=0,
                                                   position_value_cap=1., cash_floor=.2))
    plan = optimizer.solve([.1], np.zeros((5, 1)), [0.])
    assert plan.holding_values_fraction[0] == pytest.approx(.8 / 1.01, abs=1e-7)
    assert plan.cash_fraction == pytest.approx(.2, abs=1e-7)
    assert plan.target_weights[0] < .8


def test_dpp_cache_and_unavailable_solver_fallback():
    optimizer = RebalanceOptimizer(RebalanceConfig(solver='NOT_INSTALLED', position_value_cap=1))
    one = optimizer.solve([.01, .02], np.zeros((5, 2)), [0, 0])
    optimizer.solve([.02, .01], np.zeros((5, 2)), [.2, .3])
    assert one.solver == 'SCS'
    assert optimizer._problems[2][-1].is_dpp() and len(optimizer._problems) == 1


def test_closed_loop_uses_actual_marked_positions_and_strictly_prior_covariance(monkeypatch):
    forecasts = np.tile([.01, .02], (8, 1)); returns = np.zeros_like(forecasts)
    returns[3] = [.1, -.2]
    captures = []
    original = RebalanceOptimizer.solve
    def spy(self, mu, history, current, **kwargs):
        captures.append((history.copy(), current.copy()))
        return original(self, mu, history, current, **kwargs)
    monkeypatch.setattr(RebalanceOptimizer, 'solve', spy)
    result = run_rebalance_backtest(forecasts, returns, cfg=profile(), lookback=2, start_index=2,
                                    max_trade_values=np.full_like(forecasts, .1))
    for t, (history, current) in enumerate(captures):
        np.testing.assert_array_equal(history, returns[t:t + 2])
        np.testing.assert_array_equal(current, result.pretrade_weights[t])
    assert result.weights[0].sum() < .3  # Capacity-limited entry, not an assumed full target.
    assert result.gross_returns[1] == pytest.approx(np.dot(result.weights[0], returns[3]))
    assert result.metrics['failed_hold_dates'] == 0
    assert result.metrics['max_plan_fill_value_error'] < 1e-12
    assert result.metrics['max_plan_fee_error'] < 1e-12


def test_future_changes_cannot_rewrite_past_decisions_or_execution():
    forecasts = np.tile([.002, .003], (14, 1)); returns = np.zeros_like(forecasts)
    buy, sell = np.ones_like(forecasts, dtype=bool), np.ones_like(forecasts, dtype=bool)
    caps = np.full_like(forecasts, .1)
    original = run_rebalance_backtest(forecasts, returns, cfg=profile(), lookback=3, start_index=3,
                                      buy_allowed=buy, sell_allowed=sell, max_trade_values=caps)
    forecasts[9:] *= -2; returns[9:] = [-.2, .1]; buy[9:] = False; sell[9:] = False; caps[9:] = 0
    changed = run_rebalance_backtest(forecasts, returns, cfg=profile(), lookback=3, start_index=3,
                                     buy_allowed=buy, sell_allowed=sell, max_trade_values=caps)
    for attr in ('weights', 'cumulative_equity', 'trade_values', 'fees', 'pretrade_weights', 'planned_holding_values'):
        np.testing.assert_array_equal(getattr(original, attr)[:6], getattr(changed, attr)[:6])
    assert original.decision_records[:6] == changed.decision_records[:6]


def test_warmup_failed_decisions_and_exact_holds_are_recorded(monkeypatch):
    forecasts = np.full((6, 1), .01); returns = np.zeros_like(forecasts)
    def fail(*args, **kwargs):
        raise cp.error.SolverError('recorded test solver failure')
    monkeypatch.setattr(RebalanceOptimizer, 'solve', fail)
    result = run_rebalance_backtest(forecasts, returns, cfg=profile(), lookback=2)
    assert result.metrics['warmup_hold_dates'] == 2 and result.metrics['failed_hold_dates'] == 4
    assert result.metrics['optimized_dates'] == 0
    assert result.decision_records[2]['reason'] == 'recorded test solver failure'
    np.testing.assert_array_equal(result.cash_equity, 1)
    np.testing.assert_array_equal(result.fees, 0)


def test_actual_cash_conservation_units_and_inputs_preserved():
    rng = np.random.default_rng(8)
    forecasts = rng.uniform(-.005, .006, (25, 4)); returns = rng.uniform(-.05, .05, forecasts.shape)
    buy, sell = rng.random(forecasts.shape) > .1, rng.random(forecasts.shape) > .1
    caps = np.full_like(forecasts, .05)
    copies = [value.copy() for value in (forecasts, returns, buy, sell, caps)]
    cfg = RebalanceConfig(position_value_cap=.4)
    base = run_rebalance_backtest(forecasts, returns, cfg=cfg, lookback=5, start_index=5,
                                  buy_allowed=buy, sell_allowed=sell, max_trade_values=caps)
    scaled = run_rebalance_backtest(forecasts, returns, cfg=cfg, lookback=5, start_index=5,
        buy_allowed=buy, sell_allowed=sell, max_trade_values=caps * 1e6, initial_equity=1e6)
    np.testing.assert_allclose(base.cumulative_equity, scaled.cumulative_equity / 1e6, atol=1e-9)
    holdings, cash = np.zeros(4), 1.
    for t in range(20):
        marked = holdings * (1 + returns[t + 5]); trade = base.trade_values[t]
        assert np.all(trade[~buy[t + 5]] <= 0) and np.all(trade[~sell[t + 5]] >= 0)
        assert np.all(np.abs(trade) <= caps[t + 5] + 1e-14)
        assert base.cash_equity[t] == pytest.approx(cash - trade.sum() - base.fees[t], abs=1e-13)
        holdings, cash = marked + trade, base.cash_equity[t]
        assert holdings.min() >= -1e-14 and cash >= 0
        assert holdings.sum() + cash == pytest.approx(base.cumulative_equity[t])
    for value, previous in zip((forecasts, returns, buy, sell, caps), copies):
        np.testing.assert_array_equal(value, previous)


@pytest.mark.parametrize('options', [
    {'costs_bps': -1}, {'costs_bps': 10000}, {'costs_bps': True}, {'risk_aversion': -1},
    {'risk_aversion': float('nan')}, {'position_value_cap': 1.1}, {'cash_floor': -.1},
    {'shrinkage': 1}, {'solver': ''},
])
def test_invalid_profile_rejected(options):
    with pytest.raises(ValueError):
        RebalanceConfig(**options)


@pytest.mark.parametrize('options', [{'lookback': 1}, {'lookback': True}, {'start_index': 4},
    {'start_index': -1}, {'max_trade_values': [[-1]]}, {'buy_allowed': [[1]]}, {'benchmark': [0]}])
def test_invalid_rolling_options_rejected(options):
    with pytest.raises(ValueError):
        run_rebalance_backtest(np.zeros((4, 1)), np.zeros((4, 1)), **options)


def test_cli_reuses_forecasts_with_separate_actual_holdings_reports(tmp_path):
    art = tmp_path / 'art'; art.mkdir()
    close = torch.full((6, 2), 10.)
    torch.save({'dates': np.arange('2026-01-01', '2026-01-07', dtype='datetime64[D]'), 'stocks': ['a', 'b'],
                'open': close, 'high': close, 'low': close, 'close': close, 'vwap': close,
                'volume': torch.ones_like(close), 'mask': torch.ones_like(close, dtype=torch.bool),
                'amount': torch.full_like(close, 100.)}, art / 'panel.pt')
    torch.save({'pred': torch.full_like(close, .01), 'oos_start': 2}, art / 'predictions.pt')
    (art / 'backtest.pkl').write_text('ordinary ledger sentinel')
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text(f'artifacts_dir: {art}\ncov_lookback: 2\ncosts_bps: 10\nrebalance:\n  position_value_cap: 1\n  risk_aversion: 0\n')
    args = ['rebalance', '--config', str(cfg), '--currency', 'CNY', '--data-category', 'synthetic',
            '--data-source', 'constructed forecasts', '--initial-capital', '100',
            '--participation-rate', '.1', '--liquidity-lookback', '2']
    outcome = CliRunner().invoke(cli, args)
    assert outcome.exit_code == 0, outcome.output
    report = json.loads((art / 'rebalance/summary.json').read_text())
    assert report['config']['costs_bps'] == 10 and report['evaluation']['start_index'] == 2
    assert report['metrics']['optimized_dates'] == 4 and report['metrics']['failed_hold_dates'] == 0
    assert report['liquidity_ready_cells'] == 8
    with (art / 'rebalance/backtest.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    assert np.max(saved.trade_values) <= 10 + 1e-12
    assert saved.initial_equity == 100
    assert (art / 'backtest.pkl').read_text() == 'ordinary ledger sentinel'
