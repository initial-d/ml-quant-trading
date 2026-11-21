"""Historical execution controls against hand calculations and Decimal ledgers."""
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from mlquant.backtest import run_constrained_backtest, run_self_financing_backtest, panel_trade_permissions
from mlquant.cli.main import cli
from mlquant.data.panel import Panel
from scripts.audit_constrained_execution import decimal_constrained_path


def test_locked_sale_cannot_finance_new_position_and_retains_next_return():
    w = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    r = np.array([[0., 0.], [0., 0.], [-.3, .2], [0., 0.]])
    allowed = np.ones_like(w, dtype=bool); allowed[1:3, 0] = False
    result = run_constrained_backtest(w, r, buy_allowed=allowed, sell_allowed=allowed, costs_bps=0)
    np.testing.assert_array_equal(result.trade_values[1:3], 0)
    np.testing.assert_array_equal(result.cash_equity, 0)
    np.testing.assert_array_equal(result.weights[:3], [[1., 0.]] * 3)
    np.testing.assert_allclose(result.cumulative_equity, [1., 1., .7, .7])
    np.testing.assert_allclose(result.trade_values[3], [-.7, .7])
    np.testing.assert_allclose(result.permission_unfilled[1], [-1., 0.])
    np.testing.assert_allclose(result.cash_unfilled[1], [0., 1.])
    assert result.buy_fill_scale[1] == 0
    full = run_self_financing_backtest(w, r, costs_bps=0)
    assert full.cumulative_equity[-1] == pytest.approx(1.2)


def test_blocked_buy_does_not_block_sale_or_charge_unfilled_order_fees():
    allowed = np.array([[True, True], [True, False]])
    result = run_constrained_backtest([[1., 0.], [0., 1.]], np.zeros((2, 2)),
                                      buy_allowed=allowed, costs_bps=100)
    entry = 1 / 1.01
    assert result.trade_values[1, 0] == pytest.approx(-entry)
    assert result.trade_values[1, 1] == 0
    assert result.fees[1] == pytest.approx(entry * .01)
    assert result.cash_equity[1] == pytest.approx(entry * .99)
    assert result.cumulative_equity[1] == result.cash_equity[1]
    assert result.unfilled_trade_values[1, 1] > 0


def test_locked_position_leaves_only_real_cash_for_pro_rata_buys_with_fees():
    w = np.array([[.6, 0., 0.], [0., .2, .4]])
    allowed = np.ones_like(w, dtype=bool); allowed[1, 0] = False
    result = run_constrained_backtest(w, np.zeros_like(w), sell_allowed=allowed, costs_bps=100)
    np.testing.assert_allclose(result.weights[1, 0] * result.cumulative_equity[1], result.trade_values[0, 0])
    assert result.trade_values[1, 0] == 0
    assert result.trade_values[1, 2] == pytest.approx(2 * result.trade_values[1, 1])
    assert result.trade_values[1].sum() * 1.01 == pytest.approx(result.cash_equity[0])
    assert result.cash_equity[1] == pytest.approx(0, abs=1e-15)
    assert 0 < result.buy_fill_scale[1] < 1


def test_notional_caps_partial_fills_and_daily_recomputation():
    result = run_constrained_backtest([[1.], [1.], [0.]], np.zeros((3, 1)),
                                      max_trade_values=np.full((3, 1), .25), costs_bps=0)
    np.testing.assert_allclose(result.trade_values[:, 0], [.25, .25, -.25])
    np.testing.assert_allclose(result.unfilled_trade_values[:, 0], [.75, .5, -.25])
    np.testing.assert_allclose(result.cash_equity, [.75, .5, .75])
    np.testing.assert_allclose(result.weights[:, 0], [.25, .5, .25])
    assert result.metrics["notional_fill_ratio"] == pytest.approx(.75 / 2.25)
    zero = run_constrained_backtest([[1.]], [[0.]], max_trade_values=[[0.]], costs_bps=9999)
    assert zero.cash_equity[0] == 1 and zero.fees[0] == 0
    assert zero.metrics["notional_fill_ratio"] == 0


@pytest.mark.parametrize("costs_bps", [0, 5, 100, 3500, 9999])
def test_independent_decimal_oracle_and_currency_ledger(costs_bps):
    rng = np.random.default_rng(51)
    w = rng.dirichlet(np.ones(5), size=45)[:, :4] * .9
    w[[0, 12, 30]] = [[1, 0, 0, 0], [0, 0, 0, 0], [.5, .5, 0, 0]]
    r = rng.uniform(-.2, .2, w.shape)
    buy, sell = rng.random(w.shape) > .25, rng.random(w.shape) > .2
    cap = rng.uniform(.02, .25, w.shape)
    result = run_constrained_backtest(w, r, buy_allowed=buy, sell_allowed=sell,
                                      max_trade_values=cap, costs_bps=costs_bps)
    reference = decimal_constrained_path(w, r, buy, sell, cap, costs_bps)
    for label, attr in (("equity", "cumulative_equity"), ("trades", "trade_values"),
                        ("cash", "cash_equity"), ("fees", "fees"), ("requested", "requested_trade_values")):
        np.testing.assert_allclose(getattr(result, attr), [row[label] for row in reference], atol=1e-13, rtol=2e-10)
    holdings, cash = np.zeros(4), 1.
    for t in range(len(w)):
        marked = holdings * (1 + r[t])
        trades = result.trade_values[t]
        assert np.all(trades[~buy[t]] <= 0) and np.all(trades[~sell[t]] >= 0)
        assert np.all(np.abs(trades) <= cap[t] + 1e-15)
        assert result.fees[t] == pytest.approx(costs_bps * 1e-4 * np.abs(trades).sum())
        assert result.cash_equity[t] == pytest.approx(cash - trades.sum() - result.fees[t], abs=1e-13)
        holdings, cash = marked + trades, result.cash_equity[t]
        # Reconstructing fills can leave an ULP at a fully liquidated asset.
        assert holdings.min() >= -1e-15 and cash >= 0
        assert result.weights[t].min() >= 0
        assert holdings.sum() + cash == pytest.approx(result.cumulative_equity[t])
    np.testing.assert_allclose(result.unfilled_trade_values,
                               result.permission_unfilled + result.cap_unfilled + result.cash_unfilled, atol=1e-13)
    assert result.metrics["max_self_financing_residual"] < 1e-13


def test_unrestricted_arrays_exactly_match_existing_self_financing():
    rng = np.random.default_rng(97)
    w = rng.dirichlet(np.ones(6), size=30)[:, :5]
    r = rng.uniform(-.1, .1, w.shape)
    full = run_self_financing_backtest(w, r, costs_bps=100)
    released = run_constrained_backtest(w, r, costs_bps=100)
    for attr, value in vars(full).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(released, attr))
    for key, value in full.metrics.items():
        assert released.metrics[key] == value
    assert released.metrics["unfilled_notional_total"] == 0


def test_scale_permutation_and_input_preservation():
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.ones(4), size=25)[:, :3]
    r = rng.uniform(-.1, .1, w.shape)
    buy, sell = rng.random(w.shape) > .2, rng.random(w.shape) > .2
    cap = rng.uniform(.01, .2, w.shape)
    copies = [value.copy() for value in (w, r, buy, sell, cap)]
    base = run_constrained_backtest(w, r, buy_allowed=buy, sell_allowed=sell, max_trade_values=cap)
    large = run_constrained_backtest(w, r, buy_allowed=buy, sell_allowed=sell,
                                    max_trade_values=cap * 1e6, initial_equity=1e6)
    np.testing.assert_allclose(large.cumulative_equity / 1e6, base.cumulative_equity)
    np.testing.assert_allclose(large.trade_values / 1e6, base.trade_values, atol=1e-15)
    permuted = run_constrained_backtest(w[:, ::-1], r[:, ::-1], buy_allowed=buy[:, ::-1],
                                       sell_allowed=sell[:, ::-1], max_trade_values=cap[:, ::-1])
    np.testing.assert_allclose(permuted.trade_values[:, ::-1], base.trade_values, atol=1e-15)
    for value, previous in zip((w, r, buy, sell, cap), copies):
        np.testing.assert_array_equal(value, previous)


def test_future_constraints_returns_and_targets_cannot_rewrite_history():
    w = np.tile([.4, .5], (20, 1)); r = np.full_like(w, .01)
    buy = np.ones_like(w, dtype=bool); sell = buy.copy(); cap = np.full_like(w, .1)
    original = run_constrained_backtest(w, r, buy_allowed=buy, sell_allowed=sell, max_trade_values=cap)
    w[10:] = 0; r[10:] = -.5; buy[10:] = False; sell[10:] = False; cap[10:] = 0
    changed = run_constrained_backtest(w, r, buy_allowed=buy, sell_allowed=sell, max_trade_values=cap)
    for attr, value in vars(original).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value[:10], getattr(changed, attr)[:10])


def test_zero_nav_is_absorbing_and_unheld_zero_target_is_valid():
    result = run_constrained_backtest([[1.], [0.], [0.]], [[0.], [-1.], [.5]], costs_bps=0)
    assert result.bankrupt_index == 1
    np.testing.assert_array_equal(result.trade_values[1:], 0)
    np.testing.assert_array_equal(result.cumulative_equity[1:], 0)
    flat = run_constrained_backtest(np.zeros((3, 1)), np.zeros((3, 1)))
    assert flat.metrics["notional_fill_ratio"] == 1


@pytest.mark.parametrize("kwargs", [
    {"buy_allowed": [[1]]}, {"buy_allowed": [[float('nan')]]}, {"sell_allowed": [True]},
    {"buy_allowed": [[True, False]]}, {"max_trade_values": -1},
    {"max_trade_values": [[-1.]]}, {"max_trade_values": [[float('inf')]]},
    {"max_trade_values": [[float('nan')]]}, {"max_trade_values": [[1., 1.]]},
    {"costs_bps": 10000}, {"initial_equity": 0},
])
def test_invalid_permissions_cap_and_shared_options_rejected(kwargs):
    with pytest.raises(ValueError):
        run_constrained_backtest([[.5]], [[0.]], **kwargs)


def panel_for_gates():
    close = torch.tensor([[10., 10.], [11., 9.], [0., 10.], [10., 10.]])
    mask = torch.ones_like(close, dtype=torch.bool); mask[2, 0] = False
    ones = torch.ones_like(close)
    return Panel(dates=np.arange('2026-01-01', '2026-01-05', dtype='datetime64[D]'),
                 stocks=np.array(['a', 'b']), open=close, high=close, low=close, close=close,
                 vwap=close, volume=ones, mask=mask,
                 limit_up=torch.full_like(close, 11), limit_down=torch.full_like(close, 9))


def test_panel_masks_are_directional_and_do_not_use_future_availability():
    panel = panel_for_gates()
    buy, sell, meta = panel_trade_permissions(panel)
    assert not buy[1, 0] and sell[1, 0]  # Upper band allows selling, not buying.
    assert buy[1, 1] and not sell[1, 1]  # Lower band allows buying, not selling.
    assert not buy[2, 0] and not sell[2, 0]
    no_bands = replace(panel, limit_up=None, limit_down=None)
    b, s, metadata = panel_trade_permissions(no_bands)
    assert b[1].all() and s[1].all() and not metadata['limit_up_supplied']
    assert meta['limit_up_supplied']
    changed_mask = panel.mask.clone(); changed_mask[2:] = False
    b2, s2, _ = panel_trade_permissions(replace(panel, mask=changed_mask))
    np.testing.assert_array_equal(buy[:2], b2[:2]); np.testing.assert_array_equal(sell[:2], s2[:2])


def test_missing_volume_and_invalid_supplied_bands_block_relevant_directions():
    panel = panel_for_gates()
    volume = panel.volume.clone(); volume[0] = torch.tensor([0., float('nan')])
    up = panel.limit_up.clone(); up[3] = torch.tensor([0., float('nan')])
    b, s, _ = panel_trade_permissions(replace(panel, volume=volume, limit_up=up))
    assert not b[0].any() and not s[0].any()
    assert not b[3].any() and s[3].all()


@pytest.mark.parametrize("use_config", [False, True])
def test_cli_halt_retention_resumption_oos_and_saved_ledger(tmp_path, use_config):
    art = tmp_path / 'art'; art.mkdir()
    close = torch.tensor([[10., 10.], [10., 10.], [0., 10.], [7., 12.]])
    mask = torch.ones_like(close, dtype=torch.bool); mask[2, 0] = False
    torch.save({'dates': np.arange('2026-01-01', '2026-01-05', dtype='datetime64[D]'),
                'stocks': ['a', 'b'], 'open': close, 'high': close, 'low': close, 'close': close,
                'vwap': close, 'volume': torch.ones_like(close), 'mask': mask}, art / 'panel.pt')
    torch.save({'weights': torch.tensor([[0., 0.], [1., 0.], [0., 1.], [0., 1.]])}, art / 'weights.pt')
    torch.save({'oos_start': 1}, art / 'predictions.pt')
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(f'artifacts_dir: {art}\ncosts_bps: 0\n' +
                   ('backtest_accounting: constrained\n' if use_config else ''))
    args = ['backtest', '--config', str(cfg)] + ([] if use_config else ['--accounting', 'constrained'])
    outcome = CliRunner().invoke(cli, args)
    assert outcome.exit_code == 0, outcome.output
    report = json.loads((art / 'summary.json').read_text())
    assert report['accounting']['mode'] == 'constrained'
    assert report['accounting']['execution']['limit_up_supplied'] is False
    assert report['evaluation']['n_oos_dates'] == report['metrics']['n_periods'] == 3
    assert 'constrained' in (art / 'summary.md').read_text()
    with (art / 'backtest.pkl').open('rb') as handle:
        result = pickle.load(handle)
    assert result.trade_values[1, 0] == 0 and result.weights[1, 0] == 1
    assert result.unfilled_trade_values[1, 0] == -1
    assert result.cumulative_equity[-1] == pytest.approx(.7)
    assert result.buy_allowed.shape == (3, 2)
