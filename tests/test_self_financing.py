"""Independent Decimal trade partitions, cash ledgers and timing controls."""
from decimal import Decimal, localcontext
from itertools import product
import json
import pickle

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from mlquant.backtest import run_backtest, run_self_financing_backtest
from mlquant.cli.main import cli


def decimal_oracle(weights, returns, costs_bps):
    """Enumerate every buy/sell partition; no bisection or production helpers."""
    with localcontext() as context:
        context.prec = 60
        D = lambda x: Decimal(str(float(x)))  # noqa: E731
        fee_rate = D(costs_bps) / Decimal(10000)
        positions = [Decimal(0)] * weights.shape[1]
        cash = Decimal(1)
        paths = []
        for w, r in zip(weights, returns):
            target = [D(x) for x in w]
            marked = [value * (1 + D(change)) for value, change in zip(positions, r)]
            before = sum(marked) + cash
            for signs in product((-1, 1), repeat=len(target)):
                denominator = 1 + fee_rate * sum(s * value for s, value in zip(signs, target))
                after = (before + fee_rate * sum(s * value for s, value in zip(signs, marked))) / denominator
                trades = [value * after - old for value, old in zip(target, marked)]
                if 0 <= after <= before and all(s * q >= 0 for s, q in zip(signs, trades)):
                    break
            else:
                raise AssertionError("no feasible independent trade partition")
            fee = fee_rate * sum(abs(q) for q in trades)
            positions = [value * after for value in target]
            cash = after * (1 - sum(target))
            paths.append((float(after), float(fee), float(cash), [float(q) for q in trades]))
        return paths


def test_entry_is_paid_from_cash_and_first_fee_drawdown_is_visible():
    result = run_self_financing_backtest([[1.0]], [[0.0]], costs_bps=100)
    assert result.cumulative_equity[0] == pytest.approx(1 / 1.01)
    assert result.trade_values[0, 0] == pytest.approx(1 / 1.01)
    assert result.fees[0] == pytest.approx(.01 / 1.01)
    assert result.trade_values[0, 0] + result.fees[0] == pytest.approx(1)
    assert result.cash_equity[0] == pytest.approx(0)
    assert result.metrics["max_dd"] == pytest.approx(.01 / 1.01)


def test_constant_targets_still_trade_after_relative_price_changes():
    weights = np.tile([.5, .5], (2, 1))
    returns = np.array([[0., 0.], [.1, -.1]])
    result = run_self_financing_backtest(weights, returns, costs_bps=200)
    approximate = run_backtest(weights, returns, costs_bps=200)
    np.testing.assert_allclose(result.pretrade_weights[1], [.55, .45])
    assert result.turnover_l1[1] == pytest.approx(.1)
    assert result.metrics["turnover"] == pytest.approx(.05)
    assert result.fees[1] == pytest.approx(.002 / 1.02)
    assert result.portfolio_returns[1] == pytest.approx(-.002)
    assert result.cumulative_equity[-1] < approximate.cumulative_equity[-1]
    assert approximate.metrics["turnover"] == 0
    assert result.trade_values[1].sum() + result.fees[1] == pytest.approx(0, abs=1e-15)


def test_partial_investment_preserves_cash_and_sells_drifted_gain():
    result = run_self_financing_backtest([[.5], [.5]], [[0.], [.1]], costs_bps=100)
    start = 1 / 1.005
    second_fee = start * .025 * .01 / (1 - .005)
    assert result.cumulative_equity[1] == pytest.approx(start * 1.05 - second_fee)
    assert result.fees[1] == pytest.approx(second_fee)
    assert result.trade_values[1, 0] < 0
    assert result.cash_equity[1] == pytest.approx(result.cumulative_equity[1] * .5)
    assert result.portfolio_returns[1] == pytest.approx(.05 - second_fee / start)


@pytest.mark.parametrize("costs_bps", [0, 5, 100, 3500, 9999])
def test_path_matches_independent_decimal_partition_solution(costs_bps):
    rng = np.random.default_rng(20261120)
    weights = rng.dirichlet(np.ones(5), size=25)[:, :4] * .9
    weights[[0, 7, 13]] = [[1, 0, 0, 0], [0, 0, 0, 0], [.5, .5, 0, 0]]
    returns = rng.uniform(-.3, .4, weights.shape)
    returns[0] = 0
    expected = decimal_oracle(weights, returns, costs_bps)
    result = run_self_financing_backtest(weights, returns, costs_bps=costs_bps)
    np.testing.assert_allclose(result.cumulative_equity, [row[0] for row in expected], rtol=2e-11, atol=1e-14)
    np.testing.assert_allclose(result.fees, [row[1] for row in expected], rtol=2e-11, atol=1e-14)
    np.testing.assert_allclose(result.cash_equity, [row[2] for row in expected], rtol=2e-11, atol=1e-14)
    np.testing.assert_allclose(result.trade_values, [row[3] for row in expected], rtol=2e-11, atol=1e-14)


def test_each_row_balances_cash_holdings_fees_and_net_returns():
    rng = np.random.default_rng(29)
    weights = rng.dirichlet(np.ones(4), size=40)[:, :3]
    returns = rng.uniform(-.2, .3, weights.shape)
    result = run_self_financing_backtest(weights, returns, costs_bps=70)
    positions, cash, previous = np.zeros(3), 1.0, 1.0
    for t in range(40):
        marked = positions * (1 + returns[t])
        before = marked.sum() + cash
        after = result.cumulative_equity[t]
        np.testing.assert_allclose(marked + result.trade_values[t], result.weights[t] * after, atol=1e-14)
        assert result.cash_equity[t] == pytest.approx(cash - result.trade_values[t].sum() - result.fees[t])
        assert before == pytest.approx(after + result.fees[t])
        assert result.fees[t] == pytest.approx(.007 * np.abs(result.trade_values[t]).sum())
        assert result.portfolio_returns[t] == pytest.approx(result.gross_returns[t] - result.fees[t] / previous)
        positions, cash, previous = result.weights[t] * after, result.cash_equity[t], after
    np.testing.assert_allclose(np.cumprod(1 + result.portfolio_returns), result.cumulative_equity)
    assert result.metrics["max_self_financing_residual"] < 1e-13


def test_future_changes_do_not_affect_past_ledger_and_current_targets_do_not_earn_current_returns():
    weights = np.tile([.4, .5], (20, 1))
    returns = np.random.default_rng(43).uniform(-.1, .1, weights.shape)
    original = run_self_financing_backtest(weights, returns)
    changed_w, changed_r = weights.copy(), returns.copy()
    changed_w[10:] = [.8, .1]
    changed_r[10:] *= -3
    changed = run_self_financing_backtest(changed_w, changed_r)
    for key in ("portfolio_returns", "cumulative_equity", "trade_values", "fees", "pretrade_weights"):
        np.testing.assert_array_equal(getattr(changed, key)[:10], getattr(original, key)[:10])
    current_w = weights.copy(); current_w[10] = [0, 0]
    current = run_self_financing_backtest(current_w, returns)
    assert current.gross_returns[10] == original.gross_returns[10]
    assert current.gross_returns[11] == 0


def test_scale_permutation_input_preservation_and_zero_fee_legacy_agreement():
    rng = np.random.default_rng(18)
    weights = rng.dirichlet(np.ones(4), size=25)[:, :3]
    returns = rng.uniform(-.1, .1, weights.shape)
    w_before, r_before = weights.copy(), returns.copy()
    base = run_self_financing_backtest(weights, returns)
    large = run_self_financing_backtest(weights, returns, initial_equity=1e6)
    permuted = run_self_financing_backtest(weights[:, ::-1], returns[:, ::-1])
    np.testing.assert_allclose(large.cumulative_equity / 1e6, base.cumulative_equity)
    np.testing.assert_allclose(large.fees / 1e6, base.fees)
    np.testing.assert_allclose(permuted.portfolio_returns, base.portfolio_returns, atol=1e-15)
    np.testing.assert_array_equal(weights, w_before); np.testing.assert_array_equal(returns, r_before)
    free = run_self_financing_backtest(weights, returns, costs_bps=0)
    approximate = run_backtest(weights, returns, costs_bps=0)
    np.testing.assert_allclose(free.portfolio_returns, approximate.portfolio_returns, atol=1e-15)


def test_bankruptcy_is_absorbing_and_partial_asset_loss_cannot_be_rebought():
    bankrupt = run_self_financing_backtest(np.ones((4, 1)), [[0], [-1], [.2], [5]], costs_bps=10)
    assert bankrupt.bankrupt_index == 1
    np.testing.assert_array_equal(bankrupt.cumulative_equity[1:], 0)
    np.testing.assert_array_equal(bankrupt.weights[1:], 0)
    np.testing.assert_array_equal(bankrupt.trade_values[1:], 0)
    assert bankrupt.metrics["max_dd"] == 1 and bankrupt.metrics["ann_return"] == -1
    with pytest.raises(ValueError, match="total loss"):
        run_self_financing_backtest([[.5], [.5]], [[0], [-1]])
    partial = run_self_financing_backtest([[.5], [0]], [[0], [-1]], costs_bps=0)
    assert partial.bankrupt_index is None
    assert partial.cumulative_equity[-1] == .5
    assert partial.cash_equity[-1] == .5


def test_float32_budget_rounding_is_explicit_and_larger_overruns_fail():
    weights = np.tile(np.array([.2, .3, .5], dtype=np.float32), (3, 1))
    before = weights.copy()
    result = run_self_financing_backtest(weights, np.zeros_like(weights))
    assert 0 < result.target_rounding_max_excess < 1e-6
    assert result.target_weights.dtype == np.float64
    np.testing.assert_array_equal(before, weights)
    with pytest.raises(ValueError, match="budget"):
        run_self_financing_backtest([[.5, .50001]], [[0, 0]])


@pytest.mark.parametrize("weights,returns", [([], []), ([[1]], [[0, 0]]), ([1], [0]),
                                           ([[-.1]], [[0]]), ([[float('nan')]], [[0]]),
                                           ([[1.1]], [[0]]), ([[.5]], [[float('inf')]]),
                                           ([[.5]], [[-1.1]])])
def test_invalid_market_or_weight_grid_is_rejected(weights, returns):
    with pytest.raises(ValueError):
        run_self_financing_backtest(weights, returns)


@pytest.mark.parametrize("kwargs", [{"costs_bps": -1}, {"costs_bps": 10000}, {"costs_bps": True},
                                    {"costs_bps": float('nan')}, {"initial_equity": 0},
                                    {"initial_equity": True}, {"initial_equity": float('inf')},
                                    {"benchmark": [0, 0]}, {"benchmark": [float('nan')]}])
def test_invalid_accounting_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run_self_financing_backtest([[.5]], [[0]], **kwargs)


@pytest.mark.parametrize("config_mode,override,expected", [("self_financing", None, "self_financing"),
                                                         ("self_financing", "target_difference", "target_difference"),
                                                         (None, "self_financing", "self_financing")])
def test_cli_reuses_existing_artifacts_preserves_oos_and_records_mode(tmp_path, config_mode, override, expected):
    art = tmp_path / "art"; art.mkdir()
    close = torch.tensor([[10., 10.], [10., 10.], [11., 9.], [10., 10.]])
    ones = torch.ones_like(close)
    dates = np.arange('2026-01-01', '2026-01-05', dtype='datetime64[D]')
    torch.save({"dates": dates, "stocks": ["a", "b"], "open": close, "high": close, "low": close,
                "close": close, "vwap": close, "volume": ones, "mask": ones.bool()}, art / "panel.pt")
    torch.save({"weights": torch.full_like(close, .5)}, art / "weights.pt")
    torch.save({"oos_start": 1}, art / "predictions.pt")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"artifacts_dir: {art}\ncosts_bps: 200\n" +
                   (f"backtest_accounting: {config_mode}\n" if config_mode else ""))
    args = ["backtest", "--config", str(cfg)] + (["--accounting", override] if override else [])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    report = json.loads((art / "summary.json").read_text())
    assert report["accounting"]["mode"] == expected
    assert report["evaluation"]["n_oos_dates"] == report["metrics"]["n_periods"] == 3
    assert expected in (art / "summary.md").read_text()
    with (art / "backtest.pkl").open('rb') as handle:
        saved = pickle.load(handle)
    if expected == "self_financing":
        assert saved.fees[1] > 0 and saved.trade_values.shape == (3, 2)
        assert saved.initial_equity == 1
