"""Synthetic fee-aware planning audit with equal-cost fee-blind controls."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shlex
import subprocess
import sys

import click
import cvxpy as cp
import numpy as np
import torch

from audit_constrained_execution import clean, decimal_constrained_path
from mlquant.backtest import run_rebalance_backtest, panel_trade_permissions, lagged_turnover_caps
from mlquant.backtest.constrained import _run_constrained_ledger
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.portfolio import RebalanceConfig, RebalanceOptimizer


def fee_blind_control(forecasts, realized, cfg, start, lookback, buy, sell, caps, capital):
    """Identical optimizer/constraints, zero planning fee, SAME actual execution fee."""
    optimizer = RebalanceOptimizer(replace(cfg, costs_bps=0))
    failures = []
    def decide(t, marked, cash, before):
        row = start + t
        try:
            decision = optimizer.solve(forecasts[row], realized[row - lookback:row], marked / before,
                buy_allowed=buy[row], sell_allowed=sell[row],
                max_trade_weights=None if caps is None else caps[row] / before)
        except (ValueError, cp.error.SolverError) as exc:
            failures.append({'row': row, 'reason': str(exc)})
            return None
        return decision.target_weights
    result = _run_constrained_ledger(np.zeros_like(forecasts[start:]), realized[start:],
        buy_allowed=buy[start:], sell_allowed=sell[start:], max_trade_values=None if caps is None else caps[start:],
        initial_equity=capital, costs_bps=cfg.costs_bps, decision=decide)
    return result, failures


@click.command()
@click.option('--json-out', required=True, type=click.Path(path_type=Path))
@click.option('--work-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
def main(json_out, work_dir):
    seed, start, lookback, capital = 20261123, 100, 20, 1e6
    rng = np.random.default_rng(seed)
    panel = make_synthetic_panel(SyntheticConfig(n_dates=200, n_stocks=30, seed=seed))
    realized = panel.returns.numpy().astype(np.float64)
    buy, sell, permission_policy = panel_trade_permissions(panel)
    strong = rng.normal(.0005, .003, realized.shape)
    weak = np.full_like(strong, -.001)
    for t in range(len(weak)):
        weak[t, t % 2] = .0003
    caps, ready = lagged_turnover_caps(panel.amount.numpy(), participation_rate=.001, lookback=20)
    cfg = RebalanceConfig(costs_bps=5, position_value_cap=.2, cash_floor=.05)
    cases = []
    for name, forecasts, limit in [('weak_edge', weak, None), ('changing_forecasts', strong, None),
                                   ('changing_forecasts_with_liquidity', strong, caps)]:
        actual = run_rebalance_backtest(forecasts, realized, cfg=cfg, lookback=lookback, start_index=start,
            initial_equity=capital, buy_allowed=buy, sell_allowed=sell, max_trade_values=limit)
        blind, blind_failures = fee_blind_control(forecasts, realized, cfg, start, lookback, buy, sell, limit, capital)
        expected = decimal_constrained_path(actual.target_weights, realized[start:], buy[start:], sell[start:],
            None if limit is None else limit[start:], cfg.costs_bps, initial_equity=capital)
        errors = {}
        for key, attr in [('equity', 'cumulative_equity'), ('fees', 'fees'), ('cash', 'cash_equity'), ('trades', 'trade_values')]:
            reference = np.asarray([row[key] for row in expected])
            np.testing.assert_allclose(getattr(actual, attr), reference, rtol=2e-10, atol=1e-7)
            errors[key] = float(np.abs(getattr(actual, attr) - reference).max())
        assert np.all(actual.cash_equity >= 0) and np.all(actual.weights >= 0)
        np.testing.assert_allclose(actual.planned_fees, actual.fees, rtol=2e-9, atol=1e-8)
        split = 160
        changed_f, changed_r, changed_b, changed_s = (value.copy() for value in (forecasts, realized, buy, sell))
        changed_f[split:] *= -1; changed_r[split:] *= -1
        changed_b[split:] = False; changed_s[split:] = False
        changed_caps = None if limit is None else limit.copy()
        if changed_caps is not None:
            changed_caps[split:] = 0
        changed = run_rebalance_backtest(changed_f, changed_r, cfg=cfg, lookback=lookback, start_index=start,
            initial_equity=capital, buy_allowed=changed_b, sell_allowed=changed_s, max_trade_values=changed_caps)
        for attr in ('weights', 'fees', 'trade_values', 'cumulative_equity', 'planned_holding_values'):
            np.testing.assert_array_equal(getattr(actual, attr)[:split-start], getattr(changed, attr)[:split-start])
        record = {'case': name, 'one_way_execution_costs_bps': cfg.costs_bps,
                  'fee_aware_planning_costs_bps': cfg.costs_bps, 'fee_blind_planning_costs_bps': 0,
                  'fee_aware_metrics': actual.metrics, 'fee_blind_metrics': blind.metrics,
                  'fee_aware_final_nav_multiple': float(actual.cumulative_equity[-1] / capital),
                  'fee_blind_final_nav_multiple': float(blind.cumulative_equity[-1] / capital),
                  'fee_aware_mean_cash_weight': float((actual.cash_equity / actual.cumulative_equity).mean()),
                  'fee_blind_mean_cash_weight': float((blind.cash_equity / blind.cumulative_equity).mean()),
                  'decimal_ledger_max_currency_errors': errors, 'future_mutation': 'passed',
                  'fee_aware_decisions': actual.decision_records, 'fee_blind_failed_decisions': blind_failures,
                  'forecasts_sha256': hashlib.sha256(forecasts.tobytes()).hexdigest(),
                  'daily': {'equity_multiple': (actual.cumulative_equity / capital).tolist(),
                            'fees_fraction_initial': (actual.fees / capital).tolist(),
                            'fee_blind_equity_multiple': (blind.cumulative_equity / capital).tolist()}}
        cases.append(record)
    # Independent analytic one-asset optima, spanning both absolute-cost regions.
    history = np.tile([[-.01], [.01]], (20, 1))
    variance, risk, current = float(np.var(history[:, 0], ddof=1)), 10., .4
    analytic_errors = []
    for bps in (0, 10, 100):
        rate = bps * 1e-4
        optimizer = RebalanceOptimizer(RebalanceConfig(costs_bps=bps, risk_aversion=risk,
            position_value_cap=.8, cash_floor=.1, shrinkage=False))
        for mu in (-.004, -.001, 0., .001, .004):
            upper = min(.8, (1 + rate * current - .1) / (1 + rate))
            candidates = [0., current, upper, np.clip((mu + rate) / (2 * risk * variance), 0, current),
                          np.clip((mu - rate) / (2 * risk * variance), current, upper)]
            utility = lambda h, signal=mu, fee=rate: signal*h - risk*variance*h*h - fee*abs(h-current)  # noqa: E731
            expected = max(candidates, key=utility)
            observed = optimizer.solve([mu], history, [current])
            error = abs(observed.holding_values_fraction[0] - expected)
            assert error < 2e-6
            analytic_errors.append({'mu': mu, 'costs_bps': bps, 'weight_error': float(error),
                                     'objective_error': abs(observed.objective_value - utility(expected))})
    # Run the actual CLI on the frozen stronger forecasts with prior-turnover caps.
    inputs = work_dir / 'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    torch.save(dict(dates=panel.dates, stocks=panel.stocks, **{name: getattr(panel, name) for name in
        ('open', 'high', 'low', 'close', 'volume', 'vwap', 'mask', 'amount', 'last_close', 'limit_up', 'limit_down')}), inputs/'panel.pt')
    torch.save({'pred': torch.from_numpy(strong), 'oos_start': start}, inputs/'predictions.pt')
    config = work_dir / 'config.yaml'
    config.write_text(f'artifacts_dir: {inputs}\ncosts_bps: 5\ncov_lookback: 20\nrebalance:\n  position_value_cap: .2\n  cash_floor: .05\n')
    command = [sys.executable, '-m', 'mlquant.cli.main', 'rebalance', '--config', str(config),
        '--initial-capital', str(capital), '--participation-rate', '.001', '--currency', 'CNY',
        '--data-category', 'synthetic', '--data-source', 'SyntheticConfig 20261123; seeded independent forecasts']
    subprocess.run(command, check=True)
    with (inputs/'rebalance/backtest.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    cli_record = json.loads((inputs/'rebalance/summary.json').read_text())
    np.testing.assert_allclose(saved.cumulative_equity[-1]/capital, cases[-1]['fee_aware_final_nav_multiple'], atol=1e-12)
    result = dict(cli_record, audit_command=shlex.join([sys.executable, *sys.argv]),
        cli_command=shlex.join(command), seed=seed, panel_shape=list(realized.shape),
        liquidity_ready_cells=int(ready[start:].sum()), planning_profile=asdict(cfg),
        permission_policy=permission_policy, analytic_checks=analytic_errors, cases=cases,
        control_protocol='Same risk/caps/permissions/cash floor and actual fee; planning fee alone set to zero in fee-blind control.')
    result['limitations'].append('Synthetic independent forecasts contain no intended alpha; counterfactual returns are policy diagnostics, not market evidence.')
    result['environment_cpu'] = platform.processor() or platform.machine()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')
    click.echo(f'Passed 3 equal-execution-cost controls, 15 analytic optima, Decimal ledgers and actual CLI; wrote {json_out}')


if __name__ == '__main__':
    main()
