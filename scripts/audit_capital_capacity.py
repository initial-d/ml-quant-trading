"""Freeze synthetic targets and validate the capital CLI with Decimal budgets."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

import click
import numpy as np
import torch

from audit_constrained_execution import decimal_constrained_path
from mlquant.backtest import run_capacity_backtest, panel_trade_permissions
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel


@click.command()
@click.option('--work-dir', required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option('--json-out', required=True, type=click.Path(path_type=Path))
def main(work_dir, json_out):
    seed, start, lookback, participation = 20261122, 300, 20, .01
    panel = make_synthetic_panel(SyntheticConfig(n_dates=500, n_stocks=200, seed=seed))
    rng = np.random.default_rng(seed)
    targets = np.zeros((500, 200))
    # A frozen rotating sparse schedule, with no alpha signal or trained model.
    for t in range(0, 500, 7):
        targets[t:t + 7, rng.choice(200, size=20, replace=False)] = .9 / 20
    inputs = work_dir / 'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    fields = {name: getattr(panel, name) for name in ('open', 'high', 'low', 'close', 'vwap', 'volume',
                                                     'mask', 'amount', 'last_close', 'limit_up', 'limit_down')}
    torch.save(dict(dates=panel.dates, stocks=panel.stocks, **fields), inputs / 'panel.pt')
    torch.save({'weights': torch.from_numpy(targets)}, inputs / 'weights.pt')
    torch.save({'oos_start': start}, inputs / 'predictions.pt')
    config = work_dir / 'config.yaml'
    config.write_text(f'artifacts_dir: {inputs}\ncosts_bps: 20\n')
    output = work_dir / 'capacity'
    command = [sys.executable, '-m', 'mlquant.cli.main', 'capacity', '--config', str(config),
               '--capital', '1000000', '--capital', '100000000', '--capital', '10000000000',
               '--participation-rate', str(participation), '--lookback', str(lookback),
               '--currency', 'CNY', '--data-category', 'synthetic',
               '--data-source', 'SyntheticConfig seed 20261122; frozen rotating 20-name target schedule',
               '--output-dir', str(output)]
    subprocess.run(command, check=True)
    with (output / 'capacity.pkl').open('rb') as handle:
        result = pickle.load(handle)
    report = json.loads((output / 'summary.json').read_text())
    turnover = panel.amount.numpy().astype(np.float64)
    # Independently reconstruct EVERY lagged monetary cap with scalar Decimal sums.
    with localcontext() as context:
        context.prec = 60
        d_values = [[Decimal(str(value)) for value in row] for row in turnover]
        d_rate = Decimal(str(participation))
        cap_reference = np.asarray([[float(d_rate * sum(d_values[p][i] for p in range(t - lookback, t)) / lookback)
                                     for i in range(200)] for t in range(start, 500)])
    np.testing.assert_allclose(result.trade_caps, cap_reference, rtol=1e-14, atol=1e-8)
    buy, sell, _ = panel_trade_permissions(panel)
    validation = {'status': 'passed', 'audit_command': shlex.join([sys.executable, *sys.argv]),
                  'cli_command': shlex.join(command), 'seed': seed,
                  'lagged_cap_max_absolute_error': float(np.abs(result.trade_caps - cap_reference).max()),
                  'oracle_precision_digits': 60, 'cases': []}
    for capital, scenario in zip(result.capitals, result.scenarios):
        expected = decimal_constrained_path(scenario.target_weights, panel.returns.numpy()[start:],
                                            buy[start:], sell[start:], cap_reference, 20, initial_equity=capital)
        errors = {}
        for key, attr in (('equity', 'cumulative_equity'), ('fees', 'fees'), ('cash', 'cash_equity'),
                          ('trades', 'trade_values'), ('requested', 'requested_trade_values')):
            reference = np.asarray([row[key] for row in expected])
            actual = getattr(scenario, attr)
            np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=1e-6)
            errors[key] = float(np.abs(actual - reference).max())
        assert np.all(scenario.cash_equity >= 0) and np.all(scenario.weights >= 0)
        assert np.all(np.abs(scenario.trade_values) <= result.trade_caps + 1e-6)
        validation['cases'].append({'initial_capital': float(capital), 'maximum_absolute_currency_errors': errors,
                                    'max_equity_error_fraction_initial': errors['equity'] / capital,
                                    'cash_conservation_max_currency_residual': scenario.metrics['max_self_financing_residual'],
                                    'daily': {'equity_multiple': (scenario.cumulative_equity / capital).tolist(),
                                              'cash_weight': (scenario.cash_equity / scenario.cumulative_equity).tolist(),
                                              'target_weight_l1': np.abs(scenario.weights - scenario.target_weights).sum(axis=1).tolist()}})
    # Rewriting today's turnover cannot alter its own order; it affects later budgets.
    changed_turnover = turnover.copy(); changed_turnover[400:] *= 1000
    changed = run_capacity_backtest(targets, panel.returns.numpy(), changed_turnover, result.capitals,
                                    start_index=start, lookback=lookback, participation_rate=participation,
                                    buy_allowed=buy, sell_allowed=sell, costs_bps=20)
    for original, modified in zip(result.scenarios, changed.scenarios):
        np.testing.assert_array_equal(original.trade_values[:101], modified.trade_values[:101])
        np.testing.assert_array_equal(original.cumulative_equity[:101], modified.cumulative_equity[:101])
    validation['current_and_future_turnover_mutation'] = 'passed through mutated row 400 inclusive'
    # Change currency denomination consistently; the normalized economics must match.
    scaled = run_capacity_backtest(targets, panel.returns.numpy(), turnover * .001, result.capitals * .001,
                                   start_index=start, lookback=lookback, participation_rate=participation,
                                   buy_allowed=buy, sell_allowed=sell, costs_bps=20)
    for original, converted in zip(result.scenarios, scaled.scenarios):
        np.testing.assert_allclose(original.weights, converted.weights, atol=1e-13)
        np.testing.assert_allclose(original.cumulative_equity * .001, converted.cumulative_equity, rtol=1e-12)
    validation['monetary_unit_conversion'] = 'passed'
    report['validation'] = validation
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    click.echo(f'Passed 500x200 frozen CLI sweep, Decimal caps/ledgers and mutation/scaling checks; wrote {json_out}')


if __name__ == '__main__':
    main()
