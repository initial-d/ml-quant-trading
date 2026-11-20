"""Synthetic self-financing audit using an independent Decimal breakpoint solve."""
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import click
import numpy as np
import torch

from mlquant.backtest import run_backtest, run_self_financing_backtest
from mlquant.backtest import self_financing


def decimal_path(weights, returns, costs_bps):
    """Walk exact buy/sell breakpoints, independent of the FP64 bisection."""
    with localcontext() as ctx:
        ctx.prec = 50
        D = lambda value: Decimal(str(float(value)))  # noqa: E731
        c = D(costs_bps) / Decimal(10000)
        holdings, cash = [Decimal(0)] * weights.shape[1], Decimal(1)
        records = []
        for target_row, return_row in zip(weights, returns):
            w = [D(value) for value in target_row]
            marked = [h * (1 + D(r)) for h, r in zip(holdings, return_row)]
            before = sum(marked) + cash
            # Initially all trades are sales. Crossing H_i/w_i changes one
            # term's sign; the affine equation is solved in each segment.
            slope, intercept = 1 - c * sum(w), c * sum(marked)
            after = (before - intercept) / slope
            boundaries = sorted((h / target, i) for i, (target, h) in enumerate(zip(w, marked)) if target > 0)
            for boundary, i in boundaries:
                if after <= boundary:
                    break
                slope += 2 * c * w[i]
                intercept -= 2 * c * marked[i]
                after = (before - intercept) / slope
            trade = [target * after - old for target, old in zip(w, marked)]
            fee = c * sum(abs(value) for value in trade)
            if abs(before - after - fee) > Decimal('1e-40'):
                raise ValueError("independent breakpoint solution did not balance")
            holdings, cash = [target * after for target in w], after * (1 - sum(w))
            records.append((float(after), float(fee), float(cash), [float(value) for value in trade]))
        return records


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


@click.command()
@click.option('--json-out', required=True, type=click.Path(path_type=Path))
def main(json_out):
    seed = 20261120
    rng = np.random.default_rng(seed)
    symmetric = np.tile([.02, -.02], (252, 1))
    symmetric[::2] *= -1; symmetric[0] = 0
    mixed_weights = rng.dirichlet(np.ones(5), size=75)[:, :4] * .95
    mixed_weights[[0, 10, 22]] = [[1, 0, 0, 0], [0, 0, 0, 0], [.5, .5, 0, 0]]
    cases = [('fixed_50_50', np.tile([.5, .5], (252, 1)), symmetric, 20.0),
             ('changing_targets_and_cash', mixed_weights, rng.uniform(-.05, .05, (75, 4)), 100.0),
             ('synthetic_500x200', np.full((500, 200), .9 / 200), rng.normal(0, .015, (500, 200)), 5.0)]
    results = []
    for name, weights, returns, fee in cases:
        actual = run_self_financing_backtest(weights, returns, costs_bps=fee)
        approximate = run_backtest(weights, returns, costs_bps=fee)
        expected = decimal_path(weights, returns, fee)
        errors = {}
        for key, values in (('equity', actual.cumulative_equity), ('fees', actual.fees),
                            ('cash', actual.cash_equity), ('trades', actual.trade_values)):
            index = {'equity': 0, 'fees': 1, 'cash': 2, 'trades': 3}[key]
            reference = np.asarray([row[index] for row in expected])
            np.testing.assert_allclose(values, reference, rtol=2e-11, atol=1e-13)
            errors[key] = float(np.max(np.abs(values - reference)))
        # Mutate future targets/returns and verify every retained prefix ledger.
        split = len(weights) // 2
        changed_w, changed_r = weights.copy(), returns.copy()
        changed_w[split:] = 0
        changed_r[split:] *= -1
        changed = run_self_financing_backtest(changed_w, changed_r, costs_bps=fee)
        for key in ('cumulative_equity', 'trade_values', 'fees', 'cash_equity', 'pretrade_weights'):
            np.testing.assert_array_equal(getattr(actual, key)[:split], getattr(changed, key)[:split])
        results.append({'case': name, 'shape': list(weights.shape), 'one_way_costs_bps': fee,
                        'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
                        'returns_sha256': hashlib.sha256(returns.tobytes()).hexdigest(),
                        'trade_values_sha256': hashlib.sha256(actual.trade_values.tobytes()).hexdigest(),
                        'maximum_absolute_errors': errors, 'future_mutation_check': 'passed',
                        'self_financing_metrics': actual.metrics, 'target_difference_metrics': approximate.metrics,
                        'self_financing_final_equity': float(actual.cumulative_equity[-1]),
                        'target_difference_final_equity': float(approximate.cumulative_equity[-1]),
                        'daily': {'equity': actual.cumulative_equity.tolist(), 'fees': actual.fees.tolist(),
                                  'cash': actual.cash_equity.tolist(), 'turnover_l1': actual.turnover_l1.tolist(),
                                  'target_difference_equity': approximate.cumulative_equity.tolist()}})
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    cpu = platform.processor() or platform.machine()
    if sys.platform == 'darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    result = {'status': 'passed', 'category': 'synthetic historical-accounting engineering diagnostic',
              'commit': commit, 'command': shlex.join([sys.executable, *sys.argv]),
              'python': sys.version, 'numpy': np.__version__, 'pytorch': torch.__version__,
              'os': platform.platform(), 'cpu': cpu, 'device': 'cpu', 'gpu': 'unused', 'seed': seed,
              'implementation_sha256': hashlib.sha256(Path(self_financing.__file__).read_bytes()).hexdigest(),
              'data_source': 'Constructed target/return arrays and seeded NumPy synthetic returns; no market downloads',
              'cost_assumptions': 'Specified flat one-way fees on actual notional; zero cash interest; initial NAV 1',
              'cases': results,
              'limitations': ['Engineering accounting checks, not calibrated execution costs or market alpha.',
                              'Complete fills at each mark, no settlement, tradability, impact or volume model.',
                              'Comparison labels retain different accounting conventions; no new return claim.']}
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')
    click.echo(f"Passed 3 Decimal ledger cases and future-mutation checks; wrote {json_out}")


if __name__ == '__main__':
    main()
