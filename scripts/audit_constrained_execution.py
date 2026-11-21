"""Synthetic constrained-fill audit against an independent Decimal ledger."""
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

from mlquant.backtest import run_constrained_backtest, run_self_financing_backtest
from mlquant.backtest import constrained, self_financing


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def decimal_constrained_path(weights, returns, buy_allowed, sell_allowed, caps, costs_bps,
                             initial_equity=1):
    """Scalar Decimal breakpoint solve and cash reconstruction, no engine helpers."""
    with localcontext() as context:
        context.prec = 60
        D = lambda value: Decimal(str(float(value)))  # noqa: E731
        rate = D(costs_bps) / Decimal(10000)
        holdings, cash = [Decimal(0)] * weights.shape[1], D(initial_equity)
        records = []
        for t, (w_row, r_row) in enumerate(zip(weights, returns)):
            w = [D(value) for value in w_row]
            marked = [h * (1 + D(r)) for h, r in zip(holdings, r_row)]
            before = sum(marked) + cash
            slope, intercept = 1 - rate * sum(w), rate * sum(marked)
            reference_nav = (before - intercept) / slope
            for boundary, i in sorted((h / target, i) for i, (target, h) in enumerate(zip(w, marked)) if target > 0):
                if reference_nav <= boundary:
                    break
                slope += 2 * rate * w[i]
                intercept -= 2 * rate * marked[i]
                reference_nav = (before - intercept) / slope
            requested = [target * reference_nav - old for target, old in zip(w, marked)]
            limited = []
            for i, value in enumerate(requested):
                allowed = buy_allowed[t, i] if value > 0 else sell_allowed[t, i]
                permitted = value if allowed else Decimal(0)
                if caps is not None:
                    cap = D(caps[t, i])
                    permitted = max(-cap, min(cap, permitted))
                limited.append(permitted)
            sales = [-min(value, Decimal(0)) for value in limited]
            buys = [max(value, Decimal(0)) for value in limited]
            available = cash + sum(sales) * (1 - rate)
            scale = min(Decimal(1), available / ((1 + rate) * sum(buys))) if sum(buys) else Decimal(1)
            trades = [b * scale - s for b, s in zip(buys, sales)]
            fee = rate * sum(abs(value) for value in trades)
            holdings = [old + q for old, q in zip(marked, trades)]
            cash = available - sum(buys) * scale * (1 + rate)
            after = sum(holdings) + cash
            if min(holdings) < -Decimal('1e-45') or cash < -Decimal('1e-45') or abs(before - after - fee) > Decimal('1e-45'):
                raise AssertionError("independent constrained ledger failed conservation")
            # Finite-precision Decimal division can leave ~1e-60 at full investment.
            cash = max(cash, Decimal(0))
            records.append({"equity": float(after), "fees": float(fee), "cash": float(cash),
                            "requested": [float(value) for value in requested],
                            "trades": [float(value) for value in trades], "scale": float(scale)})
        return records


def audit_case(weights, returns, buy, sell, caps, costs_bps):
    actual = run_constrained_backtest(weights, returns, buy_allowed=buy, sell_allowed=sell,
                                     max_trade_values=caps, costs_bps=costs_bps)
    expected = decimal_constrained_path(weights, returns, buy, sell, caps, costs_bps)
    errors = {}
    for label, attr in (("equity", "cumulative_equity"), ("fees", "fees"), ("cash", "cash_equity"),
                        ("trades", "trade_values"), ("requested", "requested_trade_values"),
                        ("scale", "buy_fill_scale")):
        reference = np.asarray([row[label] for row in expected])
        values = getattr(actual, attr)
        np.testing.assert_allclose(values, reference, atol=2e-12, rtol=2e-10)
        errors[label] = float(np.abs(values - reference).max())
    holdings, cash = np.zeros(weights.shape[1]), 1.0
    for t, trade in enumerate(actual.trade_values):
        marked = holdings * (1 + returns[t])
        np.testing.assert_allclose(marked + trade, actual.weights[t] * actual.cumulative_equity[t], atol=2e-12)
        np.testing.assert_allclose(cash - trade.sum() - actual.fees[t], actual.cash_equity[t], atol=2e-12)
        assert np.all(trade[~buy[t]] <= 0) and np.all(trade[~sell[t]] >= 0)
        if caps is not None:
            assert np.all(np.abs(trade) <= caps[t] + 2e-12)
        holdings, cash = marked + trade, actual.cash_equity[t]
    np.testing.assert_allclose(actual.unfilled_trade_values,
                               actual.permission_unfilled + actual.cap_unfilled + actual.cash_unfilled,
                               atol=2e-12)
    # Rewrite every future input, including permissions and capacities.
    split = len(weights) // 2
    changed_w, changed_r, changed_b, changed_s = (value.copy() for value in (weights, returns, buy, sell))
    changed_w[split:] = 0; changed_r[split:] *= -1
    changed_b[split:] = ~changed_b[split:]; changed_s[split:] = ~changed_s[split:]
    changed_cap = None if caps is None else caps.copy()
    if changed_cap is not None:
        changed_cap[split:] = 0
    changed = run_constrained_backtest(changed_w, changed_r, buy_allowed=changed_b,
                                      sell_allowed=changed_s, max_trade_values=changed_cap, costs_bps=costs_bps)
    for attr in ("cumulative_equity", "trade_values", "fees", "cash_equity", "requested_trade_values",
                 "unfilled_trade_values", "pretrade_weights", "buy_fill_scale"):
        np.testing.assert_array_equal(getattr(actual, attr)[:split], getattr(changed, attr)[:split])
    return actual, errors


@click.command()
@click.option('--json-out', required=True, type=click.Path(path_type=Path))
def main(json_out):
    seed = 20261121
    rng = np.random.default_rng(seed)
    w = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    buy, sell = np.ones_like(w, dtype=bool), np.ones_like(w, dtype=bool)
    buy[1:3, 0] = False; sell[1:3, 0] = False
    random_w = rng.dirichlet(np.ones(5), size=80)[:, :4] * .95
    scale_w = np.full((500, 200), .9 / 200)
    cases = [
        ("locked_exit_followed_by_loss", w, np.array([[0., 0.], [0., 0.], [-.3, .2], [0., 0.]]), buy, sell, None, 20.),
        ("locked_exit_followed_by_gain", w, np.array([[0., 0.], [0., 0.], [.3, -.2], [0., 0.]]), buy, sell, None, 20.),
        ("mixed_directional_and_capacity_constraints", random_w, rng.uniform(-.08, .08, random_w.shape),
         rng.random(random_w.shape) > .2, rng.random(random_w.shape) > .3,
         rng.uniform(0, .15, random_w.shape), 100.),
        ("synthetic_500x200", scale_w, rng.normal(0, .015, scale_w.shape),
         rng.random(scale_w.shape) > .04, rng.random(scale_w.shape) > .05,
         np.full(scale_w.shape, .0025), 5.),
    ]
    results = []
    for name, weights, returns, buys, sells, caps, fee in cases:
        actual, errors = audit_case(weights, returns, buys, sells, caps, fee)
        full = run_self_financing_backtest(weights, returns, costs_bps=fee)
        released = run_constrained_backtest(weights, returns, costs_bps=fee)
        for attr in vars(full):
            value = getattr(full, attr)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(released, attr))
        record = {"case": name, "shape": list(weights.shape), "one_way_costs_bps": fee,
                  "maximum_absolute_errors": errors, "future_mutation_check": "passed",
                  "fully_released_compatibility": "bit-identical arrays",
                  "constrained_metrics": actual.metrics,
                  "complete_fill_final_equity": float(full.cumulative_equity[-1]),
                  "constrained_final_equity": float(actual.cumulative_equity[-1]),
                  "input_sha256": {label: hashlib.sha256(value.tobytes()).hexdigest()
                                   for label, value in (("weights", weights), ("returns", returns),
                                                        ("buy", buys), ("sell", sells))},
                  "daily": {"equity": actual.cumulative_equity.tolist(), "fees": actual.fees.tolist(),
                            "cash": actual.cash_equity.tolist(), "complete_fill_equity": full.cumulative_equity.tolist()}}
        if caps is not None:
            record["input_sha256"]["caps"] = hashlib.sha256(caps.tobytes()).hexdigest()
        if len(weights) <= 4:
            record["ledger"] = {attr: getattr(actual, attr).tolist() for attr in (
                "weights", "trade_values", "requested_trade_values", "unfilled_trade_values",
                "permission_unfilled", "cap_unfilled", "cash_unfilled", "buy_fill_scale")}
        results.append(record)
    root = Path(__file__).resolve().parents[1]
    cpu = platform.processor() or platform.machine()
    if sys.platform == 'darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    report = {"status": "passed", "category": "synthetic historical-execution engineering diagnostic",
              "commit": subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              "command": shlex.join([sys.executable, *sys.argv]),
              "python": sys.version, "numpy": np.__version__, "pytorch": torch.__version__,
              "os": platform.platform(), "cpu": cpu, "device": "cpu", "gpu": "unused", "seed": seed,
              "data_source": "Constructed ledgers and seeded NumPy arrays; no market downloads",
              "cost_assumptions": "Actual notional one-way fees; initial NAV 1; cash earns zero; same-row sale proceeds",
              "oracle_precision_digits": 60, "oracle_rounding_floor": "1e-45",
              "implementation_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                        for module in (constrained, self_financing)},
              "cases": results,
              "limitations": ["Synthetic policy/accounting validation, not alpha or market execution evidence.",
                              "No settlement, order queue, calibrated participation, slippage or lot-size model.",
                              "Complete-fill versus constrained equity can change in either direction."]}
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + '\n')
    click.echo(f"Passed 4 Decimal cases, future-mutation and released-constraint controls; wrote {json_out}")


if __name__ == '__main__':
    main()
