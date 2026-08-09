# Six Invariants I Check Before Trusting a Quantitative ML Backtest

Backtest discussions often begin with Sharpe ratio, annualized return, or the
model architecture. In practice, a pipeline can produce an attractive chart
while getting a much simpler question wrong: **what information was available,
which cells were valid, and when did a position begin earning returns?**

I added a deterministic engineering audit to
[`ml-quant-trading`](https://github.com/initial-d/ml-quant-trading), an open
PyTorch research stack that builds a 213-dimensional factor tensor and carries
it through a model, portfolio construction, and a cost-aware backtest.

The audit does not try to prove alpha. It checks six implementation invariants
that should hold before performance claims are even discussed.

## Reproduce it

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e '.[dev]'
python scripts/technical_pipeline_audit.py
```

The command uses deterministic synthetic data, needs no market-data account,
and writes both Markdown and JSON reports. The maintained reference run is
available in
[`docs/audits/technical-audit-20260809`](audits/technical-audit-20260809/technical_audit.md).

## 1. The factor catalogue has an explicit contract

The first check computes the complete registered catalogue and verifies the
shape rather than trusting a configuration label:

```python
factors, factor_mask, names = compute_legacy_set(panel)

assert factors.shape == (180, 40, 213)
assert len(names) == 213
assert torch.isfinite(factors).all()
```

On the reference CPU run, the output was a finite `[180, 40, 213]` tensor. The
runtime is recorded as environment-specific evidence, not advertised as a
universal speed claim.

This catches mundane but consequential failures: a factor silently dropping
from the registry, an unexpected axis order, or NaN/Inf propagation that is
hidden by later aggregation.

## 2. Synthetic input generation is bitwise deterministic

Synthetic data is not market evidence, but it is valuable for engineering
reproduction. Two panels generated with the same configuration and seed must
match exactly:

```python
first = make_synthetic_panel(config)
second = make_synthetic_panel(config)

assert torch.equal(first.close, second.close)
assert torch.equal(first.volume, second.volume)
assert torch.equal(first.mask, second.mask)
```

If this fails, benchmark drift and regression failures become difficult to
attribute. Determinism gives contributors a zero-account test fixture before
they introduce yfinance, AkShare, Baostock, or licensed data.

## 3. Masked cells cannot contaminate valid primitive outputs

Suspensions, pre-listing periods, limit events, and missing observations are not
ordinary zeroes. The factor engine therefore passes a `[date, asset]` Boolean
mask through its tensor primitives.

The audit performs a poisoning test. It replaces every masked input with a
value of magnitude `1e9`, recomputes the operation, and compares only cells that
are valid in both runs:

```python
poisoned_x = torch.where(mask, x, torch.full_like(x, 1e9))

clean, clean_mask = ts_mean(x, mask, 20)
poisoned, poisoned_mask = ts_mean(poisoned_x, mask, 20)
valid = clean_mask & poisoned_mask

assert (clean - poisoned).abs()[valid].max() <= 1e-6
```

The reference audit applies this test to five core primitives:

- cross-sectional rank;
- rolling mean;
- rolling rank;
- rolling correlation;
- exponentially weighted mean.

All five produced a maximum valid-cell drift of exactly `0` after masked cells
were poisoned. This is deliberately a primitive-level invariant; it should not
be misread as independent validation of every economic factor definition.

## 4. Forward labels stop at the sample boundary

For a one-period target, the label at time `t` is only valid when the asset is
tradable at both `t` and `t+1`:

```python
target[t] = close[t + 1] / close[t] - 1
target_mask[t] = tradable[t] & tradable[t + 1]
```

The audit reconstructs this expected mask independently and requires exact
equality. It also requires the final row to be masked and zero because no
`t+1` observation exists.

This small boundary condition prevents the last sample from being interpreted
as a real zero-return label and makes the execution assumption inspectable.

## 5. Positions are lagged before they earn returns

The backtest convention is explicit:

```text
weights[t-1] earn returns[t]
```

The audit feeds a hand-constructed two-asset path into the engine. With zero
costs, the expected gross returns are `[0.00, 0.10, -0.20, 0.30]`. The observed
path must match exactly.

This controlled example is more useful than checking a final equity number.
Without the lag, same-period weights can accidentally earn the return used to
choose them—a classic look-ahead error.

## 6. Transaction-cost drag scales with the configured fee

Using an alternating two-asset book with zero asset returns, the audit runs the
same weights at 1 bps and 10 bps. The cumulative cost ratio must be exactly 10:

```python
one = run_backtest(weights, returns, costs_bps=1).cost_drag_cumulative
ten = run_backtest(weights, returns, costs_bps=10).cost_drag_cumulative

assert np.isclose(ten / one, 10.0)
```

This confirms the unit convention and separates cumulative cost paid over the
run from annualized return metrics.

## Reference result

The maintained run on commit `575a71c` passed all six checks:

| Invariant | Result |
|---|---:|
| 213-factor tensor shape and finiteness | PASS |
| Deterministic synthetic generation | PASS |
| Mask isolation for five core primitives | PASS |
| Forward-label boundary and endpoint mask | PASS |
| Lagged execution alignment | PASS |
| Transaction-cost linearity | PASS |

The report includes the Python, PyTorch, platform, seed, panel dimensions, and
machine-readable evidence for each check.

## What this audit does not prove

Passing an engineering audit does not prove:

- that a factor has out-of-sample predictive power;
- that synthetic data resembles a particular live market;
- that transaction costs and slippage are fully modeled;
- that an optimizer is safe for production capital;
- that historical results will repeat.

Those claims require public or licensed market data, walk-forward evaluation,
strong baselines, execution modeling, and independent reproduction. The audit
simply moves the conversation one step earlier: before asking whether a model
is profitable, make sure the pipeline means what its code says it means.

The package can be tried with:

```bash
python -m pip install mlquantx
mlquant demo
```

Source and reproduction reports:
[github.com/initial-d/ml-quant-trading](https://github.com/initial-d/ml-quant-trading).

