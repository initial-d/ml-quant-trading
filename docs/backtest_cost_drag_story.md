# The Backtest Metric Was Correct — and Still Easy to Misread

An external contributor found a reporting problem in `ml-quant-trading` that is
worth checking in any backtest framework.

The engine reported `cost_drag` beside `ann_return`, `gross_ann_return`, and
`ann_vol`. Those neighboring metrics are annualized. `cost_drag` was not: it was
the arithmetic sum of trading costs over the entire run.

The calculation was correct. The interface made an incorrect comparison look
reasonable.

## The unit test that exposed the problem

[@sergio12S](https://github.com/sergio12S) ran the same synthetic configuration
at three different backtest lengths. Turnover and cost per year stayed roughly
constant, while the reported total grew with the number of periods:

| periods | annual return | turnover | old `cost_drag` | approximate cost per year |
|---:|---:|---:|---:|---:|
| 600 | -0.3274 | 0.1899 | 0.1592 | 0.0669 |
| 1,200 | -0.1844 | 0.1854 | 0.3112 | 0.0654 |
| 2,400 | -0.1158 | 0.1891 | 0.6351 | 0.0667 |

Doubling the run roughly doubled `cost_drag`. That is exactly what a cumulative
quantity should do — and exactly what an annualized quantity should not do.

The finding and implementation are reviewable in
[PR #47](https://github.com/initial-d/ml-quant-trading/pull/47).

## Why subtraction does not fix it

It is tempting to calculate an annual cost number as:

```text
gross_ann_return - ann_return
```

or to divide cumulative cost by the number of years. Neither produces an exact
identity:

- cumulative cost is an arithmetic sum of per-period costs;
- annual returns are geometric, compounded quantities;
- costs alter the equity path on which later returns compound.

An approximate annualized cost figure can be useful for diagnostics, but it
should not be published under an exact-sounding name without a defined formula.
This project therefore chose the narrower fix: name the quantity for what it is.

## The reporting contract now

New output uses:

```python
result.metrics["cost_drag_cumulative"]
result.cost_drag_cumulative
```

The old name remains available as a compatibility alias:

```python
result.metrics["cost_drag"]
result.cost_drag
```

Report aggregation and auditing also accept archived JSON files that contain
only `cost_drag`. New Markdown and cost-sensitivity tables emit the explicit
name.

Regression tests pin three behaviors:

1. twice as many periods produce roughly twice the cumulative cost;
2. ten times the fee produces ten times the cumulative cost;
3. the legacy alias returns the same value.

The complete unit and time-basis glossary lives in
[Public-Data Validation](public_data_validation.md#metric-glossary).

## Audit your own backtest

For every metric in a result table, write down two things:

| Question | Typical answers |
|---|---|
| What is the unit? | fraction, currency, basis points, ratio, multiple |
| What is the time basis? | per period, per year, average rebalance, cumulative run |

Then look for rows that mix cumulative and annualized values. Pay special
attention to costs, turnover, drawdown, alpha, and final equity. A correct
calculation with an ambiguous label can survive tests and still lead readers to
the wrong conclusion.

## Why this contribution matters

Open-source research becomes more credible when outsiders can challenge the
reporting contract, not only the implementation. This change was proposed by a
community contributor, reviewed against archived-report compatibility, tested
locally, and passed CI on Python 3.9, 3.10, and 3.11.

That review trail is more useful than a perfect-looking backtest screenshot.

Try the project in the
[zero-account Colab](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb),
or [report another ambiguous metric](https://github.com/initial-d/ml-quant-trading/issues/new?template=research_question.yml).
