# v0.2.4 - Metric Clarity and Contributor-Led Review

`v0.2.4` makes the time basis of backtest trading costs explicit. An external
contributor showed that the old `cost_drag` label sat beside annualized metrics
even though it represented a cumulative total over the whole run.

The calculation was correct; the reporting contract was too easy to misread.

## Highlights

- New output uses `cost_drag_cumulative` in result tables and reports.
- The legacy `cost_drag` key and property remain available as compatibility
  aliases.
- Aggregation and audit tools continue to accept archived reports.
- A metric glossary now records the unit and time basis of every validation
  column.
- Regression tests cover duration scaling, fee scaling, and legacy access.
- The README now leads with the runnable path and community-reviewed evidence.
- Package metadata and `mlquant.__version__` are synchronized at `0.2.4`.

## Community Credit

Thank you to [@sergio12S](https://github.com/sergio12S) for identifying the
ambiguity, supplying controlled evidence, designing the compatibility path,
and adding the implementation, documentation, and tests in
[PR #47](https://github.com/initial-d/ml-quant-trading/pull/47).

Read the full technical story:
[The Backtest Metric Was Correct — and Still Easy to Misread](https://github.com/initial-d/ml-quant-trading/blob/main/docs/backtest_cost_drag_story.md).

## Migration

Prefer the explicit field in new code:

```python
result.cost_drag_cumulative
result.metrics["cost_drag_cumulative"]
```

Existing reads continue to work:

```python
result.cost_drag
result.metrics["cost_drag"]
```

## Verification

- Full local test suite: 92 passed.
- Ruff: passed.
- GitHub CI: Python 3.9, 3.10, and 3.11 passed, including the CLI smoke test.

## Research Boundary

This release improves reporting clarity. It does not add a new trading signal,
change historical validation results, or claim live-trading profitability.
