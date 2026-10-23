# Observation, factor, label and execution masks

These masks answer different questions. A price-limit close can be an observed
price even when a research policy excludes it from factor windows. It can also
be a valid endpoint of a historical return label. Neither fact establishes that
a particular buy or sell order would fill.

| Mask | Question | Current implementation |
| --- | --- | --- |
| Observation | Does the loader accept this stock-day? | Original `Panel.mask`; loader-specific, sometimes halt-aware |
| Factor | Does the selected research policy accept the inputs and computed features at signal date t? | Mask-first panel, then factor-engine mask |
| Forward label | Is the return label aligned to t usable? | `target_01(original_panel)` returns values and an endpoint mask |
| Execution | Would a specified order fill at the chosen time/price? | Not provided by the masks above; requires side, timing, liquidity and queue assumptions |

`limit_move_mask` is a conservative symmetric exclusion policy. Its return proxy
is a heuristic; explicit bands, board/date thresholds and price tolerance must
be reported. Masking before rolling factors implements that policy throughout
the window. This is not proof that observed limit prices are invalid or that a
different exclusion policy contains lookahead bias.

## Training contract

```python
from mlquant.features import compute_legacy_set
from mlquant.features.bias import apply_limit_mask
from mlquant.features.target import target_01
from mlquant.training import FactorDataset

# Keep the original observations for historical labels and held-position P&L.
factors, factor_mask, names = compute_legacy_set(apply_limit_mask(panel))
labels, label_mask = target_01(panel)
dataset = FactorDataset(factors, factor_mask, labels, label_mask=label_mask)
```

`labels[t]` and `label_mask[t]` already refer to the return from t to t+1.
The dataset does not shift them again. A sample requires usable factors at t,
a valid forward label at t, and finite feature/label values. The final date is
excluded for this next-day dataset.

For example, if tomorrow closes at a price limit, tomorrow's factor mask can
be false while today's forward label remains valid. The explicit label mask
retains that sample. Omitting `label_mask` keeps the old intersection of factor
masks at t and t+1 for API compatibility. The CLI now supplies the label mask.
This changes the training sample set, so prior demo model metrics can change.

Future label availability is an offline training/evaluation filter. It must not
select today's predictions or portfolio universe. The existing public-validation
protocol separately documents [signal-date availability](public_data_validation.md#signal-date-availability).

## Deterministic regression checks

```bash
python -m pytest tests/test_dataset_masks.py tests/test_bias.py tests/test_trainer_split.py
```

The cases cover an observed limit close, an invalid zero-filled label, alignment,
nonfinite values, shape validation, and chronological samples. All inputs are
synthetic engineering controls, with no market-data fetch or profitability claim.
See the [recorded run](audits/training-masks-20261023/README.md).
