# Purged model refitting on the original date clock

The source pipeline can refit a fresh MLP for each disjoint future prediction
window. This extends the single chronological split with sequential model
updates; it does not select a profitable strategy or repair noncausal inputs.

```bash
mlquant gen-data --config configs/small.yaml
mlquant features --config configs/small.yaml
mlquant train --config configs/small.yaml --walk-forward --test-dates 50
mlquant portfolio --config configs/small.yaml
mlquant backtest --config configs/small.yaml --accounting constrained
mlquant rebalance --config configs/small.yaml \
  --currency CNY --data-category synthetic --data-source synthetic-demo
```

`train --walk-forward` deliberately replaces the usual `predictions.pt`. Run
`portfolio` again before backtesting or sweeping capital. New portfolio files
record the prediction hash; backtest/capacity reject mismatched forecasts and
weights. `rebalance` consumes forecasts directly. Earlier files without this
hash retain historical compatibility and need manual provenance checks.

Without `--train-dates`, the first test row remains `floor(T * train_frac)`
(subject to the existing small-panel clamp). Its initial training-history length
is that row minus the original-row purge. With an explicit value, first test
starts at `train_dates + purge_dates`. Test windows are adjacent, with the final
partial window retained. `--expanding` accumulates history from row zero;
otherwise every refit uses `train_dates` trailing history rows. These counts
include missing dates rather than compressing the calendar.

## Information timing

The CLI labels are one-period simple returns, aligned to signal row `t`.
For first test row `s`, history ends **exclusively** at `s-purge_dates`.
The default gap is one; every historical label therefore ends strictly before
`s`. This is a deliberately conservative information convention relative to the
legacy static training path, which can consume a label ending at `s`.

Inner validation is the latest complete observed dates in that history window.
At least `label_horizon` observed signal dates are removed immediately before
validation. Because original row IDs are retained, the final training label
ends strictly before the earliest validation signal even when dates are missing.
Existing `train.purge_dates` may enlarge this inner gap; it counts observed dates,
whereas the CLI `--purge-dates` counts original panel rows.

Every fold creates a fresh model, loss and AdamW optimizer. Select the checkpoint
with the minimum inner-validation loss and reload it before prediction. Each
signal date is predicted separately, preventing future dates from entering a
prediction batch. No preprocessing statistics are fitted to the full panel.
Both training and validation remain entirely before their test window.

Earlier test observations become available history for later refits. The combined
series is sequentially out of sample, not one permanently untouched holdout.
Choosing windows, features, universe or hyperparameters after inspecting that
series still requires an independent outer holdout.

## API and artifacts

```python
import torch
from mlquant.models.nets import MLPRegressor
from mlquant.training import TrainConfig, WalkForwardConfig, train_walk_forward

result = train_walk_forward(
    factors, factor_mask, forward_returns, label_mask,
    model_factory=lambda: MLPRegressor(factors.shape[-1], hidden=32),
    loss_factory=torch.nn.MSELoss,
    train_cfg=TrainConfig(epochs=8, save_dir="artifacts/refits"),
    cfg=WalkForwardConfig(train_dates=200, test_dates=50, label_horizon=1),
)
```

Inputs are aligned `[T,N,F]` factors and `[T,N]` masks/labels on the same device.
Labels must already be aligned and their declared horizon accurate. The CLI
supports the next-day labels it constructs; the API also accepts longer horizons.
Factories must create fresh models/losses, and model outputs must be finite
one-dimensional cross-sectional forecasts. IC/RankIC objectives reuse complete
daily batches. Model/factor dtypes must be compatible.

`result.predictions` and `result.prediction_mask` cover the original `[T,N]` grid;
`fold_id[T]` identifies each test row. Pre-test and unavailable factor cells are
zero with a false prediction mask. Zero is a storage placeholder, not a claim
that an unavailable forecast is valid. The prediction mask uses current factor
availability and finiteness, never future label availability. Portfolio selection
uses it; cost-aware rebalancing requires it for buys while retaining sell permissions.

The CLI adds `walk-forward.json` and embeds its report in `predictions.pt`.
The report records original-row boundaries, effective inner gaps, sample/date
counts, losses, selected epoch, seed and checkpoint hash per fold, coverage,
input hashes, command, software/hardware and source commit. Backtest reports
identify sequential refitting rather than describing one frozen model.

Checkpoints have content-addressed names under
`artifacts_dir/walk-forward/checkpoints/`. Re-running can retain different
checkpoints without overwriting earlier hash-addressed evidence. Temporary fold
directories are removed. Insufficient history, invalid losses/checkpoints or
invalid predictions abort with a fold-specific error; the CLI only publishes new
predictions after every scheduled fold succeeds. It does not silently count an
untrained fold as an all-cash success.

Torch CPU/CUDA random-generator states are scoped and restored; each fold uses
`(seed + fold) % 2**32`. This does not promise bit-identical training across
PyTorch releases, hardware, nondeterministic CUDA kernels, or custom factories
that consume other global random generators.

## Evidence

The audit script compares rolling and expanding refits against the exact first
fold's frozen best checkpoint, under predeclared stationary, sign-switch and
no-signal synthetic processes with two seeds. It retains per-fold MSE/IC and all
outcome directions, runtime, configuration and checkpoint provenance:

```bash
python scripts/audit_walk_forward_training.py \
  --output-dir artifacts/walk-forward-training-audit
```

The independent single-step Adam gradient oracle, checkpoint-selection test,
sparse-calendar boundaries and future-mutation checks exercise the implementation.
Planted-process results explain behavior under constructed drift; they are not
market quality, cost, portfolio P&L or investment evidence. See the
[retained run](audits/model-walk-forward-20261124/README.md).
