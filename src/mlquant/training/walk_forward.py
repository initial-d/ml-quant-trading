"""Fresh, purged models for disjoint chronological prediction windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import torch
from torch import nn

from .dataset import FactorDataset
from .trainer import TrainConfig, Trainer


@dataclass(frozen=True)
class WalkForwardConfig:
    train_dates: int = 200
    test_dates: int = 50
    label_horizon: int = 1
    purge_dates: int | None = None
    window: str = "rolling"
    seed: int = 42

    def __post_init__(self):
        for name in ("train_dates", "test_dates", "label_horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        gap = self.label_horizon if self.purge_dates is None else self.purge_dates
        if type(gap) is not int or gap < self.label_horizon:
            raise ValueError("purge_dates must be an integer >= label_horizon")
        if self.window not in ("rolling", "expanding"):
            raise ValueError("window must be rolling or expanding")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    @property
    def gap(self):
        return self.label_horizon if self.purge_dates is None else self.purge_dates


@dataclass(repr=False)
class WalkForwardResult:
    predictions: torch.Tensor
    prediction_mask: torch.Tensor
    fold_id: torch.Tensor
    report: dict


def train_walk_forward(
    factors: torch.Tensor, factor_mask: torch.Tensor, forward_returns: torch.Tensor,
    label_mask: torch.Tensor, *, model_factory: Callable[[], nn.Module],
    loss_factory: Callable, train_cfg: TrainConfig | None = None,
    cfg: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Refit from scratch on history; use the best inner-validation checkpoint.

    Labels are already aligned to signal row t and end at t+label_horizon.
    For test start s, eligible training signals stop exclusively at s-gap;
    hence every training AND validation label ends strictly before s.
    Inner validation uses complete observed dates, purging at least horizon
    observed dates; this also guarantees a strict gap on the original row clock.

    Prediction uses only finite, signal-available factors, one cross-section at
    a time. Future label availability never gates forecasts. Invalid and pre-OOS
    cells contain zero and prediction_mask=False; they are not valid forecasts.
    No normalization or feature/universe selection is inferred from this API.
    Supply causal, independently fixed inputs, factories and hyperparameters.
    """
    config, training = cfg or WalkForwardConfig(), train_cfg or TrainConfig()
    if factors.ndim != 3 or min(factors.shape) < 1 or not factors.is_floating_point():
        raise ValueError("factors must be nonempty floating [T,N,F]")
    shape = factors.shape[:2]
    for name, tensor in (("factor_mask", factor_mask), ("forward_returns", forward_returns),
                         ("label_mask", label_mask)):
        if tensor.shape != shape or tensor.device != factors.device:
            raise ValueError(name + " must share the [T,N] grid and device")
    if factor_mask.dtype != torch.bool or label_mask.dtype != torch.bool:
        raise ValueError("factor_mask and label_mask must be bool")
    if not forward_returns.is_floating_point():
        raise ValueError("forward_returns must be floating")
    if training.val_split_mode != "time":
        raise ValueError("walk-forward requires val_split_mode='time'")
    if type(training.purge_dates) is not int or training.purge_dates < 0:
        raise ValueError("training purge_dates must be a nonnegative integer")
    first_test = config.train_dates + config.gap
    if first_test >= shape[0]:
        raise ValueError("no test rows remain after training and purge")
    predictions = torch.zeros(shape, dtype=factors.dtype, device="cpu")
    prediction_mask = torch.zeros(shape, dtype=torch.bool, device="cpu")
    fold_id = torch.full((shape[0],), -1, dtype=torch.long)
    folds = []
    root = Path(training.save_dir)
    inner_gap = max(training.purge_dates, config.label_horizon)
    # manual_seed touches every CUDA generator; restore every touched state.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for start in range(first_test, shape[0], config.test_dates):
        number, stop = len(folds), min(start + config.test_dates, shape[0])
        train_stop = start - config.gap
        train_start = train_stop - config.train_dates if config.window == "rolling" else 0
        # The extra row is dropped by FactorDataset, not used as a label gate.
        section = slice(train_start, train_stop + 1)
        dataset = FactorDataset(factors[section], factor_mask[section], forward_returns[section],
                                label_mask=label_mask[section],
                                date_index=torch.arange(train_start, train_stop + 1, device=factors.device))
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".fold-{number:04d}-", dir=root) as temporary:
            fold_cfg = replace(training, save_dir=temporary, purge_dates=inner_gap)
            seed = (config.seed + number) % 2**32
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                model, loss = model_factory(), loss_factory()
                trainer = Trainer(model, loss, fold_cfg)
                try:
                    history = trainer.fit(dataset)
                except ValueError as exc:
                    raise ValueError(f"fold {number} (test row {start}): {exc}") from exc
                if any(not math.isfinite(value) for losses in history.values() for value in losses):
                    raise ValueError(f"fold {number}: nonfinite training/validation loss")
                best = Path(temporary) / "best.pt"
                if not best.exists():
                    raise ValueError(f"fold {number}: no finite best validation checkpoint")
                state = torch.load(best, map_location=fold_cfg.device, weights_only=True)
                if any(not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError(f"fold {number}: nonfinite best checkpoint")
                model.load_state_dict(state)
                for row in range(start, stop):
                    usable = factor_mask[row] & torch.isfinite(factors[row]).all(-1)
                    if not usable.any():
                        continue
                    values = trainer.predict(factors[row, usable])
                    if values.shape != (int(usable.sum()),) or not torch.isfinite(values).all():
                        raise ValueError(f"fold {number}, row {row}: model must return finite [assets] predictions")
                    positions = usable.cpu()
                    predictions[row, positions] = values.to(predictions.dtype)
                    prediction_mask[row, positions] = True
            digest = hashlib.sha256(best.read_bytes()).hexdigest()
            # Content-addressed paths preserve earlier runs' checkpoint evidence.
            checkpoint = root / f"fold-{number:04d}-{digest}.pt"
            shutil.copyfile(best, checkpoint)
        split = trainer.split_info
        partitions = split["partitions"]
        last_train = partitions["train"]["date_range"][1]
        first_validation = partitions["validation"]["date_range"][0]
        last_validation = partitions["validation"]["date_range"][1]
        if not (last_train + config.label_horizon < first_validation
                and last_validation + config.label_horizon < start):
            raise RuntimeError("label timing violates a training/validation/test boundary")
        fold_id[start:stop] = number
        folds.append({"fold": number, "seed": seed, "train_start": train_start,
                      "history_end_exclusive": train_stop, "purge_start": train_stop,
                      "purge_end_exclusive": start, "test_start": start, "test_end_exclusive": stop,
                      "partial_test": stop - start < config.test_dates,
                      "last_training_label_row": last_train + config.label_horizon,
                      "last_validation_label_row": last_validation + config.label_horizon,
                      "split": split, "history": history,
                      "best_epoch": min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__),
                      "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                      "prediction_cells": int(prediction_mask[start:stop].sum()),
                      "prediction_dates": int(prediction_mask[start:stop].any(1).sum())})
    report = {
        "protocol": "model-walk-forward-v1", "config": asdict(config),
        "training_config": asdict(training), "inner_purge_observed_dates": inner_gap,
        "shape": list(factors.shape), "oos_start": first_test, "folds": folds,
        "summary": {"folds": len(folds), "scheduled_test_dates": shape[0] - first_test,
                    "prediction_dates": int(prediction_mask.any(1).sum()),
                    "prediction_cells": int(prediction_mask.sum())},
        "clock": "original panel rows; label at t ends at t+label_horizon; strict endpoint gaps",
        "checkpoint_selection": "minimum trailing-validation loss within each history window; fresh model/optimizer each fold",
        "limitations": ["Input features, label timing, universe and preprocessing must be causal and accurately declared.",
                        "Forecasts use signal availability only; future label eligibility is exclusively an offline training/evaluation filter.",
                        "Hyperparameters and input selection are fixed by the caller; retrospective tuning needs an outer holdout.",
                        "Predictions and coverage are engineering outputs, not portfolio returns or deployable alpha.",
                        "Earlier OOS observations may become later training history; this is sequential refitting, not one permanent holdout."],
    }
    return WalkForwardResult(predictions, prediction_mask, fold_id, report)
