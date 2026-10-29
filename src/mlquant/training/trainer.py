"""A small, self-contained training loop.

This is *not* a competitor to PyTorch Lightning — it's the minimum
needed to get a model trained, validated, and checkpointed against a
:class:`FactorDataset`. Everything is explicit so the training-time
behaviour matches what's documented in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..models.losses import ICLoss, RankICLoss
from .dataset import FactorDataset
from .samplers import CrossSectionBatchSampler, date_boundaries


@dataclass
class TrainConfig:
    epochs:      int   = 30
    batch_size:  int   = 1024
    lr:          float = 1e-3
    weight_decay:float = 1e-5
    grad_clip:   float = 1.0
    device:      str   = "cpu"
    log_every:   int   = 50
    save_dir:    str   = "checkpoints"
    val_split:   float = 0.1
    # "time" holds out the most recent complete dates. "random" splits dates
    # for daily batches, or individual samples for legacy regression batches.
    val_split_mode: str = "time"
    extra:       dict  = field(default_factory=dict)
    # "auto" uses complete daily batches for IC/RankIC, flat batches otherwise.
    # "date" also supports custom cross-sectional objectives. In date mode,
    # batch_size is ignored and losses are averaged equally over observed dates.
    batch_mode: str = "auto"
    # Remove this many complete observed signal dates immediately before time
    # validation. Choose from label horizon/information timing; zero keeps the
    # default one-day close-to-close protocol. Random splits cannot use a gap.
    purge_dates: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cfg: TrainConfig,
    ) -> None:
        self.model = model.to(cfg.device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.split_info: dict = {}
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, dataset: FactorDataset) -> dict:
        if not math.isfinite(self.cfg.val_split) or not 0 < self.cfg.val_split < 1:
            raise ValueError("val_split must be finite and between zero and one")
        if self.cfg.epochs < 1 or self.cfg.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.cfg.val_split_mode not in ("time", "random"):
            raise ValueError("val_split_mode must be 'time' or 'random'")
        if self.cfg.batch_mode not in ("auto", "samples", "date"):
            raise ValueError("batch_mode must be 'auto', 'samples' or 'date'")
        if type(self.cfg.purge_dates) is not int or self.cfg.purge_dates < 0:
            raise ValueError("purge_dates must be a nonnegative integer")
        if self.cfg.purge_dates and self.cfg.val_split_mode != "time":
            raise ValueError("purge_dates requires val_split_mode='time'")
        correlation = isinstance(self.loss_fn, (ICLoss, RankICLoss))
        if correlation and self.cfg.batch_mode == "samples":
            raise ValueError("IC/RankIC require complete date batches")
        daily = self.cfg.batch_mode == "date" or (self.cfg.batch_mode == "auto" and correlation)
        if len(dataset) < 2 or dataset.dates.shape != (len(dataset),):
            raise ValueError("training requires at least two samples with aligned dates")
        if not torch.isfinite(dataset.dates).all():
            raise ValueError("sample dates must be finite")
        purged_indices = range(0)
        if self.cfg.val_split_mode == "time" or daily:
            boundaries = date_boundaries(dataset.dates)
            n_dates = len(boundaries) - 1
            if n_dates < 2:
                raise ValueError("date splitting requires at least two observed dates")
            n_val_dates = max(1, int(n_dates * self.cfg.val_split))
            if self.cfg.val_split_mode == "time":
                split = boundaries[n_dates - n_val_dates]
                n_train_dates = n_dates - n_val_dates - self.cfg.purge_dates
                if n_train_dates < 1:
                    raise ValueError("purge_dates leaves no training dates")
                train_end = boundaries[n_train_dates]
                train_ds = Subset(dataset, range(train_end))
                val_ds = Subset(dataset, range(split, len(dataset)))
                purged_indices = range(train_end, split)
            else:
                validation_days = set(torch.randperm(
                    n_dates, generator=torch.Generator().manual_seed(0)
                )[:n_val_dates].tolist())
                train_indices, val_indices = [], []
                for day in range(n_dates):
                    target = val_indices if day in validation_days else train_indices
                    target.extend(range(boundaries[day], boundaries[day + 1]))
                train_ds, val_ds = Subset(dataset, train_indices), Subset(dataset, val_indices)
        else:
            n_val = max(1, int(len(dataset) * self.cfg.val_split))
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [len(dataset) - n_val, n_val],
                generator=torch.Generator().manual_seed(0),
            )
        if daily:
            train_loader = DataLoader(train_ds, batch_sampler=CrossSectionBatchSampler(
                dataset.dates[train_ds.indices], shuffle=True))
            val_loader = DataLoader(val_ds, batch_sampler=CrossSectionBatchSampler(
                dataset.dates[val_ds.indices]))
        else:
            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size)

        history: dict = {"train_loss": [], "val_loss": []}
        partitions = {"train": train_ds.indices, "validation": val_ds.indices,
                      "purged": purged_indices}
        self.split_info = {
            "schema_version": 1, "val_split_mode": self.cfg.val_split_mode,
            "batch_mode": "date" if daily else "samples", "val_split": float(self.cfg.val_split),
            "purge_dates": self.cfg.purge_dates, "purge_unit": "observed_signal_dates",
            "partitions": {},
        }
        for name, indices in partitions.items():
            dates = dataset.dates[indices]
            self.split_info["partitions"][name] = {
                "samples": len(indices), "observed_dates": dates.unique().numel(),
                "date_range": [dates.min().item(), dates.max().item()] if dates.numel() else None,
            }
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "split.json").write_text(
            json.dumps(self.split_info, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        best_val = float("inf")

        for epoch in range(self.cfg.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss   = self._run_epoch(val_loader,   train=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"[epoch {epoch:3d}] train={train_loss:.5f}  val={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), save_dir / "best.pt")
        torch.save(self.model.state_dict(), save_dir / "last.pt")
        return history

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(features.to(self.cfg.device)).detach().cpu()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: Iterable, *, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        daily = isinstance(getattr(loader, "batch_sampler", None), CrossSectionBatchSampler)
        for x, y in loader:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            if train:
                self.optim.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
            if train:
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optim.step()
            weight = 1 if daily else x.shape[0]
            total += loss.item() * weight
            n += weight
        return total / max(1, n)
