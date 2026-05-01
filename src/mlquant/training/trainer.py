"""A small, self-contained training loop.

This is *not* a competitor to PyTorch Lightning — it's the minimum
needed to get a model trained, validated, and checkpointed against a
:class:`FactorDataset`. Everything is explicit so the training-time
behaviour matches what's documented in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import FactorDataset


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
    extra:       dict  = field(default_factory=dict)


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
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, dataset: FactorDataset) -> dict:
        n_val = max(1, int(len(dataset) * self.cfg.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.batch_size)

        history: dict = {"train_loss": [], "val_loss": []}
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
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
        for x, y in loader:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            if train:
                self.optim.zero_grad(set_to_none=True)
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            if train:
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optim.step()
            total += loss.item() * x.shape[0]
            n     += x.shape[0]
        return total / max(1, n)
