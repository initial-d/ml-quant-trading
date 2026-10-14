"""The trainer's default validation split must respect time order."""
from __future__ import annotations

import pytest
import torch

from mlquant.models.nets import MLPRegressor
from mlquant.training.dataset import FactorDataset
from mlquant.training.trainer import TrainConfig, Trainer


def _dataset(T: int = 40, N: int = 5, F: int = 3) -> FactorDataset:
    g = torch.Generator().manual_seed(0)
    return FactorDataset(
        factors=torch.randn(T, N, F, generator=g),
        mask=torch.ones(T, N, dtype=torch.bool),
        forward_returns=torch.randn(T, N, generator=g) * 0.01,
    )


def test_dataset_samples_are_ordered_by_date():
    ds = _dataset()
    assert torch.all(ds.dates[1:] >= ds.dates[:-1])


def test_time_split_validates_on_the_latest_dates(tmp_path, monkeypatch):
    ds = _dataset()
    seen = {}
    real_loader = torch.utils.data.DataLoader

    def spy(subset, *args, **kwargs):
        seen.setdefault("train" if kwargs.get("shuffle") else "val", list(subset.indices))
        return real_loader(subset, *args, **kwargs)

    monkeypatch.setattr("mlquant.training.trainer.DataLoader", spy)
    cfg = TrainConfig(epochs=1, batch_size=8, save_dir=str(tmp_path), val_split=0.25)
    Trainer(MLPRegressor(in_dim=3, hidden=4), torch.nn.MSELoss(), cfg).fit(ds)

    last_train_date = ds.dates[seen["train"]].max()
    first_val_date = ds.dates[seen["val"]].min()
    assert last_train_date <= first_val_date


def test_unknown_split_mode_is_rejected(tmp_path):
    cfg = TrainConfig(epochs=1, save_dir=str(tmp_path), val_split_mode="shuffle")
    trainer = Trainer(MLPRegressor(in_dim=3, hidden=4), torch.nn.MSELoss(), cfg)
    with pytest.raises(ValueError, match="val_split_mode"):
        trainer.fit(_dataset())
