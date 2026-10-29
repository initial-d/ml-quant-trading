"""Label-horizon-aware time splitting and inspectable partition provenance."""
import json

import pytest
import torch

from mlquant.models.losses import ICLoss
from mlquant.training import FactorDataset, TrainConfig, Trainer


def _dataset():
    # Four-period labels, only where both endpoints exist. Keep NaN for the
    # unavailable tail because the legacy dataset has no separate label mask.
    close = 10 + torch.arange(18).float()[:, None] + torch.arange(3).float()[None, :]
    labels = torch.full_like(close, float("nan"))
    labels[:-4] = close[4:] / close[:-4] - 1
    return FactorDataset(close[..., None], torch.ones_like(close, dtype=torch.bool), labels)


@pytest.mark.parametrize("objective", ["mse", "ic"])
@pytest.mark.parametrize("purge", [0, 3, 4])
def test_gap_removes_full_dates_keeps_validation_fixed_and_records_counts(tmp_path, monkeypatch, objective, purge):
    ds = _dataset()
    seen = {}
    real_loader = torch.utils.data.DataLoader

    def record(subset, *args, **kwargs):
        shuffled = kwargs.get("shuffle", getattr(kwargs.get("batch_sampler"), "shuffle", False))
        seen["train" if shuffled else "validation"] = list(subset.indices)
        return real_loader(subset, *args, **kwargs)

    monkeypatch.setattr("mlquant.training.trainer.DataLoader", record)
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    criterion = torch.nn.MSELoss() if objective == "mse" else ICLoss()
    trainer = Trainer(model, criterion, TrainConfig(epochs=1, val_split=.25,
                                                  purge_dates=purge, save_dir=str(tmp_path)))
    history = trainer.fit(ds)
    train_dates = set(ds.dates[seen["train"]].tolist())
    validation_dates = set(ds.dates[seen["validation"]].tolist())
    assert train_dates == set(range(11 - purge))
    assert validation_dates == {11, 12, 13}
    excluded = set(range(11 - purge, 11))
    assert excluded.isdisjoint(train_dates | validation_dates)
    if purge == 3:
        assert max(train_dates) + 4 == min(validation_dates)
    elif purge == 4:
        assert max(train_dates) + 4 < min(validation_dates)
    info = json.loads((tmp_path / "split.json").read_text())
    assert info == trainer.split_info
    assert info["purge_unit"] == "observed_signal_dates"
    assert info["partitions"]["validation"] == {"samples": 9, "observed_dates": 3, "date_range": [11, 13]}
    assert info["partitions"]["purged"]["samples"] == 3 * purge
    assert info["partitions"]["purged"]["observed_dates"] == purge
    assert sum(p["samples"] for p in info["partitions"].values()) == len(ds)
    assert set(history) == {"train_loss", "val_loss"}


def test_sparse_date_ids_use_observed_dates_and_never_split_a_date(tmp_path):
    ds = _dataset()
    ds.dates = ds.dates * 5 + 100
    trainer = Trainer(torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0)),
                      torch.nn.MSELoss(), TrainConfig(epochs=1, val_split=.25, purge_dates=2,
                                                      save_dir=str(tmp_path)))
    trainer.fit(ds)
    info = trainer.split_info["partitions"]
    assert info["train"]["date_range"] == [100, 140]
    assert info["purged"] == {"samples": 6, "observed_dates": 2, "date_range": [145, 150]}
    assert info["validation"]["date_range"] == [155, 165]


@pytest.mark.parametrize("overrides,match", [
    ({"purge_dates": -1}, "nonnegative integer"), ({"purge_dates": 1.5}, "nonnegative integer"),
    ({"purge_dates": True}, "nonnegative integer"), ({"purge_dates": 20}, "no training dates"),
    ({"purge_dates": 1, "val_split_mode": "random"}, "requires val_split_mode"),
])
def test_invalid_gap_preserves_existing_checkpoint_and_creates_no_report(tmp_path, overrides, match):
    old = tmp_path / "last.pt"
    old.write_bytes(b"existing checkpoint")
    trainer = Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(),
                      TrainConfig(epochs=1, save_dir=str(tmp_path), **overrides))
    with pytest.raises(ValueError, match=match):
        trainer.fit(_dataset())
    assert old.read_bytes() == b"existing checkpoint"
    assert not (tmp_path / "split.json").exists()
