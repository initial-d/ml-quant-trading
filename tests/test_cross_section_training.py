"""Daily objectives must not become pooled cross-date correlations."""
import pytest
import torch
from torch.utils.data import DataLoader

from mlquant.models.losses import ICLoss, RankICLoss
from mlquant.training import CrossSectionBatchSampler, FactorDataset, TrainConfig, Trainer


def _ragged_dataset():
    factors = torch.zeros(5, 5, 2)
    mask = torch.zeros(5, 5, dtype=torch.bool)
    labels = torch.zeros(5, 5)
    # Intersection with the next day's legacy mask gives counts 2, 4, 1, 1.
    for day, size in enumerate([2, 5, 4, 1, 5]):
        factors[day, :, 0] = day
        factors[day, :, 1] = torch.arange(5)
        labels[day] = 100 * day + torch.arange(5)
        mask[day, :size] = True
    return FactorDataset(factors, mask, labels)


def test_sampler_keeps_complete_ragged_dates_and_all_samples():
    dates = torch.tensor([0, 0, 3, 3, 3, 9])
    assert list(CrossSectionBatchSampler(dates)) == [[0, 1], [2, 3, 4], [5]]
    shuffled = list(CrossSectionBatchSampler(dates, shuffle=True))
    assert sorted(i for batch in shuffled for i in batch) == list(range(6))
    assert all(dates[batch].unique().numel() == 1 for batch in shuffled)
    assert len(CrossSectionBatchSampler(torch.empty(0))) == 0
    for invalid in (torch.tensor([2, 1]), torch.tensor([0., float("nan")]), torch.ones(2, 1)):
        with pytest.raises(ValueError, match="nondecreasing vector"):
            CrossSectionBatchSampler(invalid)


@pytest.mark.parametrize("loss_type", [ICLoss, RankICLoss])
@pytest.mark.parametrize("split_mode", ["time", "random"])
def test_fit_automatically_uses_full_dates_for_correlation_losses(tmp_path, loss_type, split_mode):
    seen = {True: [], False: []}
    model = torch.nn.Linear(2, 1)

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = model

        def forward(self, x):
            return self.linear(x).squeeze(-1)

    class RecordLoss(loss_type):
        def forward(self, pred, target):
            days = target.floor_divide(100)
            assert days.unique().numel() == 1
            seen[model.training].append((int(days[0]), len(target)))
            assert torch.is_grad_enabled() == model.training
            return super().forward(pred, target)

    cfg = TrainConfig(epochs=1, batch_size=1, val_split=.5,
                      val_split_mode=split_mode, save_dir=str(tmp_path))
    history = Trainer(ScalarModel(), RecordLoss(), cfg).fit(_ragged_dataset())
    assert dict(seen[True] + seen[False]) == {0: 2, 1: 4, 2: 1, 3: 1}
    train_days, val_days = {d for d, _ in seen[True]}, {d for d, _ in seen[False]}
    assert not train_days & val_days
    if split_mode == "time":
        assert max(train_days) < min(val_days)
    assert all(torch.isfinite(torch.tensor(history[key])).all() for key in history)
    assert (tmp_path / "best.pt").exists()


def test_daily_ic_avoids_the_spurious_positive_pooled_correlation(tmp_path):
    # Each day has reversed stock ordering, but between-day level offsets make
    # pooled correlation positive. This fixture is an objective check, not alpha.
    x = torch.tensor([[0.], [1.], [2.], [100.], [101.], [102.]], dtype=torch.float32)
    y = torch.tensor([2., 1., 0., 102., 101., 100.])
    dates = torch.tensor([0, 0, 0, 1, 1, 1])
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    class Flatten(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = model

        def forward(self, values):
            return self.linear(values).squeeze(-1)

    trainer = Trainer(Flatten(), ICLoss(), TrainConfig(save_dir=str(tmp_path)))
    loader = DataLoader(torch.utils.data.TensorDataset(x, y),
                        batch_sampler=CrossSectionBatchSampler(dates))
    assert trainer._run_epoch(loader, train=False) == pytest.approx(1.)
    assert ICLoss()(x.flatten(), y).item() < -.99


def test_daily_validation_weights_dates_equally_and_disables_gradients(tmp_path):
    x = torch.zeros(6, 1)
    y = torch.tensor([1., 3., 3., 3., 3., 3.])
    model = torch.nn.Linear(1, 1, bias=False)
    loader = DataLoader(torch.utils.data.TensorDataset(x, y),
                        batch_sampler=CrossSectionBatchSampler(torch.tensor([0, 1, 1, 1, 1, 1])))
    trainer = Trainer(model, lambda p, t: ((p.flatten() - t) ** 2).mean(),
                      TrainConfig(save_dir=str(tmp_path)))
    assert trainer._run_epoch(loader, train=False) == pytest.approx(5.)


def test_small_regression_dataset_is_not_dropped(tmp_path):
    ds = _ragged_dataset()
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))
    before = model[0].weight.detach().clone()
    trainer = Trainer(model, torch.nn.MSELoss(),
                      TrainConfig(epochs=1, batch_size=1024, save_dir=str(tmp_path)))
    history = trainer.fit(ds)
    assert history["train_loss"][0] > 0
    assert not torch.equal(model[0].weight.detach(), before)


@pytest.mark.parametrize("overrides,match", [
    ({"val_split": 0}, "val_split"), ({"val_split": 1}, "val_split"),
    ({"val_split": float("nan")}, "val_split"), ({"batch_mode": "unknown"}, "batch_mode"),
    ({"batch_mode": "samples"}, "complete date"), ({"batch_size": 0}, "positive"),
])
def test_invalid_correlation_training_config_is_rejected(tmp_path, overrides, match):
    cfg = TrainConfig(save_dir=str(tmp_path), **overrides)
    trainer = Trainer(torch.nn.Linear(2, 1), ICLoss(), cfg)
    with pytest.raises(ValueError, match=match):
        trainer.fit(_ragged_dataset())
    assert not (tmp_path / "last.pt").exists()


def test_time_split_rejects_a_single_observed_date(tmp_path):
    ds = FactorDataset(torch.ones(2, 3, 1), torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3))
    trainer = Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), TrainConfig(save_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="two observed dates"):
        trainer.fit(ds)
