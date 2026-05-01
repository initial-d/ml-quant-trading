"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    seed_everything(123)


@pytest.fixture(scope="session")
def small_panel():
    return make_synthetic_panel(SyntheticConfig(n_stocks=30, n_dates=120, seed=7))


@pytest.fixture
def small_returns(small_panel):
    return small_panel.returns


@pytest.fixture
def random_panel_tensors():
    rng = np.random.default_rng(0)
    T, N = 50, 20
    x = torch.tensor(rng.normal(size=(T, N)), dtype=torch.float32)
    mask = torch.tensor(rng.random((T, N)) > 0.05)
    return x, mask
