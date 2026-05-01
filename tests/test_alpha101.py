"""Smoke tests for the Alpha101 subset."""
from __future__ import annotations

import torch

from mlquant.features.alpha101 import ALPHA_REGISTRY, compute_alpha_set


def test_every_alpha_runs(small_panel):
    for name, fn in ALPHA_REGISTRY.items():
        v, m = fn(small_panel)
        assert v.shape == small_panel.close.shape, f"{name}: shape mismatch"
        assert m.dtype == torch.bool, f"{name}: mask not bool"
        # Output must not contain NaN/Inf on tradable cells.
        finite = torch.isfinite(v) | ~m
        assert finite.all(), f"{name}: non-finite on tradable cell"


def test_compute_alpha_set_stacks(small_panel):
    factors, mask, names = compute_alpha_set(small_panel)
    assert factors.dim() == 3
    assert factors.shape[2] == len(names)
    assert mask.shape == small_panel.close.shape
