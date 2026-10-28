"""Batch complete daily cross-sections without mixing their correlation targets."""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


def date_boundaries(dates: torch.Tensor) -> list[int]:
    """Start/end offsets for finite, date-ordered samples, including the sentinel."""
    if (dates.ndim != 1 or not torch.isfinite(dates).all()
            or (dates[1:] < dates[:-1]).any()):
        raise ValueError("sample dates must be a finite nondecreasing vector")
    if not dates.numel():
        return [0]
    starts = (dates[1:] != dates[:-1]).nonzero().flatten().add(1).tolist()
    return [0, *starts, dates.numel()]


class CrossSectionBatchSampler(Sampler[list[int]]):
    """One full observed date per batch; optionally shuffle the order of dates.

    Missing stocks stay missing. No padding, cross-date mixing, within-day
    splitting or dropped small batches. Memory for boundaries grows with dates.
    """

    def __init__(self, dates: torch.Tensor, *, shuffle: bool = False) -> None:
        self.boundaries = date_boundaries(dates)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.boundaries) - 1

    def __iter__(self):
        order = torch.randperm(len(self)).tolist() if self.shuffle else range(len(self))
        for day in order:
            yield list(range(self.boundaries[day], self.boundaries[day + 1]))
