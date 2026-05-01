"""Sign-aware regression losses + IC / RankIC objectives.

Why not plain MSE?
    Trading P&L is sign-sensitive: a prediction with the wrong sign
    *and* moderate magnitude burns money, while a prediction with the
    right sign and any magnitude makes money. Plain MSE penalises both
    "right sign, big residual" and "wrong sign, small residual" the
    same way — that is well-documented to misalign with utility.

We expose:

* :class:`AdjMSELoss`  — Du (2025) §4.2: scale up the loss when
  ``sign(pred)·sign(label) < 0``, scale it down when they agree.
* :class:`ICLoss`      — minus Pearson correlation across the batch.
* :class:`RankICLoss`  — minus Spearman correlation across the batch.

ICLoss / RankICLoss are particularly useful for *cross-sectional*
training where what matters is the ordering of predictions within a
date, not their absolute magnitude.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdjMSELoss(nn.Module):
    r"""Sign-aware MSE.

    .. math::
        \mathcal{L}_{\text{adj}}(\hat y, y) =
        \begin{cases}
            \gamma\,(\hat y - y)^2 & \text{if }\operatorname{sign}(\hat y)\cdot\operatorname{sign}(y) > 0 \\
            (1+\gamma)\,(\hat y - y)^2 & \text{otherwise}
        \end{cases}

    With ``gamma = 0.1`` (default) wrong-sign predictions are penalised
    11× more than right-sign predictions of equal magnitude.
    """

    def __init__(self, gamma: float = 0.1) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = (pred - target) ** 2
        same_sign = (pred * target) > 0
        weight = torch.where(same_sign,
                             torch.full_like(squared, self.gamma),
                             torch.full_like(squared, 1.0 + self.gamma))
        return (squared * weight).mean()


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()).clamp_min(1e-12)
    return (xm * ym).sum() / denom


class ICLoss(nn.Module):
    """Negative Pearson IC across a batch (e.g. one cross-section)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -_pearson(pred, target)


class RankICLoss(nn.Module):
    """Negative Spearman IC across a batch.

    Spearman = Pearson on ranks. We use the *soft-rank* trick (a
    differentiable approximation that sorts the inputs through a
    softmax-temperature lens) so the loss is gradient-friendly.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -_pearson(self._soft_rank(pred), self._soft_rank(target))

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        # Differentiable rank: softmax over pairwise differences.
        diffs = (x.unsqueeze(0) - x.unsqueeze(1)) / self.temperature
        return torch.sigmoid(diffs).sum(dim=0)
