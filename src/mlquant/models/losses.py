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

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = (pred - target) ** 2
        same_sign = (pred * target) > 0
        weight = torch.where(same_sign,
                             torch.full_like(squared, self.gamma),
                             torch.full_like(squared, 1.0 + self.gamma))
        return (squared * weight).mean()


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape or not x.numel():
        raise ValueError("correlation inputs must have the same nonempty shape")
    xm = x - x.mean()
    ym = y - y.mean()
    xss, yss = xm.pow(2).sum(), ym.pow(2).sum()
    defined = (xss > 0) & (yss > 0)
    # sqrt'(0) is infinite: clamping the product after sqrt cannot protect
    # backward. Replace zero-variance operands before taking either root.
    sx = torch.where(defined, xss, torch.ones_like(xss)).sqrt()
    sy = torch.where(defined, yss, torch.ones_like(yss)).sqrt()
    denom = (sx * sy).clamp_min(1e-12)
    numerator = (xm * ym).sum()
    return torch.where(defined, numerator / denom, numerator * 0.0)


class ICLoss(nn.Module):
    """Negative Pearson IC across a batch (e.g. one cross-section).

    A constant prediction or label batch has undefined correlation and returns
    zero loss/gradients. Pass aligned, finite tensors from one cross-section.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -_pearson(pred, target)


class RankICLoss(nn.Module):
    """Negative Spearman IC across a batch.

    Spearman = Pearson on ranks. We use the *soft-rank* trick (a
    differentiable approximation that sorts the inputs through a
    softmax-temperature lens) so the loss is gradient-friendly.

    Above ``chunk_size`` stocks, compute columns in blocks and checkpoint their
    activations. Pairwise work remains quadratic; temporary matrices scale as
    N * chunk_size. ``chunk_size=None`` restores the dense reference path.
    """

    def __init__(self, temperature: float = 1.0, *, chunk_size: int | None = 256) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer or None")
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape or not pred.numel():
            raise ValueError("correlation inputs must have the same nonempty shape")
        if pred.ndim != 1:
            raise ValueError("RankIC inputs must be one-dimensional cross-sections")
        return -_pearson(self._soft_rank(pred), self._soft_rank(target))

    @staticmethod
    def _rank_columns(full, columns, temperature):
        return torch.sigmoid((columns.unsqueeze(0) - full.unsqueeze(1)) / temperature).sum(0)

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is None or x.numel() <= self.chunk_size:
            return self._rank_columns(x, x, self.temperature)
        ranks = []
        for columns in x.split(self.chunk_size):
            if torch.is_grad_enabled() and x.requires_grad:
                # Non-reentrant checkpointing preserves autograd.grad and higher
                # derivatives while recomputing one block during backward.
                rank = checkpoint(self._rank_columns, x, columns, self.temperature,
                                  use_reentrant=False, preserve_rng_state=False)
            else:
                rank = self._rank_columns(x, columns, self.temperature)
            ranks.append(rank)
        return torch.cat(ranks)
