"""Backward-compatible Alpha101 interface.

The 9 curated Alpha101 factors are now registered in the unified
:data:`LEGACY_REGISTRY` (via ``_factors_alpha101.py``) alongside all
other factor families. This module provides the legacy
``compute_alpha_set`` / ``ALPHA_REGISTRY`` API for backward
compatibility.

The canonical way to compute all 213 factors is::

    from mlquant.features import compute_legacy_set
    factors, mask, names = compute_legacy_set(panel)

To compute only the Alpha101 subset::

    from mlquant.features import compute_legacy_set
    alpha_names = tuple(n for n in LEGACY_REGISTRY if n.startswith("alpha_"))
    factors, mask, names = compute_legacy_set(panel, names=alpha_names)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..data.panel import Panel
from .tensor_factors import cs_zscore
from .legacy_factors import LEGACY_REGISTRY

Tensor = torch.Tensor

# Expose the alpha_* subset as ALPHA_REGISTRY for backward compat
ALPHA_REGISTRY: Dict[str, object] = {}


def _populate_alpha_registry() -> None:
    """Lazily populate ALPHA_REGISTRY from LEGACY_REGISTRY alpha_* entries."""
    if ALPHA_REGISTRY:
        return
    for name, fn in LEGACY_REGISTRY.items():
        if name.startswith("alpha_"):
            # Map back to old naming convention (alpha_001 -> alpha001)
            old_name = name.replace("alpha_", "alpha")
            ALPHA_REGISTRY[old_name] = fn


def compute_alpha_set(
    panel: Panel,
    *,
    names: tuple[str, ...] | None = None,
    neutralize: bool = True,
) -> Tuple[Tensor, Tensor, list[str]]:
    """Compute the Alpha101 subset → tensor of shape ``[T, N, F]``.

    This is a backward-compatible wrapper. Prefer
    :func:`~mlquant.features.legacy_factors.compute_legacy_set` for new
    code.

    Parameters
    ----------
    panel : Panel
    names : tuple of str, optional
        Subset of alpha names (old style: "alpha001", "alpha002", ...).
        ``None`` → all 9 registered alphas.
    neutralize : bool
        If True, each factor is cross-sectionally z-scored.

    Returns
    -------
    factors : Tensor [T, N, F]
    mask    : Tensor [T, N]
    names   : list[str]
    """
    _populate_alpha_registry()
    names = names or tuple(ALPHA_REGISTRY)
    cols, joint_mask = [], None
    for n in names:
        v, m = ALPHA_REGISTRY[n](panel)
        if neutralize:
            v, _ = cs_zscore(v, m)
        cols.append(v)
        joint_mask = m if joint_mask is None else joint_mask & m
    factors = torch.stack(cols, dim=-1)
    return factors, joint_mask, list(names)
