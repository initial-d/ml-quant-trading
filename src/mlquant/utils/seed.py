"""Deterministic seeding across numpy, python, and (optionally) torch."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Seed all RNGs the project touches.

    Parameters
    ----------
    seed : int
        Master seed. Each library is seeded with the same value.
    deterministic_torch : bool
        If True and torch is importable, set deterministic cuDNN flags so
        that runs are bit-reproducible on the same hardware. Marginally
        slower; turn off for production training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
