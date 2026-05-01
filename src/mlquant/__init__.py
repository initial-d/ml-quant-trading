"""mlquant — reference implementation of the ML-quant-trading paper.

Public sub-packages:
    data        — panel loaders and synthetic data generators
    features    — tensor-accelerated factor engine and alpha library
    training    — datasets, GBM augmentation, training loop
    models      — neural networks and sign-aware losses
    portfolio   — cross-sectional Markowitz with α-sweep
    backtest    — vectorised backtest engine and metrics
    cli         — Click entry points used by `mlquant ...`

The package is import-light: `import mlquant` does **not** transitively
import torch or cvxpy. Sub-packages are loaded lazily on demand.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
