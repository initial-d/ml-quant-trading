"""Panel-data loaders + synthetic generator.

The package treats market data as a pair of ``[date, stock]`` torch
tensors plus a boolean mask telling us which cells are tradable. All
downstream code (factors, training, portfolio) operates on this layout.
"""
from .panel import Panel
from .synthetic import make_synthetic_panel

__all__ = ["Panel", "make_synthetic_panel"]
