"""Import helper for optional market-data provider packages."""
from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, package: str | None = None) -> ModuleType:
    """Import ``module`` or explain how to install the ``[data]`` extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{package or module} is required for this data loader. "
            "Install the public-data extra with: python -m pip install 'mlquantx[data]'"
        ) from exc
