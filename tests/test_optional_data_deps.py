"""Provider packages are optional; a missing one must explain how to install it."""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from mlquant.data._optional import require


def test_require_returns_installed_module():
    assert require("json") is importlib.import_module("json")


def test_require_points_to_data_extra_when_missing():
    with pytest.raises(ImportError, match=r"mlquantx\[data\]"):
        require("definitely_not_an_installed_provider")


def test_package_imports_without_provider_packages(monkeypatch):
    blocked = {"yfinance", "baostock", "akshare"}
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod.split(".")[0] in blocked or mod.startswith("mlquant"):
            monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", guarded)

    data = importlib.import_module("mlquant.data")
    with pytest.raises(ImportError, match=r"mlquantx\[data\]"):
        data.load_yfinance_panel(["AAPL"], "2024-01-01", "2024-02-01")
