"""Tiny YAML-config loader.

Why not Hydra? — Hydra is great but pulls 6 MB of deps for a single
``yaml.safe_load`` and a few path joins. Forty lines of stdlib do
exactly what we need without taking a dependency on a config framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """A nested-dict config with attribute access.

    Example
    -------
    >>> c = Config({"data": {"n_stocks": 200}, "seed": 42})
    >>> c.data.n_stocks
    200
    >>> c.seed
    42
    """

    _data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:                # noqa: D401
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            value = self._data[item]
        except KeyError as exc:                             # pragma: no cover
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, item: str) -> Any:
        value = self._data[item]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level of {path} must be a mapping, got {type(data).__name__}")
    return Config(data)
