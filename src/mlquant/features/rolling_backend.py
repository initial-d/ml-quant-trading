"""Scoped selection of the four rolling moment implementations.

The default is the historical direct-window (``unfold``) implementation.
``linear`` uses centred FP64 block summaries on CPU/CUDA. It excludes nonfinite
observations and clamps correlations, so nonlinear factor ranks can change;
it is an explicit numerical choice, not a bit-identical replacement.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar


_BACKEND: ContextVar[str] = ContextVar("mlquant_rolling_backend", default="unfold")


def validate_rolling_backend(backend: str) -> str:
    if not isinstance(backend, str) or backend not in ("unfold", "linear"):
        raise ValueError(f"rolling_backend must be 'unfold' or 'linear', got {backend!r}")
    return backend


def get_rolling_backend() -> str:
    return _BACKEND.get()


@contextmanager
def using_rolling_backend(backend: str):
    """Select moments for direct primitive/registered-factor calls in this scope.

    Nested scopes restore their caller's selection, including on exceptions.
    Selection is local to the current thread/async context. Factor-set APIs
    select their own explicit backend (default ``unfold``).
    """
    token = _BACKEND.set(validate_rolling_backend(backend))
    try:
        yield
    finally:
        _BACKEND.reset(token)
