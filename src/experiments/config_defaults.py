"""Shared experiment configuration defaults and helpers."""

from __future__ import annotations

from typing import Any, TypeVar

DEFAULT_MSE_THRESHOLD: float = 0.01
DEFAULT_MSE_SAMPLES: int = 8192
DEFAULT_ANCESTOR_THRESHOLD: int = 2

_T = TypeVar("_T")
_MISSING = object()


def ensure_config_value(config: Any, field_name: str, default: _T) -> _T:
    """Ensure ``config`` exposes ``field_name`` with a non-``None`` value."""

    value = getattr(config, field_name, _MISSING)
    if value is _MISSING or value is None:
        setattr(config, field_name, default)
        return default
    return value

