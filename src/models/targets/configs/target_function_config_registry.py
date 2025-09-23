from __future__ import annotations

from .base import TargetFunctionConfig

TARGET_FUNCTION_CONFIG_REGISTRY: dict[str, type[TargetFunctionConfig]] = {}


def register_target_function_config(name: str):
    """Class decorator to register a ``TargetFunctionConfig`` subclass."""

    def decorator(cls: type[TargetFunctionConfig]):
        TARGET_FUNCTION_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def build_target_function_config(name: str, **kwargs) -> TargetFunctionConfig:
    """Construct a target function config from ``name`` and ``kwargs``."""
    cfg_cls = TARGET_FUNCTION_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Target function config '{name}' is not registered.")
    # use dataclasses_json for correct encoding/decoding
    return cfg_cls.from_dict({"model_type": name, **kwargs})


def build_target_function_config_from_dict(data: dict) -> TargetFunctionConfig:
    """Construct a config from a dictionary (e.g., parsed JSON)."""
    data = dict(data)
    name = data.pop("model_type", None)
    if name is None:
        raise ValueError("Missing 'model_type' key in target function config dictionary.")
    return build_target_function_config(name, **data)
