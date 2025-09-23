from __future__ import annotations

from .base import ModelConfig

MODEL_CONFIG_REGISTRY: dict[str, type[ModelConfig]] = {}


def register_model_config(name: str):
    """Class decorator to register a ``ModelConfig`` subclass."""

    def decorator(cls: type[ModelConfig]):
        MODEL_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def build_model_config(name: str, **kwargs) -> ModelConfig:
    """Instantiate a registered ``ModelConfig`` by ``name`` with ``kwargs``."""
    cfg_cls = MODEL_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Model config '{name}' is not registered.")
    return cfg_cls(**kwargs)


def build_model_config_from_json_args(name: str, **kwargs) -> ModelConfig:
    """Construct a model config from ``name`` and ``kwargs``."""
    cfg_cls = MODEL_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Model config '{name}' is not registered.")
    # Inject the model_type key so that from_dict knows which subclass to create
    return cfg_cls.from_dict({"model_type": name, **kwargs})


def build_model_config_from_dict(data: dict) -> ModelConfig:
    """Construct a model config from a dictionary (e.g., parsed JSON)."""
    data = dict(data)  # make a shallow copy to avoid mutating the caller's dict
    name = data.pop("model_type", None)
    if name is None:
        raise ValueError("Missing 'model_type' key in model config dictionary.")
    return build_model_config_from_json_args(name, **data)

