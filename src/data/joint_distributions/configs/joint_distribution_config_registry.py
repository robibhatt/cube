from __future__ import annotations

from .base import JointDistributionConfig

JOINT_DISTRIBUTION_CONFIG_REGISTRY: dict[str, type[JointDistributionConfig]] = {}


def register_joint_distribution_config(name: str):
    """Class decorator to register a ``JointDistributionConfig`` subclass."""

    def decorator(cls: type[JointDistributionConfig]):
        JOINT_DISTRIBUTION_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def build_joint_distribution_config(name: str, **kwargs) -> JointDistributionConfig:
    """Instantiate a registered ``JointDistributionConfig`` by ``name``."""
    cfg_cls = JOINT_DISTRIBUTION_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Joint distribution config '{name}' is not registered.")
    # inject the type name so dataclasses_json can decode nested configs correctly
    return cfg_cls.from_dict({"distribution_type": name, **kwargs})


def build_joint_distribution_config_from_dict(data: dict) -> JointDistributionConfig:
    """Construct a ``JointDistributionConfig`` from a dictionary."""
    data = dict(data)
    name = data.pop("distribution_type", None)
    if name is None:
        raise ValueError("Missing 'distribution_type' key in joint distribution config dictionary.")
    return build_joint_distribution_config(name, **data)
