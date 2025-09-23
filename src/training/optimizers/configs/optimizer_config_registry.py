OPTIMIZER_CONFIG_REGISTRY = {}


def register_optimizer_config(name: str):
    """Class decorator to register an ``OptimizerConfig`` subclass."""
    def decorator(cls):
        OPTIMIZER_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator


def build_optimizer_config(name: str, **kwargs):
    """Instantiate a registered ``OptimizerConfig`` by ``name``."""
    cfg_cls = OPTIMIZER_CONFIG_REGISTRY.get(name)
    if cfg_cls is None:
        raise ValueError(f"Optimizer config '{name}' is not registered.")
    # Inject optimizer_type so dataclasses_json can decode nested configs
    return cfg_cls.from_dict({"optimizer_type": name, **kwargs})


def build_optimizer_config_from_dict(data: dict):
    """Construct an ``OptimizerConfig`` from a dictionary."""
    data = dict(data)
    name = data.pop("optimizer_type", None)
    if name is None:
        raise ValueError("Missing 'optimizer_type' key in optimizer config dictionary.")
    return build_optimizer_config(name, **data)
