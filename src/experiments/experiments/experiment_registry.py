EXPERIMENT_REGISTRY: dict[str, type] = {}


def register_experiment(name: str):
    """Class decorator to register an :class:`Experiment` subclass."""

    def decorator(cls):
        EXPERIMENT_REGISTRY[name] = cls
        return cls

    return decorator
