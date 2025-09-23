REPRESENTOR_REGISTRY: dict[str, type] = {}

def register_representor(model_type: str):
    """Class decorator to register a ModelRepresentor subclass."""
    def decorator(cls):
        REPRESENTOR_REGISTRY[model_type] = cls
        return cls
    return decorator
