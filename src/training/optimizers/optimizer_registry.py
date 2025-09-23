OPTIMIZER_REGISTRY = {}

def register_optimizer(optimizer_type: str):
    def decorator(cls):
        OPTIMIZER_REGISTRY[optimizer_type] = cls
        return cls
    return decorator