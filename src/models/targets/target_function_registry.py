TARGET_FUNCTION_REGISTRY: dict[str, type] = {}

def register_target_function(target_function_type: str):
    """Class decorator to register a TargetFunction subclass."""
    def decorator(cls):
        TARGET_FUNCTION_REGISTRY[target_function_type] = cls
        return cls
    return decorator
