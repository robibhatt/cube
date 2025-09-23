from .target_function_registry import TARGET_FUNCTION_REGISTRY
from .configs.base import TargetFunctionConfig
from .target_function import TargetFunction


def create_target_function(config: TargetFunctionConfig) -> TargetFunction:
    """Instantiate a TargetFunction based on ``config.model_type``."""
    func_cls = TARGET_FUNCTION_REGISTRY.get(config.model_type)
    if func_cls is None:
        raise ValueError(
            f"TargetFunction type '{config.model_type}' is not registered."
        )
    return func_cls(config)
