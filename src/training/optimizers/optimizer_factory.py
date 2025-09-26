from src.training.optimizers.optimizer_registry import OPTIMIZER_REGISTRY
from src.training.optimizers.configs.optimizer import OptimizerConfig
from src.training.optimizers.optimizer import Optimizer
from src.models.architectures.mlp import MLP

def create_optimizer(config: OptimizerConfig, model: MLP) -> Optimizer:
    """Return an optimizer instance bound to ``model`` using ``config``."""
    optimizer_cls = OPTIMIZER_REGISTRY.get(config.optimizer_type)
    if optimizer_cls is None:
        raise ValueError(f"Optimizer type '{config.optimizer_type}' is not registered.")

    return optimizer_cls(config, model)
