from src.training.optimizers.optimizer_registry import OPTIMIZER_REGISTRY
from src.training.optimizers.configs.optimizer import OptimizerConfig
from src.training.optimizers.optimizer import Optimizer
from src.models.architectures.model import Model
import torch.nn as nn

def create_optimizer(config: OptimizerConfig, model:Model) -> Optimizer:
    """
    Factory method that takes a ModelConfig and returns the corresponding model instance.
    """
    optimizer_cls = OPTIMIZER_REGISTRY.get(config.optimizer_type)
    if optimizer_cls is None:
        raise ValueError(f"Optimizer type '{config.optimizer_type}' is not registered.")

    return optimizer_cls(config, model)
