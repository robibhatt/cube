import os
import sys
import pytest
import torch
from dataclasses import dataclass
from src.models.targets.target_function import TargetFunction
from src.models.targets.target_function_registry import register_target_function
from src.models.targets.configs.target_function_config_registry import (
    register_target_function_config,
)
from src.models.targets.configs.base import TargetFunctionConfig

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


@register_target_function_config("LinearTargetFunction")
@dataclass(kw_only=True)
class LinearTargetFunctionConfig(TargetFunctionConfig):
    input_shape: torch.Size

    def __post_init__(self) -> None:
        self.model_type = "LinearTargetFunction"
        self.output_shape = torch.Size([1])
        super().__post_init__()


@register_target_function("LinearTargetFunction")
class LinearTargetFunction(TargetFunction):
    """Σ-over-features linear map for cheap regression."""

    def __init__(self, config: LinearTargetFunctionConfig):
        super().__init__(config)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        sum_dims = tuple(range(1, x.dim()))
        return x.sum(dim=sum_dims, keepdim=True)

    def __str__(self) -> str:
        return f"LinearTargetFunction(sum over {tuple(self.input_shape)})"


@dataclass(kw_only=True)
class QuadraticTargetFunctionConfig(TargetFunctionConfig):
    input_shape: torch.Size

    def __post_init__(self) -> None:
        self.model_type = "QuadraticTargetFunction"
        self.output_shape = torch.Size([1])
        super().__post_init__()


@register_target_function("QuadraticTargetFunction")
class QuadraticTargetFunction(TargetFunction):
    """Sum of squares over features."""

    def __init__(self, config: QuadraticTargetFunctionConfig):
        super().__init__(config)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.pow(2)
        sum_dims = tuple(range(1, x_sq.dim()))
        return x_sq.sum(dim=sum_dims, keepdim=True)

    def __str__(self) -> str:
        return f"QuadraticTargetFunction(sum of squares over {tuple(self.input_shape)})"


@pytest.fixture
def linear_function(input_shape):
    cfg = LinearTargetFunctionConfig(input_shape=input_shape)
    return LinearTargetFunction(cfg)


@pytest.fixture
def quadratic_function(input_shape):
    cfg = QuadraticTargetFunctionConfig(input_shape=input_shape)
    return QuadraticTargetFunction(cfg)
