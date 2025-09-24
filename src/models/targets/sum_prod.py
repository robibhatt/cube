import torch
from torch import Tensor

from src.models.targets.target_function import TargetFunction
from src.models.targets.target_function_registry import register_target_function
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@register_target_function("SumProdTarget")
class SumProdTarget(TargetFunction):
    """Compute a weighted sum of product terms over selected coordinates."""

    def __init__(self, config: SumProdTargetConfig):
        super().__init__(config)
        self.indices_list = [list(group) for group in config.indices_list]
        self.weights = [float(weight) for weight in config.weights]
        if len(self.indices_list) != len(self.weights):
            raise ValueError("weights length must match indices_list length")
        self.max_index = max(max(group) for group in self.indices_list)
        weights_tensor = torch.tensor(self.weights, dtype=torch.float32)
        self.register_buffer("weights_tensor", weights_tensor)
        self.scale = self._compute_normalization()

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        if flat.shape[-1] <= self.max_index:
            raise ValueError(
                "SumProdTarget expects at least "
                f"{self.max_index + 1} elements in input_shape, but got {flat.shape[-1]}"
            )
        weights = self.weights_tensor.to(flat)
        total = torch.zeros_like(flat[..., 0])
        for weight, group in zip(weights.unbind(), self.indices_list):
            total = total + weight * flat[..., group].prod(dim=-1)
        total = total * self.scale
        return total.unsqueeze(-1)

    def _compute_normalization(self) -> float:
        num_samples = 2048
        g = torch.Generator()
        g.manual_seed(0)
        X = torch.randint(0, 2, (num_samples, *self.input_shape), generator=g)
        X = X.to(torch.float32) * 2 - 1
        flat = X.view(num_samples, -1)
        total = torch.zeros(num_samples, dtype=flat.dtype)
        for weight, group in zip(self.weights_tensor.unbind(), self.indices_list):
            total = total + weight * flat[:, group].prod(dim=-1)
        var = total.var(unbiased=False)
        if var > 0:
            return float(1.0 / torch.sqrt(var))
        else:
            return 1.0

    def __str__(self) -> str:
        return (
            "SumProdTarget("
            f"input_shape={tuple(self.input_shape)}, "
            f"indices_list={self.indices_list}, "
            f"weights={self.weights})"
        )
