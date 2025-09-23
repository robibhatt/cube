import math
import torch
from torch import Tensor
from src.models.targets.target_function import TargetFunction
from src.models.targets.target_function_registry import register_target_function
from src.models.targets.configs.prod_1234 import Prod1234Config
from src.models.targets.configs.staircase import StaircaseTargetConfig
from src.models.targets.configs.prod_k import ProdKTargetConfig
from src.models.targets.configs.sum_prod import SumProdTargetConfig


@register_target_function("1234_prod")
class Prod1234(TargetFunction):
    def __init__(self, config: Prod1234Config):
        super().__init__(config)

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        if flat.shape[-1] < 4:
            raise ValueError(
                f"1234_prod expects at least four elements in input_shape, but got {flat.shape[-1]}"
            )
        prod12 = flat[..., 0] * flat[..., 1]
        prod34 = flat[..., 2] * flat[..., 3]
        return (prod12 + prod34).unsqueeze(-1)

    def __str__(self) -> str:
        return f"1234_prod(input_shape={tuple(self.input_shape)})"


@register_target_function("ProdKTarget")
class ProdKTarget(TargetFunction):
    """Compute the product of specified input coordinates.

    The indices of the coordinates to multiply are provided via the
    :class:`ProdKTargetConfig` ``indices`` field.  Indices use zero-based
    numbering with respect to the flattened input tensor.
    """

    def __init__(self, config: ProdKTargetConfig):
        super().__init__(config)
        self.indices = config.indices
        self.k = len(self.indices)

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        max_idx = max(self.indices)
        if flat.shape[-1] <= max_idx:
            raise ValueError(
                f"ProdKTarget expects at least {max_idx + 1} elements in input_shape, but got {flat.shape[-1]}"
            )
        prod = flat[..., self.indices].prod(dim=-1)
        return prod.unsqueeze(-1)

    def __str__(self) -> str:
        return f"ProdKTarget(input_shape={tuple(self.input_shape)}, indices={self.indices})"


@register_target_function("SumProdTarget")
class SumProdTarget(TargetFunction):
    """Compute the sum of multiple product terms over selected coordinates.

    ``indices_list`` specifies groups of indices; for each group the
    corresponding coordinates are multiplied together and the results are
    summed.
    """

    def __init__(self, config: SumProdTargetConfig):
        super().__init__(config)
        self.indices_list = [list(g) for g in config.indices_list]
        self.scale = self._compute_normalization()

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        max_idx = max(max(group) for group in self.indices_list)
        if flat.shape[-1] <= max_idx:
            raise ValueError(
                f"SumProdTarget expects at least {max_idx + 1} elements in input_shape, but got {flat.shape[-1]}"
            )
        total = torch.zeros_like(flat[..., 0])
        for group in self.indices_list:
            total = total + flat[..., group].prod(dim=-1)
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
        for group in self.indices_list:
            total = total + flat[:, group].prod(dim=-1)
        var = total.var(unbiased=False)
        if var > 0:
            return float(1.0 / torch.sqrt(var))
        else:
            return 1.0

    def __str__(self) -> str:
        return (
            f"SumProdTarget(input_shape={tuple(self.input_shape)}, indices_list={self.indices_list})"
        )

@register_target_function("StaircaseTarget")
class StaircaseTarget(TargetFunction):
    r"""Compute ``(x₁ + x₁x₂ + ... + x₁x₂⋯x_k) / \sqrt{k}`` for the first ``k`` coords."""

    def __init__(self, config: StaircaseTargetConfig):
        super().__init__(config)
        self.k = config.k
        # Precompute the 1/sqrt(k) scaling to avoid repeated square root ops
        self.scale = 1 / math.sqrt(self.k)

    def _forward(self, X: Tensor) -> Tensor:
        flat = X.reshape(*X.shape[:-len(self.input_shape)], -1)
        if flat.shape[-1] < self.k:
            raise ValueError(
                f"StaircaseTarget expects at least {self.k} elements in input_shape, but got {flat.shape[-1]}"
            )
        prod = flat[..., 0]
        y = prod.clone()
        for i in range(1, self.k):
            prod = prod * flat[..., i]
            y = y + prod
        y = y * self.scale
        return y.unsqueeze(-1)

    def __str__(self) -> str:
        return f"StaircaseTarget(input_shape={tuple(self.input_shape)}, k={self.k})"
