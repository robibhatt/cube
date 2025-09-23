from dataclasses import dataclass, field
from typing import Union, Mapping, Sequence
from dataclasses_json import dataclass_json
import torch
import torch.nn as nn

# Define a JSON-compatible value type
Json = Union[str, int, float, bool, None, Mapping[str, "Json"], Sequence["Json"]]


class _RegressionEvaluator(nn.Module):
    """
    Wraps an nn.Module loss and returns its scalar value (e.g., MSE).
    Forward signature matches a loss: forward(input, target) -> scalar.
    """
    def __init__(self, loss_module: nn.Module):
        super().__init__()
        self.loss = loss_module

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(input, target)


class _BinaryPM1ErrorEvaluator(nn.Module):
    """
    Classification error for ±1 labels.
    Interprets 'input' as logits/margins; predicts +1 if logit >= 0 else -1.
    Targets are coerced to {-1, +1} by thresholding at 0.
    """
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold  # logit threshold (0.0 == prob 0.5)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # predictions in {-1, +1}
        pred_pm1 = torch.where(input >= self.threshold, torch.ones_like(input), -torch.ones_like(input))
        # robustly coerce targets into {-1, +1}
        targ_pm1 = torch.where(target >= 0, torch.ones_like(target), -torch.ones_like(target))
        # elementwise mismatch -> mean over all elements
        incorrect = (pred_pm1 != targ_pm1).to(input.dtype)
        return incorrect.mean()


class _Binary01ErrorEvaluator(nn.Module):
    """
    Classification error for {0,1} labels.
    Interprets 'input' as logits; predicts 1 if logit >= 0 else 0.
    Targets are coerced to {0,1} by thresholding at 0.5.
    """
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold  # logit threshold (0.0 == prob 0.5)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # predictions in {0,1}
        pred01 = (input >= self.threshold).to(input.dtype)
        # robustly coerce targets into {0,1}
        targ01 = (target > 0.5).to(input.dtype)
        incorrect = (pred01 != targ01).to(input.dtype)
        return incorrect.mean()


@dataclass_json
@dataclass
class LossConfig:
    """
    A JSON-serializable configuration for PyTorch loss functions.

    Attributes:
        name: The name of the loss *class* in ``torch.nn``.
        params: A mapping of initialization parameters, restricted to JSON-compatible types.
        eval_type: Evaluation behavior. One of:
            - "regression": return the underlying loss value (e.g., MSE)
            - "binary_pm1": 0/1 error with logits vs. targets in {-1,+1}
            - "binary_01":  0/1 error with logits vs. targets in {0,1}
        Aliases accepted:
            - "pm1_binary" -> "binary_pm1"
            - "01_binary"  -> "binary_01"
    """
    name: str
    params: Mapping[str, Json] = field(default_factory=dict)
    eval_type: str = "regression"

    _EVAL_ALIASES = {
        "pm1_binary": "binary_pm1",
        "01_binary": "binary_01",
    }
    _EVAL_CHOICES = {"regression", "binary_pm1", "binary_01"}

    def _normalize_eval_type(self) -> str:
        et = self.eval_type
        et = self._EVAL_ALIASES.get(et, et)
        if et not in self._EVAL_CHOICES:
            raise ValueError(
                f"Invalid eval_type='{self.eval_type}'. "
                f"Choose one of {sorted(self._EVAL_CHOICES)} or aliases {sorted(self._EVAL_ALIASES.keys())}."
            )
        return et

    def build(self) -> nn.Module:
        """
        Instantiate and return the actual loss function.

        Only class-based losses from ``torch.nn`` are supported. If ``self.name``
        is not present as a class in ``torch.nn`` a ``ValueError`` is raised.
        """
        if hasattr(nn, self.name):
            LossClass = getattr(nn, self.name)
            try:
                return LossClass(**self.params)
            except TypeError as e:
                raise TypeError(f"Invalid params for loss '{self.name}': {e}") from e

        raise ValueError(f"Loss '{self.name}' must be a class in torch.nn")

    def get_evaluator(self) -> nn.Module:
        """
        Return an nn.Module that behaves like a loss (forward(input, target) -> scalar),
        but computes an evaluation metric based on eval_type:

        - 'regression': returns the raw loss value using the configured loss.
        - 'binary_pm1': returns classification error rate with logits vs. targets in {-1,+1}.
        - 'binary_01': returns classification error rate with logits vs. targets in {0,1}.

        Notes:
        - For the binary modes, inputs are interpreted as *logits* with a 0-threshold
          (equivalent to probability 0.5 after sigmoid). If you pass probabilities
          instead of logits, wrap or modify the evaluator to use threshold=0.5.
        """
        et = self._normalize_eval_type()
        if et == "regression":
            return _RegressionEvaluator(self.build())
        if et == "binary_pm1":
            return _BinaryPM1ErrorEvaluator(threshold=0.0)
        if et == "binary_01":
            return _Binary01ErrorEvaluator(threshold=0.0)

        # Should never hit due to normalization check.
        raise RuntimeError(f"Unhandled eval_type: {et}")
