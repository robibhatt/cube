# models/architectures/activations.py
import torch.nn as nn


class Quadratic(nn.Module):
    """Simple elementwise quadratic activation ``f(x) = x^2``."""

    def forward(self, x):  # pragma: no cover - simple wrapper
        return x * x


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "quadratic": Quadratic,
}
