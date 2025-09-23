from __future__ import annotations

"""Representor for the :class:`MLP` architecture.

This class exposes *all* intermediate tensors produced by an ``MLP``
instance and can construct ``ModelConfig`` objects that describe sub‑networks
between any two representations (when that mapping can be expressed with a
plain feed‑forward stack).

Representation index layout
---------------------------
Given ``n = len(hidden_dims)`` hidden layers, the representation indices are
arranged as::

    0                                   → input vector (dim = input_dim)
    [1] optional start activation
    1/2 = 2 * 0 + (1/2)                 → pre/post‑activation of hidden layer 0
    ...
    2n‑1                                → pre‑activation of hidden layer n‑1
    2n                                  → post‑activation of hidden layer n‑1
    2n+1                                → final output (dim = output_dim)
    [last] optional end activation

Total representations therefore depend on the ``start_activation`` and
``end_activation`` flags.

Notes
-----
* ``forward_config`` currently supports **input → output** only.  More
  granular slicing would require an explicit *partial‑MLP* config that is not
  yet defined in the code‑base.  Requests for unsupported ranges raise
  ``NotImplementedError``.
"""

from typing import List, Any
from pathlib import Path

import torch
import torch.nn as nn

from src.models.representors.model_representor import ModelRepresentor
from src.models.architectures.configs.mlp import MLPConfig
from src.models.architectures.configs.base import ModelConfig
from src.models.representors.representor_registry import register_representor

__all__ = ["MLPRepresentor"]


@register_representor("MLP")
class MLPRepresentor(ModelRepresentor):
    """Expose and manipulate internal representations of an :class:`MLP`."""

    def __init__(self, model_config: ModelConfig, checkpoint_dir: Path, device: torch.device):  # type: ignore[override]
        super().__init__(model_config, checkpoint_dir, device=device)
        if not isinstance(self.model_config, MLPConfig):
            raise TypeError(
                "MLPRepresentor expects an MLPConfig, got "
                f"{type(self.model_config).__name__}"
            )

    def get_modules(self) -> List[nn.Module]:
        """Return the modules of the underlying :class:`MLP`.
        
        The returned modules may be standard ``nn.Linear`` layers or μP layers
        such as :class:`mup.MuLinear`/``MuReadout`` when the ``MLP`` was
        constructed with ``mup=True`` in its config.  Callers should therefore
        avoid assumptions specific to ``nn.Linear``.
        """
        return self.model.layers

    def _slice_mlp(self, cfg: MLPConfig, i: int, j: int) -> MLPConfig:
        start_offset = 1 if cfg.start_activation else 0
        input_dim_index = (i+1 - start_offset) // 2
        start_active = (start_offset + i + 1 ) % 2 == 0
        offset = 1 if start_active else 0
        output_dim_index = input_dim_index + (j-i + 1 - offset)//2
        end_active = (start_offset + j) % 2 == 0
        if start_active and j == (i+1):
            assert False, "you can't make an MLP that is JUST an activation"
        input_dim = cfg.input_dim if input_dim_index == 0 else cfg.hidden_dims[input_dim_index - 1]
        hidden_dims = cfg.hidden_dims[input_dim_index:output_dim_index-1]
        output_dim = cfg.output_dim if output_dim_index == (len(cfg.hidden_dims) + 1) else cfg.hidden_dims[output_dim_index - 1]

        return MLPConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=cfg.activation,
            start_activation=start_active,
            end_activation=end_active,
            bias=cfg.bias,
            mup=cfg.mup,
        )

    def forward_config(self, from_rep:int, to_rep:int) -> ModelConfig:
        assert to_rep <= len(self.modules), "to_rep must at most the number of modules"
        assert from_rep < to_rep, "from_rep must be less than to_rep"
        rep_config = self._slice_mlp(self.model_config, from_rep, to_rep)
        return rep_config

    def to_representation_dict(self, rep_id: int) -> dict[str, Any]:
        """Return a dictionary of representation information."""
        start_offset = 1 if self.model_config.start_activation else 0
        layer_index = (rep_id+1 - start_offset) // 2
        start_active = (start_offset + rep_id + 1 ) % 2 == 0
        # by convention, the input layer of the network (if start_activation is False) is considered post_activation
        post_activation = not start_active
        return {"layer_index": layer_index, "post_activation": post_activation}

    def from_representation_dict(self, rep_dict: dict[str, Any]) -> int:
        """Get the id from a representation information dictionary"""
        start_offset = 1 if self.model_config.start_activation else 0
        layer_index = rep_dict["layer_index"]
        post_activation = rep_dict["post_activation"]
        number_of_operations = 2 * layer_index - 1 + start_offset + (1 if post_activation else 0)
        return number_of_operations

    def representation_shape(self, rep_id: int) -> torch.Size:
        start_offset = 1 if self.model_config.start_activation else 0
        layer_index = (rep_id + 1 - start_offset) // 2
        if layer_index == 0:
            dim = self.model_config.input_dim
        elif layer_index > len(self.model_config.hidden_dims):
            dim = self.model_config.output_dim
        else:
            dim = self.model_config.hidden_dims[layer_index - 1]
        return torch.Size((dim,))

    def describe_rep_index(self, rep_id: int) -> str:
        n_hidden = len(self.model_config.hidden_dims)
        if rep_id == 0:
            return "input_0"
        if rep_id == len(self.modules):
            return f"output_{n_hidden+1}"
        layer = self.to_representation_dict(rep_id)["layer_index"]
        return f"hidden_{layer}"

    def get_final_rep_id(self) -> int:
        """Return the representation id of the final layer."""
        start_offset = 1 if self.model_config.start_activation else 0
        # Each hidden layer has 2 operations (linear + activation)
        # Add 1 for final linear layer
        # Add 1 more if there's a final activation
        num_ops = 2 * len(self.model_config.hidden_dims) + start_offset + 1
        if hasattr(self.model_config, "end_activation") and self.model_config.end_activation:
            num_ops += 1
        return num_ops

    def get_base_rep_ids(self) -> List[int]:
        """Return base representation ids for common usage.

        Assumes start_activation and end_activation are False. The returned
        list contains the input layer, each hidden layer's post-activation,
        and the output layer.
        """
        if self.model_config.start_activation or getattr(self.model_config, "end_activation", False):
            raise ValueError(
                "get_base_rep_ids expects start_activation and end_activation to be False"
            )
        n_hidden = len(self.model_config.hidden_dims)
        ids = [0]
        ids.extend(2 * (i + 1) for i in range(n_hidden))
        ids.append(2 * n_hidden + 1)
        return ids

