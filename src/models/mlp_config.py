from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Any

from dataclasses_json import dataclass_json, config
from marshmallow import fields


@dataclass_json
@dataclass(kw_only=True)
class MLPConfig:
    """Configuration for :class:`~src.models.mlp.MLP`.

    The network now always exposes a single output dimension and omits optional
    start/end activation toggles.  ``output_dim`` is therefore modelled as a
    read-only property that always returns ``1``.
    """

    input_dim: int = field(metadata=config(mm_field=fields.Integer()))
    hidden_dims: List[int] = field(
        metadata=config(mm_field=fields.List(fields.Integer()))
    )
    exact_base_shapes: bool = field(
        default=False,
        metadata=config(mm_field=fields.Boolean(), exclude=lambda v: v is False),
    )

    @property
    def output_dim(self) -> int:
        """Return the fixed output dimension for μP MLPs."""

        return 1

    @classmethod
    def from_dict(cls, kvs: Mapping[str, Any]) -> "MLPConfig":  # type: ignore[override]
        """Create a config from a dictionary, validating deprecated keys.

        Previous configurations allowed ``output_dim``, ``start_activation`` and
        ``end_activation`` toggles.  These options have been removed; encountering
        them now results in a clear error so that users can update their configs.
        """

        data = dict(kvs)

        invalid_keys = {"output_dim", "start_activation", "end_activation"}
        present = sorted(invalid_keys.intersection(data.keys()))
        if present:
            joined = ", ".join(present)
            raise ValueError(
                "MLPConfig no longer accepts the following option(s): "
                f"{joined}. Please remove them from the configuration."
            )

        return cls.schema().load(data)

