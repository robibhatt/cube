from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC
from dataclasses_json import dataclass_json, config

from src.utils.serialization_utils import encode_path, decode_path


@dataclass_json
@dataclass
class ExperimentConfig(ABC):
    """Base configuration for experiments."""

    experiment_type: str = field(init=False)
    home_directory: Path = field(
        metadata=config(encoder=encode_path, decoder=decode_path)
    )
    # Base seed used to derive all experiment-level RNG state
    seed: int
    # Whether to run trainers in parallel (not yet implemented)
    run_parallel: bool = field(default=False, kw_only=True)
