"""Persistent Fourier statistics for :class:`~src.models.mlp.MLP` instances.

This module exposes :class:`FourierMlp`, a convenience wrapper that stores an
MLP alongside lazily-computed Fourier coefficients of its neuron activations.
The wrapper ensures that potentially expensive statistics are cached to disk so
that repeated analyses re-use previous results.  It now also records the second
moment of every neuron when the wrapper is constructed, enabling downstream
code to normalise Fourier coefficients by the variance of the corresponding
neurons without recomputing expectations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from src.checkpoints.checkpoint import Checkpoint
from src.fourier.fourier_mlp_module import FourierMlpModule
from src.models.mlp import MLP
from src.models.mlp_config import MLPConfig


class FourierMlp:
    """Manage cached Fourier coefficient computations for an :class:`MLP`.

    The class stores a serialized copy of the provided model alongside its
    configuration.  Fourier coefficients are computed lazily and cached to disk
    on a per-index basis.  Subsequent requests for the same Fourier indices
    reuse the cached results without recomputing them.
    """

    FOURIER_SUBDIR_NAME = "fourier_dir"
    CONFIG_FILENAME = "mlp_config.json"
    METADATA_FILENAME = "metadata.json"
    CHECKPOINT_SUBDIR = "checkpoint"
    SECOND_MOMENTS_FILENAME = "second_moments.pt"
    COEFFICIENT_SUBDIR = "coefficients"

    def __init__(
        self,
        mlp: MLP,
        directory: Path | str,
        *,
        sample_size: int = 65536,
        batch_size: int = 4096,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        """Create a new persistent Fourier analysis directory.

        Parameters
        ----------
        mlp:
            The model whose activations will be analysed.  A checkpoint of this
            model is stored to disk and re-used for subsequent computations.
        directory:
            Base directory under which the Fourier artefacts will be created.
        sample_size:
            Total number of Boolean-hypercube samples to use when estimating a
            Fourier coefficient.
        batch_size:
            Number of samples processed per forward pass while accumulating the
            coefficient estimate.
        dtype:
            Data type used for the generated Boolean samples.
        device:
            Device on which Fourier computations will be performed.
        """

        self.base_dir = Path(directory)
        self.fourier_dir = self.base_dir / self.FOURIER_SUBDIR_NAME
        if self.fourier_dir.exists():
            raise FileExistsError(
                f"Fourier directory already exists at {self.fourier_dir}"
            )

        self.fourier_dir.mkdir(parents=True, exist_ok=False)

        self.sample_size = self._validate_positive(sample_size, "sample_size")
        self.batch_size = self._validate_positive(batch_size, "batch_size")
        self.dtype = dtype
        self.device = torch.device(device)

        self.checkpoint_dir = self.fourier_dir / self.CHECKPOINT_SUBDIR
        self.coefficient_dir = self.fourier_dir / self.COEFFICIENT_SUBDIR
        self.coefficient_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.fourier_dir / self.CONFIG_FILENAME
        self.metadata_path = self.fourier_dir / self.METADATA_FILENAME
        self.second_moment_path = self.fourier_dir / self.SECOND_MOMENTS_FILENAME

        self._mlp_config = mlp.config
        self._mlp_cache: MLP | None = None
        self._second_moments_cache: dict[int, torch.Tensor] | None = None

        self._store_config()
        self._store_metadata()
        self._save_model_checkpoint(mlp)
        self._compute_and_store_second_moments()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _store_config(self) -> None:
        """Write the serialised MLP configuration to the analysis directory."""

        config_dict = self._mlp_config.to_dict() if hasattr(self._mlp_config, "to_dict") else asdict(self._mlp_config)  # type: ignore[arg-type]
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(config_dict, fh, indent=2)

    def _store_metadata(self) -> None:
        """Persist Fourier sampling hyperparameters for reproducibility."""

        metadata = {
            "sample_size": self.sample_size,
            "batch_size": self.batch_size,
            "dtype": self._dtype_to_string(self.dtype),
            "device": str(self.device),
        }
        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _save_model_checkpoint(self, mlp: MLP) -> None:
        """Snapshot the provided MLP so future computations use a stable copy."""

        checkpoint = Checkpoint(dir=self.checkpoint_dir)
        checkpoint.save(model=mlp, optimizer=None)

    def _compute_and_store_second_moments(self) -> None:
        """Estimate and persist the second moment of every neuron in the MLP.

        The estimation mirrors Fourier coefficient computation: Boolean samples
        are drawn uniformly from the hypercube, passed through the stored MLP
        and the squared activations are averaged.  Results are cached in
        ``second_moments.pt`` for rapid lookup via :meth:`get_neuron_second_moment`.
        """

        model = self._load_mlp().to(self.device)
        model.eval()

        module_sequence = list(model.net)
        capture_positions: list[int] = []
        for layer_idx, layer in enumerate(model.linear_layers):
            try:
                linear_pos = module_sequence.index(layer)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise RuntimeError("Stored MLP is inconsistent with its sequential net") from exc
            if layer_idx == len(model.linear_layers) - 1:
                capture_positions.append(linear_pos)
            else:
                capture_positions.append(linear_pos + 1)

        accumulators = [
            torch.zeros(layer.out_features, device=self.device)
            for layer in model.linear_layers
        ]
        total_samples = 0

        while total_samples < self.sample_size:
            batch_size = min(self.batch_size, self.sample_size - total_samples)
            batch = self._draw_boolean_batch(batch_size).to(self.device)
            with torch.no_grad():
                activations: list[torch.Tensor] = []
                current = batch
                for module in module_sequence:
                    current = module(current)
                    activations.append(current)

            for layer_idx, capture_pos in enumerate(capture_positions):
                layer_values = activations[capture_pos]
                accumulators[layer_idx] += layer_values.pow(2).sum(dim=0)

            total_samples += batch_size

        layer_second_moments: dict[str, torch.Tensor] = {
            str(idx + 1): (acc / total_samples).cpu()
            for idx, acc in enumerate(accumulators)
        }

        payload = {
            "sample_size": total_samples,
            "layer_second_moments": layer_second_moments,
        }
        torch.save(payload, self.second_moment_path)
        self._second_moments_cache = {int(k): v for k, v in layer_second_moments.items()}

        model.to("cpu")

    # ------------------------------------------------------------------
    # rehydration
    # ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, directory: Path | str) -> "FourierMlp":
        """Reconstruct a :class:`FourierMlp` instance from disk."""

        base_dir = Path(directory)
        fourier_dir = base_dir / cls.FOURIER_SUBDIR_NAME
        if not fourier_dir.exists():
            raise FileNotFoundError(
                f"Fourier directory does not exist at {fourier_dir}"
            )

        config_path = fourier_dir / cls.CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing MLP config at {config_path}")

        metadata_path = fourier_dir / cls.METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata at {metadata_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)

        mlp_config = MLPConfig.schema().load(config_data)

        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        obj = cls.__new__(cls)
        obj.base_dir = base_dir
        obj.fourier_dir = fourier_dir
        obj.checkpoint_dir = fourier_dir / cls.CHECKPOINT_SUBDIR
        obj.coefficient_dir = fourier_dir / cls.COEFFICIENT_SUBDIR
        obj.coefficient_dir.mkdir(parents=True, exist_ok=True)
        obj.config_path = config_path
        obj.metadata_path = metadata_path
        obj.sample_size = cls._validate_positive_static(
            metadata.get("sample_size"), "sample_size"
        )
        obj.batch_size = cls._validate_positive_static(
            metadata.get("batch_size"), "batch_size"
        )
        obj.dtype = cls._dtype_from_string(metadata.get("dtype"))
        obj.device = torch.device(metadata.get("device", "cpu"))
        obj._mlp_config = mlp_config
        obj._mlp_cache = None
        obj.second_moment_path = fourier_dir / cls.SECOND_MOMENTS_FILENAME
        if not obj.second_moment_path.exists():
            raise FileNotFoundError(
                f"Missing neuron second moments at {obj.second_moment_path}"
            )
        obj._second_moments_cache = None
        return obj

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_fourier_coefficient(
        self,
        indices: Sequence[int],
        layer_index: int,
        neuron_index: int,
    ) -> float:
        """Return the requested Fourier coefficient.

        Parameters
        ----------
        indices:
            Sequence of input dimensions describing the Fourier coefficient.
        layer_index:
            1-based index of the target layer (1 = first hidden layer).
        neuron_index:
            0-based index of the neuron within the specified layer.
        """

        canonical_indices = self._canonicalise_indices(indices)
        if layer_index <= 0:
            raise ValueError("layer_index must be positive and 1-indexed")

        tensor = self._load_cached_coefficients(canonical_indices, layer_index)
        if not (0 <= neuron_index < tensor.size(0)):
            raise IndexError(
                f"neuron_index {neuron_index} out of bounds for layer {layer_index}"
            )
        return float(tensor[neuron_index].item())

    def get_neuron_second_moment(self, layer_index: int, neuron_index: int) -> float:
        r"""Return :math:`\mathbb{E}[a^2]` for the requested neuron activation.

        The values are estimated during construction and cached on disk.  They
        can therefore be retrieved at any point without touching the underlying
        MLP or re-running expensive sampling.
        """

        layer_data = self._load_second_moments()
        if layer_index <= 0:
            raise ValueError("layer_index must be positive and 1-indexed")

        tensor = layer_data.get(layer_index)
        if tensor is None:
            raise ValueError(f"Layer {layer_index} not present in second moment cache")
        if not (0 <= neuron_index < tensor.size(0)):
            raise IndexError(
                f"neuron_index {neuron_index} out of bounds for layer {layer_index}"
            )
        return float(tensor[neuron_index].item())

    # ------------------------------------------------------------------
    # caching helpers
    # ------------------------------------------------------------------
    def _load_cached_coefficients(
        self, indices: tuple[int, ...], layer_index: int
    ) -> torch.Tensor:
        """Load cached Fourier coefficients for ``indices`` and ``layer_index``.

        Missing cache files trigger computation, after which the cached tensor
        for the specific layer is returned.
        """

        cache_path = self._coefficient_path(indices)
        if not cache_path.exists():
            self._compute_and_store_coefficients(indices)

        payload = torch.load(cache_path, map_location="cpu")
        layer_key = str(layer_index)
        layer_coefficients = payload["layer_coefficients"].get(layer_key)
        if layer_coefficients is None:
            raise ValueError(
                f"Layer {layer_index} not present in cached coefficients"
            )
        return layer_coefficients

    def _compute_and_store_coefficients(self, indices: tuple[int, ...]) -> None:
        """Estimate Fourier coefficients for ``indices`` across all neurons."""

        model = self._load_mlp()
        model = model.to(self.device)
        model.eval()

        if not indices:
            fourier_indices = [()]
        else:
            fourier_indices = [indices]

        max_neurons = max(layer.out_features for layer in model.linear_layers)
        module = FourierMlpModule(
            input_dim=self._mlp_config.input_dim,
            fourier_indices=[list(idx) for idx in fourier_indices],
            mlp=model,
            neuron_start_index=0,
            neuron_end_index=max_neurons,
        )
        module = module.to(self.device)
        module.eval()

        accumulators: list[torch.Tensor | None] = [None] * len(model.linear_layers)
        total_samples = 0

        while total_samples < self.sample_size:
            batch_size = min(self.batch_size, self.sample_size - total_samples)
            batch = self._draw_boolean_batch(batch_size).to(self.device)
            with torch.no_grad():
                outputs = module(batch)
            for idx, tensor in enumerate(outputs):
                if tensor is None:
                    continue
                values = tensor.squeeze(-1)
                scaled = values * batch_size
                if accumulators[idx] is None:
                    accumulators[idx] = scaled
                else:
                    accumulators[idx] = accumulators[idx] + scaled
            total_samples += batch_size

        layer_data: dict[str, torch.Tensor] = {}
        for idx, accumulator in enumerate(accumulators, start=1):
            if accumulator is None:
                layer_size = model.linear_layers[idx - 1].out_features
                layer_data[str(idx)] = torch.zeros(layer_size)
            else:
                layer_data[str(idx)] = (accumulator / total_samples).cpu()

        payload = {
            "indices": indices,
            "sample_size": total_samples,
            "layer_coefficients": layer_data,
        }

        cache_path = self._coefficient_path(indices)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)

    def _coefficient_path(self, indices: tuple[int, ...]) -> Path:
        """Return the filesystem path for the given Fourier index tuple."""

        key = "empty" if not indices else "_".join(str(idx) for idx in indices)
        filename = f"indices_{key}.pt"
        return self.coefficient_dir / filename

    def _draw_boolean_batch(self, batch_size: int) -> torch.Tensor:
        r"""Sample ``batch_size`` Boolean vectors from ``\{-1, +1\}^d``."""

        input_dim = self._mlp_config.input_dim
        samples = torch.randint(0, 2, (batch_size, input_dim), device="cpu")
        return samples.to(self.dtype).mul_(2).sub_(1)

    def _load_mlp(self) -> MLP:
        """Materialise the stored MLP from the checkpoint, caching the result."""

        if self._mlp_cache is None:
            model = MLP(self._mlp_config)
            checkpoint = Checkpoint.from_dir(self.checkpoint_dir)
            checkpoint.load(model)
            model.eval()
            self._mlp_cache = model
        return self._mlp_cache

    def _load_second_moments(self) -> dict[int, torch.Tensor]:
        """Return cached second moments, loading them from disk if necessary."""

        if self._second_moments_cache is None:
            payload = torch.load(self.second_moment_path, map_location="cpu")
            layer_data = payload.get("layer_second_moments")
            if not isinstance(layer_data, dict):  # pragma: no cover - defensive
                raise ValueError("Malformed second moment cache")
            cache: dict[int, torch.Tensor] = {}
            for key, value in layer_data.items():
                idx = int(key)
                if not isinstance(value, torch.Tensor):
                    raise ValueError("Malformed second moment tensor in cache")
                cache[idx] = value
            self._second_moments_cache = cache
        return self._second_moments_cache

    # ------------------------------------------------------------------
    # validation utilities
    # ------------------------------------------------------------------
    def _canonicalise_indices(self, indices: Sequence[int]) -> tuple[int, ...]:
        """Return a sorted, duplicate-free tuple of validated Fourier indices."""

        canonical = tuple(sorted(set(indices)))
        for idx in canonical:
            if not 0 <= idx < self._mlp_config.input_dim:
                raise ValueError(
                    f"Fourier index {idx} is outside the valid input range"
                )
        return canonical

    @staticmethod
    def _validate_positive(value: int, name: str) -> int:
        """Ensure ``value`` is a positive integer, raising :class:`ValueError` otherwise."""

        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _validate_positive_static(value: int | None, name: str) -> int:
        """Static variant of :meth:`_validate_positive` for classmethod use."""

        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Metadata field '{name}' must be a positive integer")
        return value

    @staticmethod
    def _dtype_to_string(dtype: torch.dtype) -> str:
        """Convert a ``torch.dtype`` into a plain string for JSON serialisation."""

        return str(dtype).replace("torch.", "")

    @staticmethod
    def _dtype_from_string(name: str | None) -> torch.dtype:
        """Recover a ``torch.dtype`` from a string stored in metadata."""

        if not isinstance(name, str):
            raise ValueError("Metadata field 'dtype' is missing or invalid")
        attr_name = name if name.startswith("float") or name.startswith("double") else name
        if not hasattr(torch, attr_name):
            raise ValueError(f"Unsupported dtype '{name}' in metadata")
        dtype = getattr(torch, attr_name)
        if not isinstance(dtype, torch.dtype):  # pragma: no cover - defensive
            raise ValueError(f"Attribute torch.{attr_name} is not a dtype")
        return dtype

