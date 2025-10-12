# MLP Generalization API

This document summarizes the Python modules, classes, and public functions provided by the `mlp_generalization` project.  It is generated for high level reference and does not replace docstrings or inline comments.

## Package Layout

```
src/
  checkpoints/          - Model checkpoint persistence
  data/                 - Probability distributions and target functions
  experiments/          - Experiment orchestration logic
  models/               - Model architectures and representations
  training/             - Training loops and optimizers
  utils/                - Helper utilities
```

## Modules and Classes

### `src/checkpoints`

- **`Checkpoint`** – persist PyTorch model and optimizer state.
  - `__post_init__(self)` – ensures directory exists and stores configuration.
  - `from_dir(checkpoint_dir: Path)` – load `Checkpoint` metadata from disk.
  - `save(model, optimizer=None)` – write `checkpoint.pth` with model and optional optimizer state.
  - `load(model, optimizer=None)` – restore state into provided objects.

### `src/data`

#### Joint Distributions

- **`CubeDistribution`** – generates binary inputs from a hypercube and passes
  them through :class:`~src.models.targets.sum_prod.SumProdTarget` before adding
  configurable Gaussian noise to the targets. The accompanying
  :class:`~src.data.cube_distribution_config.CubeDistributionConfig`
  dataclass stores the parameters required to instantiate the distribution.
- Instantiate :class:`~src.data.noisy_data_provider.NoisyProvider`
  directly to stream data from the distribution. Device placement is inherited
  from ``dist`` and batches are generated lazily via PyTorch ``DataLoader``
  objects.

#### Target Functions

- **`SumProdTarget`** – computes weighted products of selected coordinates and returns their (optionally normalized) sum. Configuration lives in :class:`~src.models.targets.configs.sum_prod.SumProdTargetConfig`, which specifies the coordinate groups and weights.

#### Data Providers

 - **`NoisyProvider`** – dataclass that wraps a
   :class:`CubeDistribution` and exposes deterministic iteration over noisy
   samples. Stores ``joint_distribution``, ``seed``, ``dataset_size`` and
   ``batch_size`` and internally constructs a ``SeededNoisyDataset`` backed by
   a PyTorch ``DataLoader``.
 - **`SeededNoisyDataset`** – dataset that draws reproducible noisy samples
   from a `CubeDistribution` by seeding the base distribution per index.

### `src/models`

#### Architectures

- **`MLP`** – configurable feed-forward network with optional activations before and after layers.  The implementation now always uses μP layers (``mup.MuLinear``/``MuReadout``); constructing it with ``mup=False`` raises an error.

#### Configs

- `MLPConfig` – defines layer dimensions, activation type, and a ``mup`` flag that must be ``True``.  Input and output shapes are derived from the dimensions when needed, so the config no longer stores them explicitly.

### `src/training`

#### Trainers

- **`Trainer`** – manages end-to-end training of a model.
  - `ready_for_trainer()` – sanity-check configuration; invoked by ``Trainer`` during initialization.
  - `_create_directories()` – prepares dataset and checkpoint folders.
  - `_make_loader(split)` – create a :class:`NoisyProvider` for the requested
    split and return its ``data_loader``.
  - `get_loader(split)` – public wrapper returning the same `DataLoader` used
    internally.
  - `_initialize_model_and_optimizer()` / `_load_model_and_optimizer()` – helper methods used when (re)starting training.
  - `_train_loss(model)` and `test_loss()` – compute losses over datasets.
  - `train()` – entry point that chooses between stepwise or solver-based training.

  Internal methods prefixed with `_` are not strictly private – they are invoked from tests and other modules.

#### Optimizers

- **`Sgd`** – wraps `mup.MuSGD` and expects μP-compatible models/configs. A
  fallback ``NullStepper`` is used when the target model exposes no trainable
  parameters so that training loops can continue without special casing.

#### Configs

- `SgdConfig` – defines learning rate, μP toggle, and weight decay for the SGD
  optimiser.
- `TrainerConfig` – bundles model/optimizer configs and dataset sizes.

### `src/experiments`

-Experiments coordinate multiple trainers and datasets.

- **`Experiment`** (abstract) – saves experiment metadata to disk and loads via `from_dir`. Provides a `train()` helper. Each experiment holds a `SeedManager` derived from ``ExperimentConfig.seed`` and uses ``spawn_seed()`` to create a unique seed for every trainer.
- **`RepeatedSamplingExperiment`** – runs several independent `Trainer` instances and aggregates statistics in `consolidate_results()`.
- **`FeatureLearningExperiment`**, **`SigmaScalingExperiment`**, **`BenignOverfitMLPExperiment`** – domain-specific experiment types varying noise levels or model structures.
- **`ExperimentInspector`** – meta-experiment that loads an existing experiment and spawns new experiments on model representations.

### `src/utils`

- `seed_all(seed)` – utility to seed Python, NumPy, and PyTorch RNGs for reproducibility. `Trainer` calls this during initialization so datasets and optimizers start from a known seed.
- `SeedManager` – dataclass wrapping `random.Random` for spawning deterministic seeds and `torch.Generator` instances. Experiments hold a `SeedManager` to spawn per‑trainer seeds; each trainer keeps its own manager for generators during training.

## Configuration Objects

Configuration throughout the codebase is captured via small dataclasses rather than external YAML or JSON.  Key configuration classes include `MLPConfig`, `SgdConfig`, and `TrainerConfig` in `src/training/trainer_config`.  Experiments also define dataclass configs (e.g., `RepeatedSamplingExperimentConfig`).  These objects are created frequently in tests and scripts, so correct field values are critical for proper directory creation and checkpoint handling.

## Interactions and Dependencies

- `Trainer` relies on `Checkpoint` for saving/loading model parameters and optimizer state.
- `Experiments` compose multiple `Trainer` instances and use various `JointDistribution` implementations to generate data.
- `Sgd` instances are created per model and are stored in checkpoints.

## Automatic Plugin Imports

Optimisers are instantiated directly; the only implementation currently
supported is μP-aware stochastic gradient descent. Configuration is expressed
through :class:`SgdConfig` and passed straight to :class:`Sgd`, so no plugin
system or registry is required. Data providers and targets are likewise created
explicitly from their configs.

The training loop itself is provided by the concrete :class:`Trainer` in
``src/training/trainer.py`` and configured via :class:`TrainerConfig` from
``src/training/trainer_config.py``.

## Device Placement

`Trainer` automatically selects ``cuda`` if available, falling back to ``cpu``.
Data providers inherit placement from the ``JointDistribution``.

```python
from pathlib import Path
import torch
from src.data.cube_distribution import CubeDistribution
from src.data.cube_distribution_config import CubeDistributionConfig
from src.models.targets.sum_prod import SumProdTarget
from src.models.mlp_config import MLPConfig
from src.training.sgd_config import SgdConfig
from src.training.trainer_config import TrainerConfig
from src.training.trainer import Trainer

cube_cfg = CubeDistributionConfig(
    input_dim=4,
    indices_list=[[0], [1], [2], [3]],
    weights=[0.5, 1.5, 0.25, 0.75],
    noise_mean=0.0,
    noise_std=0.1,
)
cfg = TrainerConfig(
    mlp_config=MLPConfig(
        input_dim=cube_cfg.input_dim,
        output_dim=1,
        hidden_dims=[4],
        activation="relu",
        start_activation=False,
        end_activation=False,
    ),
    optimizer_config=SgdConfig(lr=1e-3),
    cube_distribution_config=cube_cfg,
    training_size=32,
    test_size=16,
    batch_size=8,
    epochs=1,
    home_dir=Path("/tmp/home"),
    seed=0,
)

trainer = Trainer(cfg)
target = SumProdTarget(cube_cfg.target_function_config)
dist = CubeDistribution(cube_cfg, trainer.device)
```

## Non‑Public Functions Used Elsewhere

Methods such as `ready_for_trainer` and `_create_directories` in `Trainer` and helper methods like `_forward` inside `SumProdTarget` are named with a leading underscore but are invoked across the codebase (e.g., in unit tests).  They should therefore be considered part of the semi-public interface.

## Side Effects

Several operations interact with the filesystem:

- `Checkpoint.__post_init__` and `Checkpoint.save` write to disk immediately.
- `Trainer` creates directories and log files within its `_create_directories` method and writes a CSV-style log each time `_save_checkpoint` is called.

## Summary

The project provides a collection of small, composable building blocks for studying neural network generalization.  Models and optimizers are configured via dataclasses and instantiated directly, while the training loop uses the :class:`Trainer` class configured by :class:`TrainerConfig`.  Experiments orchestrate multiple trainers to evaluate training recipes on different data distributions.

