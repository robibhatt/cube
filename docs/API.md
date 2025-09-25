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

- **`JointDistribution`** – base class providing `(X, y)` sampling. Simple
  distributions like `Gaussian` are implemented as joint distributions with
  `y=None`.

- **`JointDistribution`** – base class providing `(X, y)` sampling.
  - `sample(n, seed)` – draw pairs using the given seed. A generator is
        constructed internally.
    - `preferred_provider()` – returns a string identifying the desired
          data provider (e.g., ``"TensorDataProvider"``). This string is used to
          look up the class in :data:`DATA_PROVIDER_REGISTRY`.
  - Use ``create_joint_distribution(cfg, device)`` to build an instance. The
        factory resolves ``cfg.distribution_type`` via
        :data:`JOINT_DISTRIBUTION_REGISTRY`.
  - Use ``create_data_provider_from_distribution(dist, dataset_dir, batch_size, dataset_size, seed)``
        to build the provider.  Device placement is inherited from ``dist`` and
        the factory resolves the provider class via :data:`DATA_PROVIDER_REGISTRY`.
  - **`MappedJointDistribution`** – pairs a `Distribution` with a `TargetFunction`; `sample(seed)` draws `X` then computes `y = f(X)`.
- **`NoisyDistribution`** – mirrors a `MappedJointDistribution` built from a base distribution and target function, then adds Gaussian noise with configurable mean and standard deviation to its targets.

#### Target Functions

- **`SumProdTarget`** – computes weighted products of selected coordinates and returns their (optionally normalized) sum. Configuration lives in :class:`~src.models.targets.configs.sum_prod.SumProdTargetConfig`, which specifies the coordinate groups and weights.

#### Data Providers

 - **`DataIterator`** – dataclass interface for creating datasets and loaders.
   Stores ``joint_distribution``, ``dataset_dir``, ``seed``, ``dataset_size``
   and ``batch_size`` for use by subclasses when constructing ``DataLoader``
   instances.
  - **`TensorDataProvider`** – streams samples using ``DeterministicDataset``.
  - `make_loader()` – instantiate a ``DeterministicDataset`` of that size and return a `DataLoader`.
 - **`DeterministicDataset`** – dataset that draws samples deterministically from a `JointDistribution`.

### `src/models`

#### Architectures

- **`Model`** (abstract, subclass of `nn.Module`) – base class holding a `ModelConfig`.
- **`MLP`** – configurable feed-forward network with optional activations before and after layers.  Set ``mup=True`` in its config to build the network using μP layers (``mup.MuLinear``/``MuReadout``) instead of standard ``nn.Linear`` modules.
- **`create_model(config)`** – factory that looks up architectures via `MODEL_REGISTRY`.

#### Configs

- `ModelConfig` – dataclass base with optional `input_shape`, `output_shape`.
- `MLPConfig` – defines layer dimensions, activation type, and a ``mup`` flag to enable μP layers.

### `src/training`

#### Trainers

- **`Trainer`** – manages end-to-end training of a model.
  - `ready_for_trainer()` – sanity-check configuration; invoked by ``Trainer`` during initialization.
  - `_create_directories()` – prepares dataset and checkpoint folders.
  - `_make_loader(split)` – create a data provider for the requested split and return its ``data_loader``.
  - `get_loader(split)` – public wrapper returning the same `DataLoader` used internally.
  - The provider is always created internally via ``create_data_provider_from_distribution()``,
        which constructs an instance after resolving the class from
        ``DATA_PROVIDER_REGISTRY``.
  - `_initialize_model_and_optimizer()` / `_load_model_and_optimizer()` – helper methods used when (re)starting training.
  - `_train_loss(model)` and `test_loss()` – compute losses over datasets.
  - `train()` – entry point that chooses between stepwise or solver-based training.

  Internal methods prefixed with `_` are not strictly private – they are invoked from tests and other modules.

#### Optimizers

- **`Optimizer`** (abstract) – wrapper over optimization algorithms.
- **`Adam`** – simple example using `torch.optim.Adam`.
- `create_optimizer(config, model)` – factory using `OPTIMIZER_REGISTRY`.

#### Configs

- `OptimizerConfig` – base dataclass with `optimizer_type` field.
- `AdamConfig` – defines learning rate for the `Adam` optimizer.
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

Configuration throughout the codebase is captured via small dataclasses rather than external YAML or JSON.  Key configuration classes include `ModelConfig` and its subclasses, `OptimizerConfig` subclasses, and `TrainerConfig` in `src/training/trainer_config` and optimizer configs in `src/training/optimizers/configs`.  Experiments also define dataclass configs (e.g., `RepeatedSamplingExperimentConfig`).  These objects are created frequently in tests and scripts, so correct field values are critical for proper directory creation and checkpoint handling.

## Interactions and Dependencies

- `Trainer` relies on `Checkpoint` for saving/loading model parameters and optimizer state.
- `Experiments` compose multiple `Trainer` instances and use various `JointDistribution` implementations to generate data.
 - `Optimizer` instances are created per model via `create_optimizer` and are stored in checkpoints.

## Automatic Plugin Imports

The project relies on small registries for models, optimizers, and other
plugins. Each package's ``__init__`` module calls
``import_submodules(__name__)`` from :mod:`src.utils.plugin_loader`, which walks
the package and imports every submodule. Importing a module executes decorators
like ``@register_optimizer`` that populate the relevant registry. Target
functions are simpler: the only implementation, :class:`SumProdTarget`, is
imported directly and constructed from its config.

The training loop itself is provided by the concrete :class:`Trainer` in
``src/training/trainer.py`` and configured via :class:`TrainerConfig` from
``src/training/trainer_config.py``; no registry is involved.

Adding a new experiment or optimizer therefore only requires placing the module
inside the appropriate package and decorating the class. No manual import
statements are needed to enable discovery. Target functions do not participate
in this plugin mechanism anymore because they are instantiated explicitly.

## Device Placement

`Trainer` automatically selects ``cuda`` if available, falling back to ``cpu``.
Data providers inherit placement from the ``JointDistribution``.

```python
from pathlib import Path
import torch
from src.data.joint_distributions.configs.gaussian import GaussianConfig
from src.data.joint_distributions.joint_distribution_factory import create_joint_distribution
from src.models.targets.sum_prod import SumProdTarget
from src.models.targets.configs.sum_prod import SumProdTargetConfig
from src.data.joint_distributions.configs.mapped_joint_distribution import (
    MappedJointDistributionConfig,
)
from src.models.architectures.configs.mlp import MLPConfig
from src.training.optimizers.configs.adam import AdamConfig
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.training.trainer import Trainer

dist_cfg = GaussianConfig(input_shape=torch.Size([4]), mean=0.0, std=1.0)
dist = create_joint_distribution(dist_cfg, device=torch.device("cpu"))
target_cfg = SumProdTargetConfig(
    input_shape=dist.input_shape,
    indices_list=[[0, 1], [2, 3]],
    weights=[0.5, 1.5],
)
joint_cfg = MappedJointDistributionConfig(
    distribution_config=dist_cfg,
    target_function_config=target_cfg,
)
cfg = TrainerConfig(
    model_config=MLPConfig(
        input_dim=2,
        output_dim=1,
        hidden_dims=[4],
        activation="relu",
        start_activation=False,
        end_activation=False,
    ),
    optimizer_config=AdamConfig(lr=1e-3),
    joint_distribution_config=joint_cfg,
    training_size=32,
    test_size=16,
    batch_size=8,
    epochs=1,
    home_dir=Path("/tmp/home"),
    loss_config=LossConfig(name="MSELoss"),
    seed=0,
)

trainer = Trainer(cfg)
target = SumProdTarget(target_cfg)
```

## Non‑Public Functions Used Elsewhere

Methods such as `ready_for_trainer` and `_create_directories` in `Trainer` and helper methods like `_forward` inside `SumProdTarget` are named with a leading underscore but are invoked across the codebase (e.g., in unit tests).  They should therefore be considered part of the semi-public interface.

## Side Effects

Several operations interact with the filesystem:

- `Checkpoint.__post_init__` and `Checkpoint.save` write to disk immediately.
- `Trainer` creates directories and log files within its `_create_directories` method and writes a CSV-style log each time `_save_checkpoint` is called.

## Summary

The project provides a collection of small, composable building blocks for studying neural network generalization.  Models and optimizers are configured via dataclasses and linked together by factory functions, while the training loop uses the :class:`Trainer` class configured by :class:`TrainerConfig`.  Experiments orchestrate multiple trainers to evaluate training recipes on different data distributions.

