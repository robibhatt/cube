# Changelog

## Unreleased
- Removed the internal `_distribution` attribute in favour of direct distribution implementations.
- Added `covariance_matrix` helper to `Gaussian` for convenient covariance access.
- Fixed CPU fallback behaviour to ensure Cholesky decomposition runs on CPU when CUDA is unavailable.
- Experiments now always initialize their trainers automatically when created or
  loaded via ``from_dir``.
- Removed `coordinate_model` and `composed_model` along with associated configs and tests.
- Added `SumProdTarget` for summing multiple product terms via configurable index groups.
- Removed representor distributions, model representors, and related helpers.
- Removed the cube experiment script, configuration, implementation, and associated tests.
