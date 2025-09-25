import importlib


def test_optimizer_registry_bootstrap():
    import src.training.optimizers as optimizers
    importlib.reload(optimizers)
    assert "Adam" in optimizers.OPTIMIZER_REGISTRY
    assert "sgd" in optimizers.OPTIMIZER_REGISTRY
def test_trainer_module_bootstrap():
    import src.training.trainer as trainer
    importlib.reload(trainer)
    assert hasattr(trainer, "Trainer")


def test_data_provider_registry_bootstrap():
    import src.data.providers as providers
    importlib.reload(providers)
    assert "TensorDataProvider" in providers.DATA_PROVIDER_REGISTRY
    assert "NoisyProvider" in providers.DATA_PROVIDER_REGISTRY


def test_joint_distribution_registry_bootstrap():
    import src.data.joint_distributions as dists
    importlib.reload(dists)
    assert "Gaussian" in dists.JOINT_DISTRIBUTION_REGISTRY
    assert "Hypercube" in dists.JOINT_DISTRIBUTION_REGISTRY


def test_model_registry_bootstrap():
    import src.models.bootstrap  # noqa: F401
    from src.models.architectures import MODEL_REGISTRY

    assert "MLP" in MODEL_REGISTRY
