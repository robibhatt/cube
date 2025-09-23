import torch

from src.data.joint_distributions import create_joint_distribution
from src.data.joint_distributions.configs.paired_representor_distribution import (
    PairedRepresentorDistributionConfig,
)


def _make_cfg(trainer):
    return PairedRepresentorDistributionConfig(
        base_distribution_config=trainer.config.joint_distribution_config,
        teacher_model_config=trainer.config.model_config,
        teacher_checkpoint_dir=trainer.checkpoint_dir,
        teacher_rep_id=1,
        teacher_from_rep_id=0,
        student_model_config=trainer.config.model_config,
        student_checkpoint_dir=trainer.checkpoint_dir,
        student_rep_id=0,
        student_from_rep_id=0,
    )


def test_paired_representor_distribution_sample(trained_trainer, model_representor):
    cfg = _make_cfg(trained_trainer)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    n_samples = 4
    s, t = dist.sample(n_samples, seed=0)
    assert s.shape == (
        n_samples, *model_representor.representation_shape(cfg.student_rep_id)
    )
    assert t.shape == (
        n_samples, *model_representor.representation_shape(cfg.teacher_rep_id)
    )
    dist_str = str(dist)
    assert "PairedRepresentorDistribution" in dist_str


def test_paired_representor_forward_matches_forward_X(trained_trainer):
    cfg = _make_cfg(trained_trainer)
    dist = create_joint_distribution(cfg, torch.device("cpu"))
    X, _ = dist.base_sample(3, seed=0)
    s_fwd, t_fwd = dist.forward(X)
    s_fx, t_fx = dist.forward_X(X)
    assert torch.allclose(s_fwd, s_fx)
    assert torch.allclose(t_fwd, t_fx)
