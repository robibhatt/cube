import torch
import pytest
import src.models.bootstrap  # noqa: F401
from src.experiments.utilities.mlp_comparer import MLPComparer
from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig
from src.training.loss.configs.loss import LossConfig
from src.models.representors.representor_factory import create_model_representor
from src.data.joint_distributions.paired_representor_distribution import (
    PairedRepresentorDistribution,
)
from tests.helpers.stubs import StubJointDistribution


def _train(home_dir, mlp_cfg, opt_cfg) -> Trainer:
    home_dir.mkdir()
    cfg = TrainerConfig(
        model_config=mlp_cfg,
        optimizer_config=opt_cfg,
        joint_distribution_config=StubJointDistribution._Config(
            X=torch.zeros(4, mlp_cfg.input_dim),
            y=torch.zeros(4, mlp_cfg.output_dim),
        ),
        train_size=4,
        test_size=2,
        batch_size=2,
        epochs=1,
        home_dir=home_dir,
        loss_config=LossConfig(name="MSELoss"),
        seed=0,
    )
    trainer = Trainer(cfg)
    trainer.train()
    return trainer


def test_layer_to_rep_id(tmp_path, mlp_config, adam_config):
    teacher = _train(tmp_path / "teacher", mlp_config, adam_config)
    student = _train(tmp_path / "student", mlp_config, adam_config)
    comparer = MLPComparer(
        teacher.config.home_dir, student.config.home_dir, seed=0
    )

    student_repr = create_model_representor(
        student.config.model_config,
        student.checkpoint_dir,
        device=comparer.device,
    )

    assert comparer._layer_to_rep_id(student_repr, 0) == 0

    expected_hidden = student_repr.from_representation_dict(
        {"layer_index": 1, "post_activation": True}
    )
    assert comparer._layer_to_rep_id(student_repr, 1) == expected_hidden

    final_layer = len(student_repr.model_config.hidden_dims) + 1
    assert comparer._layer_to_rep_id(student_repr, final_layer) == student_repr.get_final_rep_id()


def test_get_layers_distribution(tmp_path, mlp_config, adam_config):
    teacher = _train(tmp_path / "teacher", mlp_config, adam_config)
    student = _train(tmp_path / "student", mlp_config, adam_config)
    comparer = MLPComparer(
        teacher.config.home_dir, student.config.home_dir, seed=0
    )

    dist = comparer.get_layers_distribution(student_layer=1, teacher_layer=2)
    assert isinstance(dist, PairedRepresentorDistribution)
    assert isinstance(dist.base_joint_distribution, StubJointDistribution)

    student_repr = dist.student_representor
    teacher_repr = dist.teacher_representor
    assert dist.student_rep_id == comparer._layer_to_rep_id(student_repr, 1)
    assert dist.teacher_rep_id == comparer._layer_to_rep_id(teacher_repr, 2)
    assert dist.student_from_rep_id == comparer._layer_to_rep_id(student_repr, 0)
    assert dist.teacher_from_rep_id == comparer._layer_to_rep_id(teacher_repr, comparer.start_layer)

    s, t = dist.sample(2, seed=comparer.seed_mgr.spawn_seed())
    assert s.shape == (2, *student_repr.representation_shape(dist.student_rep_id))
    assert t.shape == (2, *teacher_repr.representation_shape(dist.teacher_rep_id))


def test_get_test_provider(tmp_path, mlp_config, adam_config):
    teacher = _train(tmp_path / "teacher", mlp_config, adam_config)
    student = _train(tmp_path / "student", mlp_config, adam_config)
    comparer = MLPComparer(
        teacher.config.home_dir, student.config.home_dir, seed=0
    )

    provider = comparer.get_test_provider(seed=123)
    batches = list(provider)

    # Should match student's batch size and cover the whole test set
    assert provider.batch_size == student.config.batch_size
    total = sum(b[0].shape[0] for b in batches)
    assert total == student.config.test_size

    # Same seed -> identical data
    provider2 = comparer.get_test_provider(seed=123)
    batches2 = list(provider2)
    assert all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for a, b in zip(batches, batches2)
    )


def test_save_loss_table_creates_file(tmp_path, mlp_config, adam_config, monkeypatch):
    teacher = _train(tmp_path / "teacher", mlp_config, adam_config)
    student = _train(tmp_path / "student", mlp_config, adam_config)
    comparer = MLPComparer(teacher.config.home_dir, student.config.home_dir, seed=0)

    import torch.nn as nn

    monkeypatch.setattr(
        MLPComparer, "compute_test_loss", lambda self, s, t, bias=True, lambdas=None: 0.0
    )

    out_file = tmp_path / "table.png"
    comparer.save_loss_table(out_file)

    from PIL import Image

    img = Image.open(out_file)
    width, height = img.size
    assert width > 0 and height > 0


def test_save_loss_table_respects_start_layer(tmp_path, mlp_config, adam_config, monkeypatch):
    teacher = _train(tmp_path / "teacher", mlp_config, adam_config)
    student = _train(tmp_path / "student", mlp_config, adam_config)
    comparer = MLPComparer(teacher.config.home_dir, student.config.home_dir, seed=0)

    calls: list[tuple[int, int]] = []

    def fake_compute(self, s, t, bias=True, lambdas=None):
        calls.append((s, t))
        return 0.0

    monkeypatch.setattr(MLPComparer, "compute_test_loss", fake_compute)

    out_file = tmp_path / "table.png"
    comparer.save_loss_table(out_file, start_teacher_layer=1)

    assert all(t >= 1 for (_, t) in calls)


def test_handles_different_depths(tmp_path, monkeypatch, adam_config):
    """Comparer operates when teacher and student depths differ."""
    from src.models.architectures.configs.mlp import MLPConfig
    import torch.nn as nn

    teacher_cfg = MLPConfig(
        input_dim=3,
        hidden_dims=[4, 2],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )
    student_cfg = MLPConfig(
        input_dim=3,
        hidden_dims=[4],
        activation="relu",
        output_dim=1,
        start_activation=False,
        end_activation=False,
    )

    teacher = _train(tmp_path / "teacher", teacher_cfg, adam_config)
    student = _train(tmp_path / "student", student_cfg, adam_config)

    comparer = MLPComparer(teacher.config.home_dir, student.config.home_dir, seed=0)

    monkeypatch.setattr(
        PairedRepresentorDistribution,
        "k_fold_linear",
        lambda self, seed, n_samples=None, bias=True, lambdas=None, k=5: (nn.Identity(), 0.0),
    )
    dummy_provider = [
        (torch.zeros(1, teacher_cfg.input_dim), torch.zeros(1, teacher_cfg.output_dim))
    ]
    monkeypatch.setattr(
        MLPComparer, "get_test_provider", lambda self, seed: dummy_provider
    )

    loss = comparer.compute_test_loss(student_layer=1, teacher_layer=1)
    assert loss == pytest.approx(0.0)
