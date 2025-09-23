import csv
import src.models.bootstrap  # noqa: F401
import src.data.joint_distributions.bootstrap  # noqa: F401

from src.experiments.configs.cube_experiment import CubeExperimentConfig
from src.experiments.experiments.cube_experiment import CubeExperiment
from src.utils.seed_manager import SeedManager


def _make_config(tmp_path, weight_decay_l1: float = 0.0):
    return CubeExperimentConfig(
        home_directory=tmp_path,
        seed=0,
        dimension=2,
        k=1,
        depth=1,
        width=1,
        learning_rate=0.01,
        weight_decay=0.0,
        weight_decay_l1=weight_decay_l1,
        mup=False,
        epochs=1,
        early_stopping=0.0,
        train_size=1,
        batch_size=1,
    )


def test_trainer_seed_from_experiment_seed(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.seed = 42
    exp = CubeExperiment(cfg)
    trainer_seed = exp.get_trainer_configs()[0][0].seed
    expected_seed = SeedManager(42).spawn_seed()
    assert trainer_seed == expected_seed
    trainer_seed2 = exp.get_trainer_configs()[0][0].seed
    assert trainer_seed2 == trainer_seed


def test_train_and_consolidate(tmp_path):
    cfg = _make_config(tmp_path)
    exp = CubeExperiment(cfg)
    exp.train()
    rows = exp.consolidate_results()
    results_csv = tmp_path / "results.csv"
    trainer_dir = tmp_path / "trainer"
    grads_csv = trainer_dir / "neuron_input_gradients.csv"
    vis_png = trainer_dir / "visualization.png"
    assert results_csv.exists()
    assert grads_csv.exists()
    assert vis_png.exists()
    assert len(rows) == 1
    with open(results_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        data_row = next(reader)
    assert float(data_row["final_train_loss"]) == rows[0]["final_train_loss"]


def test_weight_decay_l1_passed_to_trainer(tmp_path):
    cfg = _make_config(tmp_path, weight_decay_l1=0.5)
    exp = CubeExperiment(cfg)
    trainer_cfg = exp.get_trainer_configs()[0][0]
    assert trainer_cfg.weight_decay_l1 == 0.5
