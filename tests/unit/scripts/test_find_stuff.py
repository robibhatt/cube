import json
from pathlib import Path

from scripts.find_stuff import (
    MAX_ANCESTOR_COUNT,
    MAX_FINAL_TEST_LOSS,
    MIN_ACTIVATION,
    MIN_LINEAR_LOSS,
    find_training_directories_with_criteria,
    has_activation_with_few_ancestors,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_has_activation_accepts_prefixed_graph_directory(tmp_path: Path) -> None:
    training_dir = tmp_path / "trainer"
    graph_dir = training_dir / "mlp_graph_0.5"
    graph_dir.mkdir(parents=True)

    neuron_path = graph_dir / "layer_00_neuron_000.json"
    _write_json(
        neuron_path,
        {
            "ancestors": list(range(MAX_ANCESTOR_COUNT)),
        },
    )

    activation_path = neuron_path.with_name("layer_00_neuron_000_activations.csv")
    activation_path.write_text(
        "activation\n" f"{MIN_ACTIVATION + 0.5}\n",
        encoding="utf-8",
    )

    assert has_activation_with_few_ancestors(str(training_dir))


def test_find_training_directories_with_criteria_includes_prefixed_graph(tmp_path: Path) -> None:
    training_dir = tmp_path / "trainer"
    training_dir.mkdir()

    # Provide ancestor data in a prefixed graph directory
    graph_dir = training_dir / "mlp_graph_1.0"
    graph_dir.mkdir()

    neuron_path = graph_dir / "layer_00_neuron_000.json"
    _write_json(
        neuron_path,
        {
            "ancestors": list(range(MAX_ANCESTOR_COUNT)),
        },
    )
    neuron_path.with_name("layer_00_neuron_000_activations.csv").write_text(
        "activation\n" f"{MIN_ACTIVATION + 0.25}\n",
        encoding="utf-8",
    )

    # Supply linear probe results with a sufficiently large test error
    linear_results = training_dir / "linear_results.csv"
    linear_results.write_text(
        "test_mse\n" f"{MIN_LINEAR_LOSS}\n",
        encoding="utf-8",
    )

    # Provide final test loss information
    trainer_results = training_dir / "results.csv"
    trainer_results.write_text(
        "final_test_loss\n" f"{MAX_FINAL_TEST_LOSS}\n",
        encoding="utf-8",
    )

    result = find_training_directories_with_criteria(str(tmp_path))

    assert [Path(path) for path in result] == [training_dir]


def _create_training_directory(
    root: Path,
    name: str,
    *,
    activation: float,
    test_mse: float,
    final_test_loss: float,
    ancestor_count: int = MAX_ANCESTOR_COUNT,
) -> Path:
    training_dir = root / name
    training_dir.mkdir()

    graph_dir = training_dir / "mlp_graph_0.5"
    graph_dir.mkdir()

    neuron_path = graph_dir / "layer_00_neuron_000.json"
    _write_json(
        neuron_path,
        {
            "ancestors": list(range(ancestor_count)),
        },
    )
    neuron_path.with_name("layer_00_neuron_000_activations.csv").write_text(
        "activation\n" f"{activation}\n",
        encoding="utf-8",
    )

    linear_results = training_dir / "linear_results.csv"
    linear_results.write_text(
        "test_mse\n" f"{test_mse}\n",
        encoding="utf-8",
    )

    trainer_results = training_dir / "results.csv"
    trainer_results.write_text(
        "final_test_loss\n" f"{final_test_loss}\n",
        encoding="utf-8",
    )

    return training_dir


def test_find_training_directories_with_criteria_filters_candidates(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()

    qualifying = _create_training_directory(
        run_root,
        "qualifying_run",
        activation=MIN_ACTIVATION + 0.3,
        test_mse=MIN_LINEAR_LOSS + 0.1,
        final_test_loss=MAX_FINAL_TEST_LOSS,
    )

    _create_training_directory(
        run_root,
        "too_many_ancestors",
        activation=MIN_ACTIVATION + 1.0,
        test_mse=MIN_LINEAR_LOSS + 0.5,
        final_test_loss=MAX_FINAL_TEST_LOSS,
        ancestor_count=MAX_ANCESTOR_COUNT + 1,
    )

    _create_training_directory(
        run_root,
        "high_final_loss",
        activation=MIN_ACTIVATION + 0.5,
        test_mse=MIN_LINEAR_LOSS + 0.2,
        final_test_loss=MAX_FINAL_TEST_LOSS + 0.1,
    )

    extra = run_root / "incomplete_run"
    (extra / "mlp_graph_0.5").mkdir(parents=True)

    result = {
        Path(path) for path in find_training_directories_with_criteria(str(run_root))
    }

    assert result == {qualifying}
