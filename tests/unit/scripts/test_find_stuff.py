import json
from pathlib import Path

from scripts.find_stuff import (
    find_training_directories_with_criteria,
    has_activation_with_few_ancestors,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_has_activation_accepts_prefixed_graph_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    training_dir = run_dir / "trainer"
    training_dir.mkdir(parents=True)

    graph_dir = run_dir / "mlp_graph_0.5"
    graph_dir.mkdir()
    _write_json(graph_dir / "graph.json", {"ancestors": [1, 2, 3]})

    assert has_activation_with_few_ancestors(str(training_dir))


def test_find_training_directories_with_criteria_includes_prefixed_graph(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment"
    training_dir = run_dir / "trainer"
    training_dir.mkdir(parents=True)

    # Identify the training directory
    (training_dir / "frobenius_drifts.json").write_text("{}", encoding="utf-8")

    # Provide ancestor data in a prefixed graph directory
    graph_dir = run_dir / "mlp_graph_1.0"
    graph_dir.mkdir()
    _write_json(graph_dir / "graph.json", {"ancestors": [1, 2, 3]})

    # Supply linear probe results with a sufficiently large test error
    linear_dir = training_dir / "linear_results"
    linear_dir.mkdir()
    _write_json(linear_dir / "results.json", {"test_error": 1e-6})

    result = find_training_directories_with_criteria(str(tmp_path))

    assert [Path(path) for path in result] == [training_dir]
