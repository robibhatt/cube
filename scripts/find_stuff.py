import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional


MAX_ANCESTOR_COUNT = 3
"""Maximum allowed number of ancestors for an activation to qualify."""

MIN_LINEAR_LOSS = 0.2
"""Minimum linear loss (``test_error``) value required to flag a directory."""

MAX_FINAL_TEST_LOSS = 0.01
"""Maximum allowed ``final_test_loss`` found in ``trainer/results.csv``."""

MIN_ACTIVATION = 0.1
"""Minimum activation value required for a neuron to be considered."""


@dataclass(frozen=True)
class NeuronMatch:
    """Data describing a neuron that satisfies the search criteria."""

    neuron_path: str
    activation_path: str
    ancestor_count: int
    max_activation: float


def iter_training_directories(root: str) -> Iterable[str]:
    """Yield directories that look like completed training runs."""

    for dirpath, _, filenames in os.walk(root):
        if any(name.startswith("linear_results") for name in filenames):
            yield dirpath


def _iter_neuron_json_paths(training_dir: str) -> Iterator[str]:
    """Yield JSON files that describe individual neurons within a training run."""

    for current_root, _, filenames in os.walk(training_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if not filename.startswith("layer_") or "_neuron_" not in filename:
                continue
            yield os.path.join(current_root, filename)


def _activation_csv_path(json_path: str) -> str:
    base, _ = os.path.splitext(json_path)
    return f"{base}_activations.csv"


def _max_activation_from_csv(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "activation" not in reader.fieldnames:
                return None
            values = []
            for row in reader:
                raw = row.get("activation")
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
            if not values:
                return None
            return max(values)
    except OSError:
        return None


def _load_neuron_match(json_path: str) -> Optional[NeuronMatch]:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    ancestors = data.get("ancestors")
    if not isinstance(ancestors, list) or len(ancestors) > MAX_ANCESTOR_COUNT:
        return None

    activation_path = _activation_csv_path(json_path)
    max_activation = _max_activation_from_csv(activation_path)
    if max_activation is None or max_activation < MIN_ACTIVATION:
        return None

    return NeuronMatch(
        neuron_path=json_path,
        activation_path=activation_path,
        ancestor_count=len(ancestors),
        max_activation=max_activation,
    )


def find_activation_with_few_ancestors(training_dir: str) -> Optional[NeuronMatch]:
    """Return the first neuron that satisfies the ancestor and activation filters."""

    for json_path in _iter_neuron_json_paths(training_dir):
        match = _load_neuron_match(json_path)
        if match is not None:
            return match
    return None


def has_activation_with_few_ancestors(training_dir: str) -> bool:
    """Return ``True`` if the training directory contains a qualifying neuron."""

    return find_activation_with_few_ancestors(training_dir) is not None


TEST_ERROR_KEYS = {
    "test_error",
    "test_mse",
    "final_test_error",
    "final_test_loss",
}


CSV_TEST_ERROR_COLUMNS = (
    "test_error",
    "test_mse",
    "mse",
)


def _extract_test_errors_from_json(data: object) -> List[float]:
    values: List[float] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in TEST_ERROR_KEYS:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
            values.extend(_extract_test_errors_from_json(value))
    elif isinstance(data, list):
        for item in data:
            values.extend(_extract_test_errors_from_json(item))
    return values


def _read_test_errors_from_json(path: str) -> List[float]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []

    return _extract_test_errors_from_json(data)


def _read_test_errors_from_csv(path: str) -> List[float]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            candidate_columns = [
                column for column in reader.fieldnames if column in CSV_TEST_ERROR_COLUMNS
            ]
            if not candidate_columns:
                return []
            values = []
            for row in reader:
                for column in candidate_columns:
                    raw = row.get(column)
                    if raw is None:
                        continue
                    try:
                        values.append(float(raw))
                        break
                    except (TypeError, ValueError):
                        continue
            return values
    except OSError:
        return []


def collect_test_errors(training_dir: str) -> List[float]:
    """Gather all ``test_error`` values from files tied to ``linear_results``."""

    errors: List[float] = []
    for root, _, files in os.walk(training_dir):
        basename = os.path.basename(root)
        root_has_linear = "linear_results" in basename
        for filename in files:
            file_has_linear = "linear_results" in filename
            is_results_file = filename == "results.json"
            if not (root_has_linear or file_has_linear or is_results_file):
                continue

            path = os.path.join(root, filename)
            if filename.endswith(".json"):
                errors.extend(_read_test_errors_from_json(path))
            elif filename.endswith(".csv"):
                errors.extend(_read_test_errors_from_csv(path))

    return errors


def max_test_error(training_dir: str) -> Optional[float]:
    errors = collect_test_errors(training_dir)
    if not errors:
        return None
    return max(errors)


def read_final_test_loss(training_dir: str) -> Optional[float]:
    """Read the ``final_test_loss`` from ``results.csv`` if available."""

    candidate_paths = [
        os.path.join(training_dir, "results.csv"),
        os.path.join(training_dir, "trainer", "results.csv"),
    ]

    for results_path in candidate_paths:
        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or "final_test_loss" not in reader.fieldnames:
                    continue

                final_loss: Optional[float] = None
                for row in reader:
                    raw = row.get("final_test_loss")
                    if raw is None:
                        continue
                    try:
                        final_loss = float(raw)
                    except (TypeError, ValueError):
                        continue
                if final_loss is not None:
                    return final_loss
        except OSError:
            continue

    return None


def find_training_directories_with_criteria(root: str) -> List[str]:
    qualifying: List[str] = []
    for training_dir in iter_training_directories(root):
        neuron = find_activation_with_few_ancestors(training_dir)
        if neuron is None:
            continue

        final_test_loss = read_final_test_loss(training_dir)
        if final_test_loss is None or final_test_loss > MAX_FINAL_TEST_LOSS:
            continue

        max_error = max_test_error(training_dir)
        if max_error is None or max_error < MIN_LINEAR_LOSS:
            continue

        qualifying.append(training_dir)
        print(training_dir)
        print(
            "  activation_path:"
            f" {neuron.activation_path} (ancestors: {neuron.ancestor_count},"
            f" max_activation: {neuron.max_activation})"
        )
        print(f"  final_test_loss: {final_test_loss}")
        print(f"  max_linear_loss: {max_error}")

    return qualifying


def main() -> None:
    parser = argparse.ArgumentParser(description="Find training directories meeting analysis criteria.")
    parser.add_argument(
        "directory",
        help="Base directory to scan for training directories (identified by frobenius_drifts.json).",
    )
    args = parser.parse_args()

    qualifying = find_training_directories_with_criteria(args.directory)
    if not qualifying:
        print("No training directories meeting the criteria were found.")


if __name__ == "__main__":
    main()
