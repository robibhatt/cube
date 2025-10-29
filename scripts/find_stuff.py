import argparse
import csv
import json
import os
from typing import Iterable, List, Optional


MAX_ANCESTOR_COUNT = 30
"""Maximum allowed number of ancestors for an activation to qualify."""

MIN_LINEAR_LOSS = 0.0000005
"""Minimum linear loss (``test_error``) value required to flag a directory."""

MAX_FINAL_TEST_LOSS = 10.0
"""Maximum allowed ``final_test_loss`` found in ``trainer/results.csv``."""


def iter_training_directories(root: str) -> Iterable[str]:
    """Yield directories containing ``frobenius_drifts.json``."""

    for dirpath, _, filenames in os.walk(root):
        if "frobenius_drifts.json" in filenames:
            yield dirpath


def has_activation_with_few_ancestors(training_dir: str) -> bool:
    """Return ``True`` if an ancestor with <= ``MAX_ANCESTOR_COUNT`` nodes exists."""

    parent_dir = os.path.dirname(training_dir)

    try:
        candidate_dirs = [
            os.path.join(parent_dir, entry)
            for entry in os.listdir(parent_dir)
            if entry.startswith("mlp_graph")
        ]
    except OSError:
        return False

    if not candidate_dirs:
        return False

    for mlp_graph_dir in candidate_dirs:
        if not os.path.isdir(mlp_graph_dir):
            continue

        for root, _, files in os.walk(mlp_graph_dir):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue

                ancestors = data.get("ancestors")
                if isinstance(ancestors, list) and len(ancestors) <= MAX_ANCESTOR_COUNT:
                    return True

    return False


TEST_ERROR_KEYS = {
    "test_error",
    "final_test_error",
    "final_test_loss",
}


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
            if reader.fieldnames is None or "test_error" not in reader.fieldnames:
                return []
            values = []
            for row in reader:
                raw = row.get("test_error")
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
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
    """Read the ``final_test_loss`` from ``trainer/results.csv`` if available."""

    results_path = os.path.join(training_dir, "trainer", "results.csv")
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "final_test_loss" not in reader.fieldnames:
                return None

            final_loss: Optional[float] = None
            for row in reader:
                raw = row.get("final_test_loss")
                if raw is None:
                    continue
                try:
                    final_loss = float(raw)
                except (TypeError, ValueError):
                    continue
            return final_loss
    except OSError:
        return None


def find_training_directories_with_criteria(root: str) -> List[str]:
    qualifying: List[str] = []
    for training_dir in iter_training_directories(root):
        if not has_activation_with_few_ancestors(training_dir):
            continue

        final_test_loss = read_final_test_loss(training_dir)
        if final_test_loss is None or final_test_loss > MAX_FINAL_TEST_LOSS:
            continue

        max_error = max_test_error(training_dir)
        if max_error is not None and max_error > MIN_LINEAR_LOSS:
            qualifying.append(training_dir)
            print(training_dir)

    return qualifying


def main() -> None:
    parser = argparse.ArgumentParser(description="Find training directories meeting analysis criteria.")
    parser.add_argument(
        "directory",
        help="Base directory to scan for training directories (identified by frobenius_drifts.json).",
    )
    args = parser.parse_args()

    qualifying = find_training_directories_with_criteria(args.directory)
    if qualifying:
        for path in qualifying:
            print(path)
    else:
        print("No training directories meeting the criteria were found.")


if __name__ == "__main__":
    main()
