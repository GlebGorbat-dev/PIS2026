from __future__ import annotations

import numpy as np
import pytest

from src.model.evaluate import bootstrap_intervals, compute_metrics, expected_calibration_error

NAMES = ["healthy", "yellow_rust", "brown_rust"]


def one_hot(indices: list[int], confidence: float = 0.9) -> np.ndarray:
    rest = (1.0 - confidence) / (len(NAMES) - 1)
    probabilities = np.full((len(indices), len(NAMES)), rest)
    probabilities[np.arange(len(indices)), indices] = confidence
    return probabilities


def test_perfect_predictions_give_perfect_scores():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    metrics = compute_metrics(y_true, one_hot(list(y_true)), NAMES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro"]["f1"] == 1.0
    assert metrics["mAP"] == 1.0
    assert metrics["count_error"]["mae_per_class"] == 0.0
    assert np.trace(metrics["confusion_matrix"]) == len(y_true)


def test_per_class_precision_recall_and_count_error():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = [0, 1, 1, 1, 2, 2]
    metrics = compute_metrics(y_true, one_hot(y_pred), NAMES)
    healthy = metrics["per_class"]["healthy"]
    yellow = metrics["per_class"]["yellow_rust"]
    assert healthy["precision"] == 1.0
    assert healthy["recall"] == 0.5
    assert yellow["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert yellow["recall"] == 1.0
    assert healthy["count_error"] == -1
    assert yellow["count_error"] == 1
    assert metrics["count_error"]["misassigned_share"] == pytest.approx(1 / 6, abs=1e-4)
    assert metrics["confusion_matrix"][0] == [1, 1, 0]


def test_calibration_error_is_gap_between_confidence_and_accuracy():
    y_true = np.array([0, 1, 2, 0])
    probabilities = one_hot([0, 1, 2, 1], confidence=0.8)
    assert expected_calibration_error(y_true, probabilities) == pytest.approx(0.05)


def test_bootstrap_is_reproducible_and_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, (y_true + 1) % 3)
    first = bootstrap_intervals(y_true, y_pred, 3, rounds=200, seed=1)
    second = bootstrap_intervals(y_true, y_pred, 3, rounds=200, seed=1)
    assert first == second
    low, high = first["accuracy_95ci"]
    assert low <= (y_true == y_pred).mean() <= high
