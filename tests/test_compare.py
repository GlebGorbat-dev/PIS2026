from __future__ import annotations

import pandas as pd
import pytest

from src.model.compare import compare_frames


def test_compare_counts_the_yellow_rust_confusions():
    baseline = pd.DataFrame(
        [
            {"path": "a", "class_name": "septoria", "predicted": "yellow_rust", "correct": False},
            {"path": "b", "class_name": "septoria", "predicted": "septoria", "correct": True},
            {"path": "c", "class_name": "brown_rust", "predicted": "yellow_rust", "correct": False},
        ]
    )
    candidate = pd.DataFrame(
        [
            {"path": "a", "class_name": "septoria", "predicted": "septoria", "correct": True},
            {"path": "b", "class_name": "septoria", "predicted": "septoria", "correct": True},
            {"path": "c", "class_name": "brown_rust", "predicted": "brown_rust", "correct": True},
        ]
    )
    report = compare_frames(baseline, candidate)
    assert report["same_paths"] is True
    assert report["baseline_errors"] == 2
    assert report["candidate_errors"] == 0
    pairs = {row["pair"]: row for row in report["focus_pairs"]}
    assert pairs["septoria -> yellow_rust"]["delta"] == -1
    assert pairs["brown_rust -> yellow_rust"]["delta"] == -1
    septoria = next(row for row in report["recall"] if row["class_name"] == "septoria")
    assert septoria["baseline"] == 0.5
    assert septoria["candidate"] == 1.0
    paired = report["paired_accuracy"]
    assert paired["fixed"] == 2
    assert paired["regressed"] == 0
    assert paired["delta"] == pytest.approx(2 / 3, abs=1e-4)
    assert paired["delta_95ci"][0] <= paired["delta"] <= paired["delta_95ci"][1]
