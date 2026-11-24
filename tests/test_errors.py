from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.model.errors import (
    BLUR_VARIANCE,
    POOR_LIGHT_MEAN,
    build_summary,
    condition_flags,
    measure_image,
    primary_category,
)


def _features(**overrides) -> dict[str, float]:
    base = {
        "mean_luminance": 120.0,
        "dark_fraction": 0.02,
        "blur": 2000.0,
        "leaf_fraction": 0.5,
        "pustule_fraction": 0.1,
        "yellow_fraction": 0.1,
        "pale_fraction": 0.05,
        "soil_fraction": 0.05,
        "pot_fraction": 0.01,
    }
    base.update(overrides)
    return base


def test_primary_category_puts_each_error_in_one_bucket():
    assert primary_category("septoria", "yellow_rust") == "similar_diseases"
    assert primary_category("brown_rust", "septoria") == "similar_diseases"
    assert primary_category("healthy", "mildew") == "healthy_looks_like_disease"
    assert primary_category("mildew", "septoria") == "mildew_missed"
    assert primary_category("septoria", "mildew") == "confused_with_mildew"
    assert primary_category("brown_rust", "healthy") == "called_healthy"
    assert primary_category("yellow_rust", "yellow_rust") == "correct"


def test_condition_flags_follow_the_fixed_thresholds():
    assert "poor_light" in condition_flags(_features(mean_luminance=POOR_LIGHT_MEAN - 1), "septoria")
    assert "blur" in condition_flags(_features(blur=BLUR_VARIANCE - 1), "septoria")
    assert "complex_background" in condition_flags(_features(soil_fraction=0.2), "septoria")
    assert "small_symptom" in condition_flags(_features(pustule_fraction=0.01, yellow_fraction=0.01), "brown_rust")
    assert "small_symptom" not in condition_flags(_features(pustule_fraction=0.01, yellow_fraction=0.01), "healthy")
    assert condition_flags(_features(), "septoria") == []


def test_measure_image_sees_a_dark_blurry_frame(tmp_path: Path):
    path = tmp_path / "dark.png"
    Image.new("RGB", (64, 64), (12, 14, 16)).save(path)
    features = measure_image(path)
    flags = condition_flags(features, "septoria")
    assert features["mean_luminance"] < POOR_LIGHT_MEAN
    assert "poor_light" in flags


def test_summary_counts_partition_the_errors():
    frame = pd.DataFrame(
        [
            {"class_name": "septoria", "predicted": "yellow_rust", "correct": False, "confidence": 0.95, "primary": "similar_diseases", "conditions": "poor_light", "pot_fraction": 0.1, "p_septoria": 0.04, "p_yellow_rust": 0.95, "p_healthy": 0.01},
            {"class_name": "healthy", "predicted": "mildew", "correct": False, "confidence": 0.6, "primary": "healthy_looks_like_disease", "conditions": "", "pot_fraction": 0.02, "p_healthy": 0.2, "p_mildew": 0.6, "p_yellow_rust": 0.2},
            {"class_name": "septoria", "predicted": "septoria", "correct": True, "confidence": 0.8, "primary": "correct", "conditions": "poor_light", "pot_fraction": 0.05, "p_septoria": 0.8, "p_yellow_rust": 0.1, "p_healthy": 0.1},
        ]
    )
    summary = build_summary(frame, ["healthy", "yellow_rust", "septoria"])
    assert summary["errors"] == 2
    assert sum(row["count"] for row in summary["primary"]) == 2
    assert summary["conditions"]["poor_light"]["errors"] == 1
    assert summary["conditions"]["poor_light"]["correct_share"] == pytest.approx(1.0)
    assert summary["true_class_in_top2_count"] == 2
    assert summary["high_confidence_errors"] == 1
