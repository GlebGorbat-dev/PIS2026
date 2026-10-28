from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import main, storage
from src.app.main import app
from src.control.build import build_control_sets
from src.control.run import ask_program, check_class, parse_answer, wilson_interval
from src.model.predict import Prediction

NAMES = ["healthy", "yellow_rust", "brown_rust", "septoria", "mildew"]
RU_TO_CODE = {meta["ru"]: code for code, meta in main.CLASSES.items()}
FULL_SUPPORT = {"config": True, "model": True, "interface": True}


def split_frame(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for name, count in counts.items():
        for i in range(count):
            rows.append(
                {"path": f"data/{name}{i}.png", "class_name": name, "split": "test", "scene_group": i,
                 "width": 600, "height": 800, "md5": f"{name}{i}", "dhash": "0"}
            )
    return pd.DataFrame(rows)


def results_frame(name: str, correct: int, total: int, *, answered: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_name": [name] * total,
            "md5_ok": [True] * total,
            "answered": [answered] * total,
            "predicted": [name if i < correct else "yellow_rust" for i in range(total)] if answered else [""] * total,
            "correct": [answered and i < correct for i in range(total)],
            "top2_hit": [True] * total,
            "confidence": [0.8] * total,
            "low_confidence": [False] * total,
        }
    )


def test_wilson_interval_matches_known_values():
    low, high = wilson_interval(9, 19)
    assert low == pytest.approx(0.273, abs=1e-3)
    assert high == pytest.approx(0.683, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_every_declared_class_gets_its_own_set():
    sets = build_control_sets(split_frame({name: 16 for name in NAMES}), NAMES, min_images=15)
    assert list(sets) == NAMES
    assert all(len(frame) == 16 for frame in sets.values())
    assert all(set(frame["class_name"]) == {name} for name, frame in sets.items())


def test_too_small_set_is_rejected():
    counts = {name: 16 for name in NAMES} | {"mildew": 3}
    with pytest.raises(ValueError, match="mildew"):
        build_control_sets(split_frame(counts), NAMES, min_images=15)


def test_undeclared_class_in_split_is_rejected():
    counts = {name: 16 for name in NAMES} | {"stem_rust": 16}
    with pytest.raises(ValueError, match="stem_rust"):
        build_control_sets(split_frame(counts), NAMES, min_images=15)


def test_class_recognised_above_chance_passes():
    rows = results_frame("brown_rust", correct=9, total=19)
    result = check_class("brown_rust", rows, rows, FULL_SUPPORT, recall_floor=0.2, level=0.95)
    assert result["passed"] is True
    assert result["recall"] == pytest.approx(9 / 19, abs=1e-4)


def test_class_at_chance_level_fails():
    rows = results_frame("brown_rust", correct=4, total=19)
    result = check_class("brown_rust", rows, rows, FULL_SUPPORT, recall_floor=0.2, level=0.95)
    assert result["passed"] is False
    assert result["checks"]["recognised_above_chance"] is False


def test_class_missing_from_interface_fails():
    rows = results_frame("mildew", correct=20, total=20)
    support = FULL_SUPPORT | {"interface": False}
    result = check_class("mildew", rows, rows, support, recall_floor=0.2, level=0.95)
    assert result["passed"] is False
    assert result["checks"]["declared_everywhere"] is False


def test_unanswered_images_fail_the_class():
    rows = results_frame("healthy", correct=0, total=18, answered=False)
    result = check_class("healthy", rows, rows, FULL_SUPPORT, recall_floor=0.2, level=0.95)
    assert result["checks"]["all_answered"] is False
    assert result["passed"] is False


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def test_answer_is_read_back_from_the_page(client, monkeypatch, tmp_path):
    probabilities = {"healthy": 0.05, "yellow_rust": 0.40, "brown_rust": 0.37, "septoria": 0.10, "mildew": 0.08}
    monkeypatch.setattr(
        main,
        "run_model",
        lambda path: Prediction("yellow_rust", "Жёлтая ржавчина", 0.40, probabilities),
    )
    image_path = tmp_path / "leaf.png"
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 130, 70)).save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())

    answer, error = ask_program(client, image_path, RU_TO_CODE)
    assert error == ""
    assert answer.class_name == "yellow_rust"
    assert answer.confidence == pytest.approx(0.40)
    assert answer.ranked[:2] == ["yellow_rust", "brown_rust"]
    assert answer.low_confidence is True


def test_page_without_result_is_not_parsed():
    with pytest.raises(ValueError):
        parse_answer("<html><body>Выберите изображение</body></html>", RU_TO_CODE)
