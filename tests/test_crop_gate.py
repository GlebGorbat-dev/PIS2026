from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.model.crop_gate import CONFIDENCE_FLOOR, decide
from src.paths import CROP_GATE_WEIGHTS, PREVIEWS_DIR

CELL = 196
THUMB = 180


def test_uncertain_leaf_is_allowed():
    assert decide("wall clock", 0.02, "wheat") is None
    assert decide("banana", CONFIDENCE_FLOOR - 0.01, "wheat") is None


def test_leaf_lookalikes_are_not_rejected():
    for label in ("ear", "pineapple", "vine snake", "cardoon"):
        assert decide(label, 0.90, "wheat") is None
        assert decide(label, 0.90, "barley") is None
        assert decide(label, 0.90, "maize") is None


def test_banana_is_rejected_for_every_crop():
    expected = {
        "wheat": "На снимке не пшеница. Это не лист пшеницы, поэтому болезнь не определяется. Загрузите фотографию выбранной культуры.",
        "barley": "На снимке не ячмень. Это не лист ячменя, поэтому болезнь не определяется. Загрузите фотографию выбранной культуры.",
        "maize": "На снимке не кукуруза. Это не лист кукурузы, поэтому болезнь не определяется. Загрузите фотографию выбранной культуры.",
    }
    for crop, message in expected.items():
        assert decide("banana", 0.98, crop) == message


def test_corn_matches_maize_and_rejects_the_other_crops():
    assert decide("corn", 0.91, "maize") is None
    assert decide("corn", 0.91, "wheat") == (
        "На снимке не пшеница. Это не лист пшеницы, "
        "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
    )
    assert "кукуруза" not in decide("corn", 0.91, "barley")


def test_unknown_confident_label_does_not_name_the_object():
    message = decide("sports car", 0.88, "wheat")
    assert message == (
        "На снимке не пшеница. Это не лист пшеницы, "
        "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
    )


needs_gate = pytest.mark.skipif(not CROP_GATE_WEIGHTS.exists(), reason="нет models/crop_gate.pt")
needs_previews = pytest.mark.skipif(
    not (PREVIEWS_DIR / "healthy.jpg").exists(),
    reason="нет контактных листов",
)


@needs_gate
@needs_previews
def test_wheat_contact_sheet_leaves_pass_the_gate():
    from src.model.crop_gate import classify, decide

    rejected = []
    for name in ("healthy", "yellow_rust", "brown_rust", "septoria", "mildew"):
        sheet = Image.open(PREVIEWS_DIR / f"{name}.jpg").convert("RGB")
        for index in range(24):
            column, line = index % 6, index // 6
            x, y = column * CELL, line * CELL
            crop = sheet.crop((x, y, x + THUMB, y + THUMB))
            label, probability = classify(crop)
            message = decide(label, probability, "wheat")
            if message:
                rejected.append((name, index, label, round(probability, 3)))
    assert rejected == []


@needs_gate
def test_real_banana_photo_is_not_wheat():
    path = Path("tests/fixtures/banana.jpg")
    if not path.exists():
        pytest.skip("нет tests/fixtures/banana.jpg")
    from src.model.crop_gate import check_crop

    message = check_crop(path, "wheat")
    assert message == (
        "На снимке не пшеница. Это не лист пшеницы, "
        "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
    )
