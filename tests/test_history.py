from __future__ import annotations

import io

from PIL import Image, ImageDraw

from src.app import history, storage


def _store(tmp_path, monkeypatch) -> storage.StoredImage:
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "history")
    image = Image.new("RGB", (320, 240), (90, 130, 70))
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 8):
        draw.line((x, 0, x, 240), fill=(40, 100, 35))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return storage.save(buffer.getvalue(), "лист.png")


def test_record_stores_required_fields_and_keeps_the_photo(tmp_path, monkeypatch):
    stored = _store(tmp_path, monkeypatch)
    reliable = history.record(
        stored,
        crop="wheat",
        class_name="yellow_rust",
        class_ru="Жёлтая ржавчина",
        confidence=0.88,
    )
    unsure = history.record(
        stored,
        crop="wheat",
        class_name="septoria",
        class_ru="Септориоз",
        confidence=0.41,
    )
    unknown = history.record(
        stored,
        crop="barley",
        class_name="ramularia",
        class_ru="Рамулярия",
        confidence=0.28,
    )

    assert reliable["reliable"] is True
    assert unsure["reliable"] is False
    assert unknown["reliable"] is False
    for entry, class_name in (
        (reliable, "yellow_rust"),
        (unsure, "septoria"),
        (unknown, "ramularia"),
    ):
        assert entry["class_name"] == class_name
        assert "T" in entry["created_at"]
        assert history.photo(entry["id"]) is not None

    storage.delete(stored.token)
    assert len(history.list_entries()) == 3
    assert history.photo(reliable["id"]).path.is_file()
    assert [entry["class_name"] for entry in history.list_entries()] == [
        "ramularia",
        "septoria",
        "yellow_rust",
    ]


def test_history_photo_rejects_a_foreign_id(tmp_path, monkeypatch):
    stored = _store(tmp_path, monkeypatch)
    history.record(
        stored,
        crop="wheat",
        class_name="healthy",
        class_ru="Здоровое растение",
        confidence=0.91,
    )
    assert history.photo("../secrets") is None
    assert history.photo("no-such-id-000000") is None
