from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import main, storage
from src.app.main import app
from src.model.predict import Prediction
from src.paths import RAW_IMAGES_DIR, STAGE4_CHECKPOINT

NAMES = ["healthy", "yellow_rust", "brown_rust", "septoria", "mildew"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def make_image(width: int = 640, height: int = 480) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 130, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_prediction(probabilities: dict[str, float]) -> Prediction:
    best = max(probabilities, key=probabilities.get)
    ru = {name: meta["ru"] for name, meta in main.CLASSES.items()}
    return Prediction(
        class_name=best,
        class_ru=ru[best],
        confidence=probabilities[best],
        probabilities=probabilities,
    )


def upload(client) -> None:
    client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})


def test_predict_shows_class_and_confidence(client, monkeypatch):
    prediction = fake_prediction(
        {"healthy": 0.02, "yellow_rust": 0.87, "brown_rust": 0.05, "septoria": 0.04, "mildew": 0.02}
    )
    monkeypatch.setattr(main, "run_model", lambda path: prediction)
    upload(client)

    response = client.post("/predict")
    assert response.status_code == 200
    assert "Результат распознавания" in response.text
    assert "Жёлтая ржавчина" in response.text
    assert "Уверенность модели: <strong>87%</strong>" in response.text
    assert "Модель не уверена" not in response.text
    assert "Определить ещё раз" in response.text
    assert "Выбранное изображение" in response.text


def test_predict_lists_all_classes_from_most_likely(client, monkeypatch):
    prediction = fake_prediction(
        {"healthy": 0.10, "yellow_rust": 0.05, "brown_rust": 0.15, "septoria": 0.62, "mildew": 0.08}
    )
    monkeypatch.setattr(main, "run_model", lambda path: prediction)
    upload(client)

    text = client.post("/predict").text
    block = text[text.index('class="probabilities"') :]
    expected_order = ("septoria", "brown_rust", "healthy", "mildew", "yellow_rust")
    positions = [block.index(main.CLASSES[name]["ru"]) for name in expected_order]
    assert positions == sorted(positions)
    assert "62.0%" in block


def test_low_confidence_warns_and_names_runner_up(client, monkeypatch):
    prediction = fake_prediction(
        {"healthy": 0.05, "yellow_rust": 0.41, "brown_rust": 0.38, "septoria": 0.10, "mildew": 0.06}
    )
    monkeypatch.setattr(main, "run_model", lambda path: prediction)
    upload(client)

    text = client.post("/predict").text
    assert "Модель не уверена" in text
    assert "бурая ржавчина (38%)" in text


def test_predict_without_image_asks_to_choose_one(client):
    response = client.post("/predict")
    assert response.status_code == 400
    assert "Сначала выберите изображение" in response.text


def test_missing_weights_give_clear_error(client, monkeypatch):
    def missing(path):
        raise FileNotFoundError("нет весов")

    monkeypatch.setattr(main, "run_model", missing)
    upload(client)
    response = client.post("/predict")
    assert response.status_code == 503
    assert "Модель недоступна" in response.text
    assert "Выбранное изображение" in response.text


def test_unreadable_image_gives_clear_error(client, monkeypatch):
    def broken(path):
        raise OSError("truncated")

    monkeypatch.setattr(main, "run_model", broken)
    upload(client)
    response = client.post("/predict")
    assert response.status_code == 422
    assert "Не удалось обработать изображение" in response.text


def test_result_disappears_when_image_is_replaced(client, monkeypatch):
    prediction = fake_prediction(
        {"healthy": 0.9, "yellow_rust": 0.04, "brown_rust": 0.02, "septoria": 0.02, "mildew": 0.02}
    )
    monkeypatch.setattr(main, "run_model", lambda path: prediction)
    upload(client)
    assert "Результат распознавания" in client.post("/predict").text

    page = client.post("/upload", files={"photo": ("other.png", make_image(800, 600), "image/png")})
    assert "Результат распознавания" not in page.text
    assert "Определить заболевание" in page.text


needs_weights = pytest.mark.skipif(not STAGE4_CHECKPOINT.exists(), reason="нет models/stage4.pt")


@needs_weights
def test_real_model_answers_through_the_page(client):
    upload(client)
    response = client.post("/predict")
    assert response.status_code == 200
    assert "Уверенность модели" in response.text
    assert any(meta["ru"] in response.text for meta in main.CLASSES.values())
    assert client.get("/health").json()["model_loaded"] is True


@needs_weights
@pytest.mark.skipif(not (RAW_IMAGES_DIR / "YellowRust1010.png").exists(), reason="датасет не скачан")
def test_real_model_recognises_dataset_photo(client):
    raw = (RAW_IMAGES_DIR / "YellowRust1010.png").read_bytes()
    client.post("/upload", files={"photo": ("YellowRust1010.png", raw, "image/png")})
    text = client.post("/predict").text
    verdict = text[text.index('class="verdict-name"') :][:80]
    assert "Жёлтая ржавчина" in verdict
