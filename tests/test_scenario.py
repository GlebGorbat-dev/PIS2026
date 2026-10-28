from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import main, storage
from src.app.main import app
from src.app.settings import UPLOAD_COOKIE
from src.model.predict import Prediction
from src.paths import RAW_IMAGES_DIR, STAGE4_CHECKPOINT

CONFIDENT = {"healthy": 0.02, "yellow_rust": 0.03, "brown_rust": 0.88, "septoria": 0.05, "mildew": 0.02}
UNSURE = {"healthy": 0.05, "yellow_rust": 0.41, "brown_rust": 0.38, "septoria": 0.10, "mildew": 0.06}
HEALTHY = {"healthy": 0.95, "yellow_rust": 0.01, "brown_rust": 0.01, "septoria": 0.02, "mildew": 0.01}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def model_returns(monkeypatch, probabilities: dict[str, float]) -> None:
    best = max(probabilities, key=probabilities.get)
    prediction = Prediction(best, main.CLASSES[best]["ru"], probabilities[best], probabilities)
    monkeypatch.setattr(main, "run_model", lambda path: prediction)


def photo(width: int = 640, height: int = 480) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 130, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def choose_and_recognise(client, name: str = "leaf.png", raw: bytes | None = None):
    return client.post(
        "/upload",
        files={"photo": (name, raw or photo(), "image/png")},
        data={"then": "predict"},
    )


def test_one_step_choose_and_recognise(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    response = choose_and_recognise(client)
    assert response.status_code == 200
    assert client.cookies.get(UPLOAD_COOKIE)
    assert "Результат распознавания" in response.text
    assert "Бурая ржавчина" in response.text
    assert "Выбранное изображение" in response.text


def test_plain_upload_shows_photo_without_result(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    response = client.post("/upload", files={"photo": ("leaf.png", photo(), "image/png")})
    assert "Выбранное изображение" in response.text
    assert "Результат распознавания" not in response.text


def test_result_survives_page_refresh(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    choose_and_recognise(client)
    monkeypatch.setattr(main, "run_model", lambda path: pytest.fail("модель не должна вызываться повторно"))
    page = client.get("/")
    assert "Результат распознавания" in page.text
    assert "88%" in page.text


def test_new_photo_drops_old_result(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    choose_and_recognise(client)
    page = client.post("/upload", files={"photo": ("other.png", photo(800, 600), "image/png")})
    assert "Результат распознавания" not in page.text
    assert "800 × 600" in page.text


def test_reset_clears_photo_and_result(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    choose_and_recognise(client)
    token = client.cookies.get(UPLOAD_COOKIE)
    page = client.post("/reset")
    assert "Выберите изображение" in page.text
    assert storage.load_prediction(token) is None


def test_confident_answer_explains_its_symptoms(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    text = choose_and_recognise(client).text
    block = text[text.index('class="symptoms"') :]
    assert "Как это выглядит" in block
    assert main.CLASSES["brown_rust"]["symptoms"][:30] in block
    assert main.CLASSES["yellow_rust"]["symptoms"][:30] not in block
    assert "покажите растение агроному" in text


def test_unsure_answer_offers_both_variants_to_compare(client, monkeypatch):
    model_returns(monkeypatch, UNSURE)
    text = choose_and_recognise(client).text
    block = text[text.index('class="symptoms"') :]
    assert "Сравните с фотографией" in block
    assert main.CLASSES["yellow_rust"]["symptoms"][:30] in block
    assert main.CLASSES["brown_rust"]["symptoms"][:30] in block


def test_healthy_answer_gives_no_treatment_advice(client, monkeypatch):
    model_returns(monkeypatch, HEALTHY)
    text = choose_and_recognise(client).text
    assert "Признаков болезни модель не нашла" in text
    assert "решение об обработке" not in text


def test_one_step_without_model_keeps_photo_and_explains(client, monkeypatch):
    def missing(path):
        raise FileNotFoundError("нет весов")

    monkeypatch.setattr(main, "run_model", missing)
    response = choose_and_recognise(client)
    assert response.status_code == 503
    assert "Модель недоступна" in response.text
    assert "Выбранное изображение" in response.text
    assert client.cookies.get(UPLOAD_COOKIE)


def test_bad_file_in_one_step_is_rejected_before_model(client, monkeypatch):
    monkeypatch.setattr(main, "run_model", lambda path: pytest.fail("модель не должна вызываться"))
    response = client.post(
        "/upload",
        files={"photo": ("report.pdf", b"%PDF-1.4 not an image", "image/png")},
        data={"then": "predict"},
    )
    assert response.status_code == 400
    assert "не удалось прочитать" in response.text.lower()


def test_prediction_is_not_stored_for_unknown_token(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    storage.save_prediction("A" * 32, {"class_name": "healthy"})
    assert list(tmp_path.iterdir()) == []
    assert storage.load_prediction("../../etc/passwd") is None


def test_broken_meta_file_means_no_result(client, monkeypatch):
    model_returns(monkeypatch, CONFIDENT)
    choose_and_recognise(client)
    token = client.cookies.get(UPLOAD_COOKIE)
    (storage.UPLOADS_DIR / f"{token}.json").write_text("{не json", encoding="utf-8")
    page = client.get("/")
    assert page.status_code == 200
    assert "Выбранное изображение" in page.text
    assert "Результат распознавания" not in page.text


@pytest.mark.skipif(not STAGE4_CHECKPOINT.exists(), reason="нет models/stage4.pt")
@pytest.mark.skipif(not (RAW_IMAGES_DIR / "BrownRust476.png").exists(), reason="датасет не скачан")
def test_full_scenario_with_real_model(client):
    assert "Выберите изображение" in client.get("/").text

    raw = (RAW_IMAGES_DIR / "BrownRust476.png").read_bytes()
    page = choose_and_recognise(client, "BrownRust476.png", raw).text
    verdict = page[page.index('class="verdict-name"') :][:80]
    assert "Бурая ржавчина" in verdict
    assert "BrownRust476.png" in page

    refreshed = client.get("/").text
    assert "Бурая ржавчина" in refreshed[refreshed.index('class="verdict-name"') :][:80]

    healthy = (RAW_IMAGES_DIR / "Healthy1878.png").read_bytes()
    page = choose_and_recognise(client, "Healthy1878.png", healthy).text
    verdict = page[page.index('class="verdict-name"') :][:80]
    assert "Здоровое растение" in verdict

    assert "Выберите изображение" in client.post("/reset").text
