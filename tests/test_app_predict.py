from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.app import history, main, storage
from src.app.main import app
from src.model.predict import Prediction
from src.paths import RAW_IMAGES_DIR, STAGE4_CHECKPOINT

NAMES = ["healthy", "yellow_rust", "brown_rust", "septoria", "mildew"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "history")
    with TestClient(app) as test_client:
        yield test_client


def make_image(width: int = 640, height: int = 480) -> bytes:
    image = Image.new("RGB", (width, height), (90, 130, 70))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 8):
        draw.line((x, 0, x, height), fill=(40 + (x % 50), 100, 35))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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


def test_banana_photo_is_rejected_on_the_page(client):
    path = Path("tests/fixtures/banana.jpg")
    if not path.exists():
        pytest.skip("нет tests/fixtures/banana.jpg")
    raw = path.read_bytes()
    response = client.post(
        "/upload",
        data={"then": "predict", "crop": "wheat"},
        files={"photo": ("Bananas.jpg", raw, "image/jpeg")},
    )
    assert response.status_code == 422
    assert (
        "На снимке не пшеница. Это не лист пшеницы, "
        "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
    ) in response.text
    assert "Результат распознавания" not in response.text
    assert "Уверенность модели" not in response.text
    assert "Bananas.jpg" in response.text
    assert history.list_entries() == []


def test_foreign_subject_is_rejected_before_the_disease_model(client, monkeypatch):
    def forbidden(path, **kwargs):
        raise AssertionError("модель болезни не должна вызываться")

    monkeypatch.setattr(
        main,
        "check_crop",
        lambda path, crop: (
            "На снимке не пшеница. Это не лист пшеницы, "
            "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
        ),
    )
    monkeypatch.setattr(main, "run_model", forbidden)
    upload(client)

    response = client.post("/predict")
    assert response.status_code == 422
    assert "Это не лист пшеницы, поэтому болезнь не определяется" in response.text
    assert "Результат распознавания" not in response.text
    assert "Выбранное изображение" in response.text


def test_analyses_remain_after_the_upload_is_removed(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "run_model",
        lambda path, **kwargs: fake_prediction(
            {"healthy": 0.02, "yellow_rust": 0.87, "brown_rust": 0.05, "septoria": 0.04, "mildew": 0.02}
        ),
    )
    upload(client)
    first = client.post("/predict")
    assert "Надёжно" in first.text

    monkeypatch.setattr(
        main,
        "run_model",
        lambda path, **kwargs: fake_prediction(
            {"healthy": 0.10, "yellow_rust": 0.05, "brown_rust": 0.15, "septoria": 0.50, "mildew": 0.20}
        ),
    )
    upload(client)
    second = client.post("/predict")
    assert "Септориоз" in second.text
    assert "Ненадёжно" in second.text
    assert "Жёлтая ржавчина" in second.text

    page = client.post("/reset")
    assert "Выбранное изображение" not in page.text
    assert "Жёлтая ржавчина" in page.text
    assert "Септориоз" in page.text
    entries = history.list_entries()
    assert len(entries) == 2
    assert entries[0]["reliable"] is False
    assert entries[1]["reliable"] is True
    for entry in entries:
        photo = client.get(f"/history/{entry['id']}/photo")
        assert photo.status_code == 200
        assert photo.headers["content-type"].startswith("image/")


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
def test_crop_switch_shows_maize_classes(client):
    page = client.post("/crop", data={"crop": "maize"}, follow_redirects=True)
    assert page.status_code == 200
    assert "Серая пятнистость" in page.text
    assert "Повреждение совкой" in page.text
    assert "Распознаваемые состояния: кукуруза" in page.text
    assert "Септориоз" not in page.text


def test_maize_predict_uses_maize_labels(client, monkeypatch):
    prediction = Prediction(
        "gray_spot",
        "Серая пятнистость",
        0.91,
        {"healthy": 0.03, "blight": 0.04, "gray_spot": 0.91, "fall_armyworm": 0.02},
    )
    monkeypatch.setattr(main, "run_model", lambda path, **kwargs: prediction)
    client.post("/crop", data={"crop": "maize"})
    upload(client)
    text = client.post("/predict", data={"crop": "maize"}).text
    assert "Серая пятнистость" in text
    assert "Уверенность модели: <strong>91%</strong>" in text
    assert "Повреждение совкой" in text
    assert "Жёлтая ржавчина" not in text


def test_barley_predict_uses_barley_labels(client, monkeypatch):
    prediction = Prediction(
        "ramularia",
        "Рамулярия",
        0.77,
        {"healthy": 0.10, "brown_rust": 0.13, "ramularia": 0.77},
    )
    monkeypatch.setattr(main, "run_model", lambda path, **kwargs: prediction)
    client.post("/crop", data={"crop": "barley"})
    upload(client)
    text = client.post("/predict", data={"crop": "barley"}).text
    assert "Рамулярия" in text
    assert "Бурая ржавчина ячменя" in text
    assert "Мучнистая роса" not in text


def test_changing_crop_hides_previous_result(client, monkeypatch):
    prediction = fake_prediction(
        {"healthy": 0.02, "yellow_rust": 0.87, "brown_rust": 0.05, "septoria": 0.04, "mildew": 0.02}
    )
    monkeypatch.setattr(main, "run_model", lambda path, **kwargs: prediction)
    upload(client)
    assert "Жёлтая ржавчина" in client.post("/predict").text
    page = client.post("/crop", data={"crop": "barley"}, follow_redirects=True)
    assert "Результат распознавания" not in page.text
    assert "Выбранное изображение" in page.text
    assert "Распознаваемые состояния: ячмень" in page.text


def test_real_model_recognises_dataset_photo(client):
    raw = (RAW_IMAGES_DIR / "YellowRust1010.png").read_bytes()
    client.post("/upload", files={"photo": ("YellowRust1010.png", raw, "image/png")})
    text = client.post("/predict").text
    verdict = text[text.index('class="verdict-name"') :][:80]
    assert "Жёлтая ржавчина" in verdict
