from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import storage
from src.app.main import app
from src.app.settings import MAX_UPLOAD_BYTES, UPLOAD_COOKIE


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def make_image(width: int = 640, height: int = 480, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 130, 70)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_index_offers_upload_when_nothing_selected(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Выберите изображение" in response.text


def test_upload_then_image_is_displayed(client):
    response = client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})
    assert response.status_code == 200
    assert "Выбранное изображение" in response.text
    assert "640 × 480" in response.text
    assert client.cookies.get(UPLOAD_COOKIE)

    image = client.get("/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(image.content)).size == (640, 480)


def test_page_shows_name_of_file_chosen_by_user(client):
    client.post(
        "/upload",
        files={"photo": ("YellowRust1003.jpg", make_image(image_format="JPEG"), "image/jpeg")},
    )
    assert "YellowRust1003.jpg" in client.get("/").text


def test_path_in_file_name_is_not_trusted(client):
    client.post(
        "/upload",
        files={"photo": ("../../../etc/passwd.png", make_image(), "image/png")},
    )
    page = client.get("/")
    assert "passwd.png" in page.text
    assert "../" not in page.text

    stored = storage.describe(client.cookies.get(UPLOAD_COOKIE))
    assert stored is not None
    assert stored.path.parent == storage.UPLOADS_DIR


def test_uploaded_image_can_be_replaced(client):
    client.post("/upload", files={"photo": ("first.png", make_image(640, 480), "image/png")})
    first_token = client.cookies.get(UPLOAD_COOKIE)

    client.post(
        "/upload",
        files={"photo": ("second.jpg", make_image(800, 600, "JPEG"), "image/jpeg")},
    )
    second_token = client.cookies.get(UPLOAD_COOKIE)

    assert second_token != first_token
    assert storage.resolve(first_token) is None

    page = client.get("/")
    assert "800 × 600" in page.text
    assert client.get("/image").headers["content-type"] == "image/jpeg"


def test_reset_removes_selected_image(client):
    client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})
    token = client.cookies.get(UPLOAD_COOKIE)

    response = client.post("/reset")
    assert response.status_code == 200
    assert "Выберите изображение" in response.text
    assert storage.resolve(token) is None
    assert client.get("/image").status_code == 404


def test_failed_replace_keeps_previous_image(client):
    client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})
    token = client.cookies.get(UPLOAD_COOKIE)
    response = client.post(
        "/upload",
        files={"photo": ("report.pdf", b"%PDF-1.4 not an image", "image/png")},
    )
    assert response.status_code == 400
    assert client.cookies.get(UPLOAD_COOKIE) == token
    assert "Выбранное изображение" in response.text
    assert storage.resolve(token) is not None


def test_non_image_file_is_rejected(client):
    response = client.post(
        "/upload",
        files={"photo": ("report.pdf", b"%PDF-1.4 not an image at all", "image/png")},
    )
    assert response.status_code == 400
    assert "не удалось прочитать" in response.text.lower()
    assert not client.cookies.get(UPLOAD_COOKIE)


def test_empty_file_is_rejected(client):
    response = client.post("/upload", files={"photo": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert "пустой" in response.text.lower()


def test_too_small_image_is_rejected(client):
    response = client.post(
        "/upload",
        files={"photo": ("tiny.png", make_image(32, 32), "image/png")},
    )
    assert response.status_code == 400
    assert "мелкое" in response.text.lower()


def test_oversized_file_is_rejected(client):
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * MAX_UPLOAD_BYTES
    response = client.post("/upload", files={"photo": ("huge.png", oversized, "image/png")})
    assert response.status_code == 400
    assert "слишком большой" in response.text.lower()


def test_gif_format_is_rejected(client):
    buffer = io.BytesIO()
    Image.new("P", (320, 240)).save(buffer, format="GIF")
    response = client.post(
        "/upload",
        files={"photo": ("anim.gif", buffer.getvalue(), "image/gif")},
    )
    assert response.status_code == 400
    assert "не поддерживается" in response.text.lower()


def test_foreign_token_gives_no_access(client):
    client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})
    client.cookies.set(UPLOAD_COOKIE, "../../etc/passwd")
    assert client.get("/image").status_code == 404


def test_predict_is_not_available_until_stage_six(client):
    client.post("/upload", files={"photo": ("leaf.png", make_image(), "image/png")})
    response = client.post("/predict")
    assert response.status_code == 501
    assert "этапе 6" in response.text


def test_health_reports_classes_and_no_model(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["classes"] == [
        "brown_rust",
        "healthy",
        "mildew",
        "septoria",
        "yellow_rust",
    ]
