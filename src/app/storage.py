from __future__ import annotations

import io
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.app.settings import (
    ALLOWED_FORMATS,
    FORMAT_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    MIN_SIDE_PX,
)
from src.paths import UPLOADS_DIR

TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class UploadError(Exception):
    pass


@dataclass(frozen=True)
class StoredImage:
    token: str
    path: Path
    original_name: str
    image_format: str
    media_type: str
    width: int
    height: int
    size_bytes: int

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def _describe_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} МБ"
    return f"{size_bytes / 1024:.0f} КБ"


def validate(raw: bytes, original_name: str) -> tuple[str, int, int]:
    if not raw:
        raise UploadError("Файл пустой. Выберите фотографию.")

    if len(raw) > MAX_UPLOAD_BYTES:
        limit = _describe_size(MAX_UPLOAD_BYTES)
        actual = _describe_size(len(raw))
        raise UploadError(f"Файл слишком большой: {actual}, допустимо до {limit}.")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format or ""
            width, height = image.size
            image.verify()
    except Image.DecompressionBombError:
        raise UploadError(
            "Изображение слишком большое по числу пикселей и может исчерпать память."
        ) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise UploadError(
            f"Не удалось прочитать «{original_name}» как изображение. "
            f"Допустимые форматы: {', '.join(sorted(ALLOWED_FORMATS))}."
        ) from None

    if image_format not in ALLOWED_FORMATS:
        raise UploadError(
            f"Формат {image_format or 'неизвестный'} не поддерживается. "
            f"Допустимые форматы: {', '.join(sorted(ALLOWED_FORMATS))}."
        )

    if width * height > MAX_IMAGE_PIXELS:
        raise UploadError(
            f"Изображение {width}x{height} слишком большое: "
            f"допустимо до {MAX_IMAGE_PIXELS // 1_000_000} мегапикселей."
        )

    if min(width, height) < MIN_SIDE_PX:
        raise UploadError(
            f"Изображение {width}x{height} слишком мелкое: "
            f"меньшая сторона должна быть не менее {MIN_SIDE_PX} пикселей."
        )

    return image_format, width, height


def save(raw: bytes, original_name: str) -> StoredImage:
    image_format, width, height = validate(raw, original_name)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    path = UPLOADS_DIR / f"{token}{FORMAT_EXTENSIONS[image_format]}"
    path.write_bytes(raw)

    stored = StoredImage(
        token=token,
        path=path,
        original_name=Path(original_name).name or "снимок",
        image_format=image_format,
        media_type=ALLOWED_FORMATS[image_format],
        width=width,
        height=height,
        size_bytes=len(raw),
    )
    _write_meta(stored)
    return stored


def _meta_path(token: str) -> Path:
    return UPLOADS_DIR / f"{token}.json"


def _write_meta(stored: StoredImage) -> None:
    _meta_path(stored.token).write_text(
        json.dumps({"original_name": stored.original_name}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_original_name(token: str, fallback: str) -> str:
    path = _meta_path(token)
    if not path.is_file():
        return fallback
    try:
        name = json.loads(path.read_text(encoding="utf-8")).get("original_name")
    except (json.JSONDecodeError, OSError):
        return fallback
    return Path(str(name)).name if name else fallback


def resolve(token: str | None) -> Path | None:
    if not token or not TOKEN_PATTERN.match(token):
        return None
    for extension in FORMAT_EXTENSIONS.values():
        candidate = UPLOADS_DIR / f"{token}{extension}"
        if candidate.is_file():
            return candidate
    return None


def describe(token: str | None) -> StoredImage | None:
    path = resolve(token)
    if path is None:
        return None

    try:
        with Image.open(path) as image:
            image_format = image.format or ""
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    return StoredImage(
        token=str(token),
        path=path,
        original_name=_read_original_name(str(token), path.name),
        image_format=image_format,
        media_type=ALLOWED_FORMATS.get(image_format, "application/octet-stream"),
        width=width,
        height=height,
        size_bytes=path.stat().st_size,
    )


def delete(token: str | None) -> bool:
    path = resolve(token)
    if path is None:
        return False
    path.unlink(missing_ok=True)
    _meta_path(str(token)).unlink(missing_ok=True)
    return True


def human_size(size_bytes: int) -> str:
    return _describe_size(size_bytes)
