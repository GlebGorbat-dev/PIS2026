"""Журнал успешных распознаваний. Кадры, которые программа отклонила, сюда не попадают."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.app.confidence import confidence_band
from src.app.settings import ALLOWED_FORMATS, FORMAT_EXTENSIONS, LOW_CONFIDENCE_THRESHOLD
from src.app.storage import StoredImage
from src.paths import CROP_TITLES, HISTORY_DIR

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
JOURNAL_NAME = "journal.jsonl"
_LOCK = threading.Lock()

EXT_MEDIA = {extension: ALLOWED_FORMATS[fmt] for fmt, extension in FORMAT_EXTENSIONS.items()}


@dataclass(frozen=True)
class HistoryPhoto:
    path: Path
    media_type: str


def record(
    image: StoredImage,
    *,
    crop: str,
    class_name: str,
    class_ru: str,
    confidence: float,
) -> dict:
    """Сохраняет копию снимка и поля анализа. Надёжен только уверенный ответ, от 60%."""
    extension = image.path.suffix.lower()
    if extension not in EXT_MEDIA:
        raise OSError(f"Нельзя сохранить историю для формата {extension}")
    entry_id = _new_id()
    created = datetime.now().astimezone()
    image_name = f"{entry_id}{extension}"
    images = _images_dir()
    destination = images / image_name
    shutil.copyfile(image.path, destination)
    entry = {
        "id": entry_id,
        "created_at": created.isoformat(timespec="seconds"),
        "created_label": created.strftime("%d.%m.%Y %H:%M"),
        "crop": crop,
        "crop_ru": CROP_TITLES.get(crop, crop),
        "class_name": class_name,
        "class_ru": class_ru,
        "confidence": round(float(confidence), 4),
        "reliable": confidence_band(confidence) == "confident",
        "reliable_threshold": LOW_CONFIDENCE_THRESHOLD,
        "image_name": image_name,
    }
    with _LOCK:
        with _journal_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def list_entries() -> list[dict]:
    path = HISTORY_DIR / JOURNAL_NAME
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("id"):
            entries.append(entry)
    entries.reverse()
    return entries


def photo(entry_id: str) -> HistoryPhoto | None:
    if not ID_PATTERN.match(entry_id or ""):
        return None
    for entry in list_entries():
        if entry.get("id") != entry_id:
            continue
        name = str(entry.get("image_name") or "")
        path = (HISTORY_DIR / "images" / name).resolve()
        if path.parent != (HISTORY_DIR / "images").resolve() or not path.is_file():
            return None
        return HistoryPhoto(path, EXT_MEDIA.get(path.suffix.lower(), "application/octet-stream"))
    return None


def _new_id() -> str:
    for _ in range(5):
        entry_id = secrets.token_urlsafe(12)
        if ID_PATTERN.match(entry_id):
            return entry_id
    raise OSError("Не удалось выдать идентификатор записи")


def _images_dir() -> Path:
    directory = HISTORY_DIR / "images"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _journal_path() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / JOURNAL_NAME
