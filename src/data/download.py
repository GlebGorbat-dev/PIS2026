"""Шаг 3.1 — получение исходных данных.

Скачивает архив с Zenodo, сверяет md5 с опубликованным в записи датасета
и распаковывает его в data/raw/extracted. Архив и изображения не попадают
в git, поэтому этот скрипт — единственный способ воспроизвести набор данных.

Запуск:  python -m src.data.download
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.paths import RAW_DIR, RAW_IMAGES_DIR, ensure_dirs, load_dataset_config

CHUNK = 1 << 20  # 1 МиБ


def file_md5(path: Path, chunk_size: int = CHUNK) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Качает файл, показывая прогресс в одну строку."""
    print(f"Загрузка: {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while block := response.read(CHUNK):
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r  {done / 2**20:8.1f} / {total / 2**20:.1f} МиБ ({pct:5.1f}%)", end="")
    print()


def extract(archive: Path, destination: Path) -> int:
    """Распаковывает архив, пропуская служебный мусор macOS/Windows."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename)
            if member.is_dir():
                continue
            if "__MACOSX" in member.filename or name.name.startswith("._"):
                continue
            if name.name in {".DS_Store", "Thumbs.db"}:
                continue
            zf.extract(member, destination)
            extracted += 1
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачать и распаковать датасет.")
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="не проверять контрольную сумму архива (для отладки)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    config = load_dataset_config()
    source = config["source"]

    archive = RAW_DIR / source["archive_name"]
    if not archive.exists():
        download(source["download_url"], archive)
    else:
        print(f"Архив уже на диске: {archive} ({archive.stat().st_size / 2**20:.1f} МиБ)")

    if args.skip_md5:
        print("Проверка md5 пропущена.")
    else:
        print("Проверка md5...")
        actual = file_md5(archive)
        expected = source["archive_md5"]
        if actual != expected:
            print(f"ОШИБКА: md5 не совпал.\n  ожидался: {expected}\n  получен:  {actual}")
            return 1
        print(f"  md5 совпал: {actual}")

    print(f"Распаковка в {RAW_IMAGES_DIR}...")
    count = extract(archive, RAW_IMAGES_DIR)
    print(f"Готово: распаковано файлов — {count}")

    entries = sorted(RAW_IMAGES_DIR.iterdir())
    directories = [p.name for p in entries if p.is_dir()]
    print(f"Верхний уровень: каталогов — {len(directories)}, файлов — {len(entries) - len(directories)}")
    if directories:
        print(f"  каталоги: {directories}")
    else:
        print("  архив плоский, класс закодирован в имени файла")
        print(f"  примеры: {[p.name for p in entries[:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
