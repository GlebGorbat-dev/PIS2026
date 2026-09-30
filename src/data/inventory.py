"""Шаг 3.2 — инвентаризация распакованных данных.

Проходит по data/raw/extracted, сопоставляет имена папок с каноническими
классами из configs/dataset.yaml, открывает каждое изображение и складывает
всё найденное в data/interim/inventory.csv.

Отдельно отвечает на вопрос «проверены классы»: любая метка, не описанная
в конфиге, попадает в отчёт как unmapped и требует ручного решения.

Запуск:  python -m src.data.inventory
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.hashing import dhash, dhash_to_hex, file_md5
from src.paths import (
    INVENTORY_CSV,
    RAW_IMAGES_DIR,
    STAGE3_REPORTS_DIR,
    ensure_dirs,
    load_dataset_config,
    relative_to_root,
)

# Pillow предупреждает о больших файлах; снимки с телефонов легально большие.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class ImageRecord:
    path: str
    class_name: str
    label_source: str
    file_name: str
    width: int
    height: int
    mode: str
    image_format: str
    file_bytes: int
    md5: str
    dhash: str
    status: str
    reject_reason: str


def build_alias_map(classes_config: dict) -> dict[str, str]:
    """Строит отображение «нормализованный псевдоним -> канонический класс»."""
    alias_map: dict[str, str] = {}
    for canonical, meta in classes_config.items():
        aliases = set(meta.get("aliases", [])) | {canonical}
        for alias in aliases:
            alias_map[normalize(alias)] = canonical
    return alias_map


def normalize(text: str) -> str:
    """Приводит метку к виду, по которому можно искать псевдоним."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def filename_prefix(file_name: str) -> str:
    """Буквенный префикс имени файла: "YellowRust1003.png" -> "YellowRust"."""
    stem = Path(file_name).stem
    match = re.match(r"^([A-Za-z][A-Za-z _-]*)", stem)
    return match.group(1).strip(" _-") if match else ""


def resolve_class(path: Path, root: Path, alias_map: dict[str, str]) -> tuple[str, str]:
    """Определяет класс изображения по его пути внутри распакованного архива.

    В архиве Zenodo метка хранится в префиксе имени файла (BrownRust1018.png),
    но встречаются и раскладки с папкой на класс, поэтому проверяются оба
    варианта: сначала каталоги от ближнего к дальнему, затем имя файла.

    Возвращает (канонический класс или "", источник метки).
    """
    relative_parts = path.relative_to(root).parts[:-1]
    for part in reversed(relative_parts):
        canonical = alias_map.get(normalize(part))
        if canonical:
            return canonical, part

    prefix = filename_prefix(path.name)
    canonical = alias_map.get(normalize(prefix))
    if canonical:
        return canonical, prefix

    return "", prefix or (relative_parts[-1] if relative_parts else "")


def scan(root: Path, config: dict) -> tuple[list[ImageRecord], Counter]:
    alias_map = build_alias_map(config["classes"])
    filters = config["quality_filters"]
    allowed = {ext.lower() for ext in filters["allowed_extensions"]}

    records: list[ImageRecord] = []
    unmapped: Counter = Counter()

    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        if path.name.startswith("._") or path.name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.suffix.lower() not in allowed:
            continue

        class_name, label_source = resolve_class(path, root, alias_map)
        if not class_name:
            unmapped[label_source] += 1
            continue

        file_bytes = path.stat().st_size
        try:
            with Image.open(path) as image:
                image.load()  # ловит обрезанные JPEG, а не только битый заголовок
                width, height = image.size
                mode = image.mode
                image_format = image.format or ""
                digest_visual = dhash_to_hex(dhash(image))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            records.append(
                ImageRecord(
                    path=relative_to_root(path),
                    class_name=class_name,
                    label_source=label_source,
                    file_name=path.name,
                    width=0,
                    height=0,
                    mode="",
                    image_format="",
                    file_bytes=file_bytes,
                    md5="",
                    dhash="",
                    status="rejected",
                    reject_reason=f"unreadable: {type(error).__name__}",
                )
            )
            continue

        reason = ""
        if file_bytes < filters["min_file_bytes"]:
            reason = f"file_too_small: {file_bytes} B"
        elif min(width, height) < filters["min_side_px"]:
            reason = f"resolution_too_low: {width}x{height}"

        records.append(
            ImageRecord(
                path=relative_to_root(path),
                class_name=class_name,
                label_source=label_source,
                file_name=path.name,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
                file_bytes=file_bytes,
                md5=file_md5(path),
                dhash=digest_visual,
                status="rejected" if reason else "ok",
                reject_reason=reason,
            )
        )

    return records, unmapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать инвентаризацию датасета.")
    parser.add_argument(
        "--root",
        type=Path,
        default=RAW_IMAGES_DIR,
        help="каталог с распакованными изображениями",
    )
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"Нет каталога {args.root}. Сначала выполните: python -m src.data.download")
        return 1

    ensure_dirs()
    config = load_dataset_config()

    print(f"Сканирование {args.root}...")
    records, unmapped = scan(args.root, config)
    if not records:
        print("Изображения не найдены — проверьте структуру архива.")
        return 1

    frame = pd.DataFrame([asdict(r) for r in records])
    frame.to_csv(INVENTORY_CSV, index=False)

    declared = set(config["classes"])
    found = set(frame["class_name"].unique())
    missing = sorted(declared - found)

    by_class = (
        frame.groupby("class_name")
        .agg(
            total=("path", "count"),
            ok=("status", lambda s: int((s == "ok").sum())),
            rejected=("status", lambda s: int((s == "rejected").sum())),
        )
        .sort_index()
    )

    summary = {
        "root": relative_to_root(args.root),
        "files_indexed": int(len(frame)),
        "declared_image_count": config["source"]["declared_image_count"],
        "ok": int((frame["status"] == "ok").sum()),
        "rejected": int((frame["status"] == "rejected").sum()),
        "reject_reasons": frame.loc[frame["status"] == "rejected", "reject_reason"]
        .value_counts()
        .to_dict(),
        "classes_declared": sorted(declared),
        "classes_found": sorted(found),
        "classes_missing": missing,
        "unmapped_labels": dict(unmapped),
        "per_class": by_class.to_dict(orient="index"),
        "label_source_to_class": frame.groupby("label_source")["class_name"]
        .agg(lambda s: sorted(set(s)))
        .to_dict(),
        "resolution": {
            "width_min": int(frame.loc[frame["status"] == "ok", "width"].min()),
            "width_max": int(frame.loc[frame["status"] == "ok", "width"].max()),
            "height_min": int(frame.loc[frame["status"] == "ok", "height"].min()),
            "height_max": int(frame.loc[frame["status"] == "ok", "height"].max()),
        },
        "modes": frame.loc[frame["status"] == "ok", "mode"].value_counts().to_dict(),
        "formats": frame.loc[frame["status"] == "ok", "image_format"].value_counts().to_dict(),
    }

    report_path = STAGE3_REPORTS_DIR / "inventory_summary.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nПроиндексировано файлов: {summary['files_indexed']}")
    print(f"  годных: {summary['ok']}, отбраковано: {summary['rejected']}")
    print(f"\nРаспределение по классам:\n{by_class}")
    if unmapped:
        print(f"\nВНИМАНИЕ: метки без класса (нужно решение вручную): {dict(unmapped)}")
    if missing:
        print(f"\nВНИМАНИЕ: в данных нет изображений для классов: {missing}")
    print(f"\nИнвентаризация: {relative_to_root(INVENTORY_CSV)}")
    print(f"Сводка:          {relative_to_root(report_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
