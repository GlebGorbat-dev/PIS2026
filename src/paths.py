"""Единая точка правды про расположение файлов проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "extracted"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INVENTORY_CSV = INTERIM_DIR / "inventory.csv"
DUPLICATES_CSV = INTERIM_DIR / "duplicates.csv"
CLEAN_CSV = INTERIM_DIR / "clean.csv"
SPLITS_DIR = PROCESSED_DIR / "splits"

REPORTS_DIR = PROJECT_ROOT / "reports"
STAGE3_REPORTS_DIR = REPORTS_DIR / "stage3"
PREVIEWS_DIR = STAGE3_REPORTS_DIR / "previews"


def load_dataset_config(path: Path | None = None) -> dict[str, Any]:
    """Читает configs/dataset.yaml."""
    config_path = path or DATASET_CONFIG
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    """Создаёт каталоги, в которые пишут скрипты этапа 3."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        STAGE3_REPORTS_DIR,
        PREVIEWS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    """Путь относительно корня проекта — чтобы CSV не зависели от машины."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
