from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"
TRAIN_CONFIG = CONFIGS_DIR / "train.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "extracted"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INVENTORY_CSV = INTERIM_DIR / "inventory.csv"
DUPLICATES_CSV = INTERIM_DIR / "duplicates.csv"
CLEAN_CSV = INTERIM_DIR / "clean.csv"
SPLITS_DIR = PROCESSED_DIR / "splits"

UPLOADS_DIR = DATA_DIR / "uploads"

MODELS_DIR = PROJECT_ROOT / "models"
STAGE4_CHECKPOINT = MODELS_DIR / "stage4.pt"

REPORTS_DIR = PROJECT_ROOT / "reports"
STAGE2_REPORTS_DIR = REPORTS_DIR / "stage2"
STAGE3_REPORTS_DIR = REPORTS_DIR / "stage3"
STAGE4_REPORTS_DIR = REPORTS_DIR / "stage4"
PREVIEWS_DIR = STAGE3_REPORTS_DIR / "previews"

APP_DIR = PROJECT_ROOT / "src" / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dataset_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DATASET_CONFIG)


def load_train_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or TRAIN_CONFIG)


def ensure_dirs() -> None:
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        UPLOADS_DIR,
        MODELS_DIR,
        STAGE2_REPORTS_DIR,
        STAGE3_REPORTS_DIR,
        STAGE4_REPORTS_DIR,
        PREVIEWS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
