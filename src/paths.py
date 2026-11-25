from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIG = CONFIGS_DIR / "dataset.yaml"
TRAIN_CONFIG = CONFIGS_DIR / "train.yaml"
CONTROL_CONFIG = CONFIGS_DIR / "control.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "extracted"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INVENTORY_CSV = INTERIM_DIR / "inventory.csv"
DUPLICATES_CSV = INTERIM_DIR / "duplicates.csv"
CLEAN_CSV = INTERIM_DIR / "clean.csv"
SPLITS_DIR = PROCESSED_DIR / "splits"
CONTROL_DIR = DATA_DIR / "control"

UPLOADS_DIR = DATA_DIR / "uploads"
HISTORY_DIR = DATA_DIR / "history"

MODELS_DIR = PROJECT_ROOT / "models"
STAGE4_CHECKPOINT = MODELS_DIR / "stage4.pt"
CROP_GATE_WEIGHTS = MODELS_DIR / "crop_gate.pt"
BARLEY_CHECKPOINT = MODELS_DIR / "barley.pt"
MAIZE_CHECKPOINT = MODELS_DIR / "maize.pt"
CROP_CHECKPOINTS = {
    "wheat": STAGE4_CHECKPOINT,
    "barley": BARLEY_CHECKPOINT,
    "maize": MAIZE_CHECKPOINT,
}
CROP_DATASET_CONFIGS = {
    "wheat": DATASET_CONFIG,
    "barley": CONFIGS_DIR / "dataset_barley.yaml",
    "maize": CONFIGS_DIR / "dataset_maize.yaml",
}
CROP_TITLES = {
    "wheat": "Пшеница",
    "barley": "Ячмень",
    "maize": "Кукуруза",
}
CROP_GENITIVE = {
    "wheat": "пшеницы",
    "barley": "ячменя",
    "maize": "кукурузы",
}

REPORTS_DIR = PROJECT_ROOT / "reports"
STAGE2_REPORTS_DIR = REPORTS_DIR / "stage2"
STAGE3_REPORTS_DIR = REPORTS_DIR / "stage3"
STAGE4_REPORTS_DIR = REPORTS_DIR / "stage4"
STAGE5_REPORTS_DIR = REPORTS_DIR / "stage5"
STAGE7_REPORTS_DIR = REPORTS_DIR / "stage7"
STAGE9_REPORTS_DIR = REPORTS_DIR / "stage9"
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


def load_control_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONTROL_CONFIG)


def resolve_data_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    custom = (config or {}).get("paths") or {}
    defaults = {
        "raw_images": "data/raw/extracted",
        "inventory_csv": "data/interim/inventory.csv",
        "duplicates_csv": "data/interim/duplicates.csv",
        "clean_csv": "data/interim/clean.csv",
        "splits_dir": "data/processed/splits",
        "reports_dir": "reports/stage3",
    }
    resolved: dict[str, Path] = {}
    for key, default in defaults.items():
        value = custom.get(key, default)
        path = PROJECT_ROOT / value
        resolved[key] = path
        if key.endswith("_dir") or key == "raw_images":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dirs() -> None:
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        CONTROL_DIR,
        UPLOADS_DIR,
        MODELS_DIR,
        STAGE2_REPORTS_DIR,
        STAGE3_REPORTS_DIR,
        STAGE4_REPORTS_DIR,
        STAGE5_REPORTS_DIR,
        STAGE7_REPORTS_DIR,
        STAGE9_REPORTS_DIR,
        PREVIEWS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
