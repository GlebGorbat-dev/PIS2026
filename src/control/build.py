from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.paths import (
    CONTROL_DIR,
    SPLITS_DIR,
    STAGE7_REPORTS_DIR,
    ensure_dirs,
    load_control_config,
    load_dataset_config,
    relative_to_root,
)

COLUMNS = ["path", "class_name", "md5", "width", "height", "scene_group"]


def build_control_sets(
    split: pd.DataFrame,
    class_names: list[str],
    min_images: int,
) -> dict[str, pd.DataFrame]:
    unknown = sorted(set(split["class_name"]) - set(class_names))
    if unknown:
        raise ValueError(f"В выборке есть незаявленные классы: {unknown}")

    sets: dict[str, pd.DataFrame] = {}
    for name in class_names:
        frame = split[split["class_name"] == name][COLUMNS].sort_values("path")
        if len(frame) < min_images:
            raise ValueError(
                f"Для класса {name} всего {len(frame)} снимков, нужно не меньше {min_images}"
            )
        sets[name] = frame.reset_index(drop=True)
    return sets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сформировать контрольный набор для каждого класса.")
    parser.parse_args(argv)

    ensure_dirs()
    config = load_control_config()
    class_names = list(load_dataset_config()["classes"].keys())
    split_csv = SPLITS_DIR / f"{config['source_split']}.csv"
    if not split_csv.exists():
        print(f"Нет {split_csv}. Сначала: python -m src.data.split")
        return 1

    try:
        sets = build_control_sets(
            pd.read_csv(split_csv), class_names, int(config["min_images_per_class"])
        )
    except ValueError as error:
        print(error)
        return 1

    summary = {
        "source_split": relative_to_root(split_csv),
        "classes": {},
    }
    for name, frame in sets.items():
        path = CONTROL_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        summary["classes"][name] = {
            "file": relative_to_root(path),
            "images": len(frame),
            "scenes": int(frame["scene_group"].nunique()),
        }
        print(f"  {name:12} {len(frame):3d} снимков -> {relative_to_root(path)}")

    STAGE7_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE7_REPORTS_DIR / "control_sets.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Всего: {sum(len(f) for f in sets.values())} снимков в {len(sets)} наборах")
    return 0


if __name__ == "__main__":
    sys.exit(main())
