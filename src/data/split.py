from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.paths import (
    CLEAN_CSV,
    SPLITS_DIR,
    STAGE3_REPORTS_DIR,
    ensure_dirs,
    load_dataset_config,
    relative_to_root,
)

SPLIT_NAMES = ("train", "val", "test")


def assign_groups(
    group_sizes: dict[int, int],
    ratios: dict[str, float],
    order: list[int],
) -> dict[int, str]:
    total = sum(group_sizes.values())
    targets = {name: ratios[name] * total for name in SPLIT_NAMES}
    current = {name: 0 for name in SPLIT_NAMES}
    assignment: dict[int, str] = {}

    for group_id in order:
        size = group_sizes[group_id]
        chosen = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - current[name], -SPLIT_NAMES.index(name)),
        )
        assignment[group_id] = chosen
        current[chosen] += size

    return assignment


def split_frame(clean: pd.DataFrame, config: dict) -> pd.DataFrame:
    split_config = config["split"]
    ratios = {name: float(split_config[name]) for name in SPLIT_NAMES}
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Доли выборок должны давать 1.0, получено {total_ratio}")

    seed = int(split_config["random_seed"])
    use_groups = bool(split_config.get("group_near_duplicates", True))

    result = clean.copy()
    if not use_groups:
        result["scene_group"] = range(len(result))

    result["split"] = ""
    for class_name, class_frame in result.groupby("class_name"):
        group_sizes = class_frame["scene_group"].value_counts().to_dict()
        order = (
            pd.Series(group_sizes)
            .sample(frac=1.0, random_state=seed)
            .sort_values(ascending=False, kind="stable")
            .index.tolist()
        )
        assignment = assign_groups(group_sizes, ratios, order)
        mask = result["class_name"] == class_name
        result.loc[mask, "split"] = result.loc[mask, "scene_group"].map(assignment)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Разбить датасет на 3 выборки.")
    parser.add_argument("--clean", type=Path, default=CLEAN_CSV, help="путь к clean.csv")
    args = parser.parse_args(argv)

    if not args.clean.exists():
        print(f"Нет {args.clean}. Сначала: python -m src.data.dedup")
        return 1

    ensure_dirs()
    config = load_dataset_config()
    clean = pd.read_csv(args.clean)

    result = split_frame(clean, config)

    columns = ["path", "class_name", "split", "scene_group", "width", "height", "md5", "dhash"]
    for name in SPLIT_NAMES:
        subset = result.loc[result["split"] == name, columns].sort_values("path")
        subset.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    result[columns].sort_values("path").to_csv(SPLITS_DIR / "all.csv", index=False)

    counts = pd.crosstab(result["class_name"], result["split"])
    counts = counts.reindex(columns=list(SPLIT_NAMES), fill_value=0).sort_index()

    leaks = (
        result.groupby("scene_group")["split"].nunique().loc[lambda s: s > 1].index.tolist()
    )

    summary = {
        "clean_images": int(len(result)),
        "target_ratios": {name: config["split"][name] for name in SPLIT_NAMES},
        "random_seed": config["split"]["random_seed"],
        "group_near_duplicates": config["split"]["group_near_duplicates"],
        "counts_per_split": {
            name: int((result["split"] == name).sum()) for name in SPLIT_NAMES
        },
        "actual_ratios": {
            name: round(float((result["split"] == name).mean()), 4) for name in SPLIT_NAMES
        },
        "counts_per_class_split": counts.to_dict(orient="index"),
        "class_share_per_split": {
            name: (
                result.loc[result["split"] == name, "class_name"]
                .value_counts(normalize=True)
                .round(4)
                .sort_index()
                .to_dict()
            )
            for name in SPLIT_NAMES
        },
        "scene_groups_leaked_across_splits": leaks,
        "classes_missing_in_split": {
            name: sorted(
                set(result["class_name"].unique())
                - set(result.loc[result["split"] == name, "class_name"].unique())
            )
            for name in SPLIT_NAMES
        },
    }
    report_path = STAGE3_REPORTS_DIR / "split_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Всего изображений: {summary['clean_images']}")
    print(f"По выборкам: {summary['counts_per_split']}")
    print(f"Фактические доли: {summary['actual_ratios']}")
    print(f"\nКлассы по выборкам:\n{counts}")
    if leaks:
        print(f"\nОШИБКА: сцены разъехались по выборкам: {leaks}")
    empty = {k: v for k, v in summary["classes_missing_in_split"].items() if v}
    if empty:
        print(f"\nВНИМАНИЕ: в выборках не хватает классов: {empty}")
    print(f"\nВыборки: {relative_to_root(SPLITS_DIR)}")
    print(f"Сводка:  {relative_to_root(report_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
