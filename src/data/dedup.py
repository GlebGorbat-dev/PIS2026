"""Шаг 3.3 — удаление явных дубликатов.

Работает в три приёма:

1. Точные дубликаты — совпадение md5. Остаётся одна копия.
2. Явные near-duplicate — расстояние Хэмминга между dHash не больше
   phash_hamming_threshold. Тоже остаётся одна копия.
3. Сцены — расстояние не больше scene_group_hamming_threshold. Такие кадры
   остаются оба, но получают общий scene_group: этап разбиения не разведёт их
   по разным выборкам, иначе тест окажется завышенным.

Отдельно выделяется конфликт разметки: одинаковое изображение в двух разных
классах. Это не дубликат, а ошибка данных, и она выносится в отчёт.

Запуск:  python -m src.data.dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data.hashing import hamming, hex_to_dhash
from src.paths import (
    CLEAN_CSV,
    DUPLICATES_CSV,
    INVENTORY_CSV,
    STAGE3_REPORTS_DIR,
    ensure_dirs,
    load_dataset_config,
    relative_to_root,
)


class UnionFind:
    """Склейка изображений в группы по попарной похожести."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # сжатие пути
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            result[self.find(item)].append(item)
        return {root: sorted(members) for root, members in result.items()}


def similar_pairs(frame: pd.DataFrame, threshold: int) -> list[tuple[str, str, int]]:
    """Все пары изображений с расстоянием dHash не больше порога.

    На тысяче изображений полный перебор — это полмиллиона сравнений битов,
    то есть доли секунды, поэтому индексы не нужны.
    """
    hashes = {row.path: hex_to_dhash(row.dhash) for row in frame.itertuples()}
    pairs = []
    for left, right in combinations(hashes, 2):
        distance = hamming(hashes[left], hashes[right])
        if distance <= threshold:
            pairs.append((left, right, distance))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Найти и удалить дубликаты.")
    parser.add_argument("--inventory", type=Path, default=INVENTORY_CSV, help="путь к inventory.csv")
    args = parser.parse_args(argv)

    if not args.inventory.exists():
        print(f"Нет {args.inventory}. Сначала: python -m src.data.inventory")
        return 1

    ensure_dirs()
    config = load_dataset_config()
    dedup_config = config["deduplication"]
    exact_threshold = dedup_config["phash_hamming_threshold"]
    scene_threshold = dedup_config["scene_group_hamming_threshold"]

    inventory = pd.read_csv(args.inventory)
    usable = inventory[inventory["status"] == "ok"].copy().sort_values("path").reset_index(drop=True)
    print(f"Годных изображений на входе: {len(usable)}")

    class_by_path = dict(zip(usable["path"], usable["class_name"]))
    removal_records: list[dict] = []
    removed: set[str] = set()

    # --- 1. Точные дубликаты по md5 ---
    exact_groups = 0
    for md5_value, group in usable.groupby("md5"):
        if len(group) < 2:
            continue
        exact_groups += 1
        paths = sorted(group["path"])
        keeper, rest = paths[0], paths[1:]
        for path in rest:
            removed.add(path)
            removal_records.append(
                {
                    "path": path,
                    "kind": "exact",
                    "kept_instead": keeper,
                    "distance": 0,
                    "md5": md5_value,
                    "class_name": class_by_path[path],
                    "kept_class": class_by_path[keeper],
                }
            )

    # --- 2. Явные near-duplicate по dHash ---
    remaining = usable[~usable["path"].isin(removed)]
    near_pairs = similar_pairs(remaining, exact_threshold)
    pair_distance = {frozenset(pair): distance for *pair, distance in near_pairs}

    near_union = UnionFind(sorted(remaining["path"]))
    for left, right, _ in near_pairs:
        near_union.union(left, right)

    near_groups = 0
    for members in near_union.groups().values():
        if len(members) < 2:
            continue
        near_groups += 1
        keeper, rest = members[0], members[1:]
        for path in rest:
            removed.add(path)
            removal_records.append(
                {
                    "path": path,
                    "kind": "near",
                    "kept_instead": keeper,
                    # -1, если удаляемый кадр похож не на keeper напрямую,
                    # а на другого участника той же группы.
                    "distance": pair_distance.get(frozenset((path, keeper)), -1),
                    "md5": "",
                    "class_name": class_by_path[path],
                    "kept_class": class_by_path[keeper],
                }
            )

    # --- 3. Конфликты разметки: один и тот же кадр в разных классах ---
    conflicts = [
        record
        for record in removal_records
        if record["class_name"] != record["kept_class"]
    ]

    # --- 4. Группы сцен для честного разбиения ---
    clean = usable[~usable["path"].isin(removed)].copy().reset_index(drop=True)
    scene_pairs = similar_pairs(clean, scene_threshold)
    scene_union = UnionFind(sorted(clean["path"]))
    for left, right, _ in scene_pairs:
        # Склеиваем только внутри класса: похожие кадры разных болезней —
        # это сложные примеры, а не одна сцена.
        if class_by_path[left] == class_by_path[right]:
            scene_union.union(left, right)

    scene_root = {}
    for root, members in scene_union.groups().items():
        for member in members:
            scene_root[member] = root
    scene_ids = {root: index for index, root in enumerate(sorted(set(scene_root.values())))}
    clean["scene_group"] = clean["path"].map(lambda p: scene_ids[scene_root[p]])

    multi_scene = int((clean["scene_group"].value_counts() > 1).sum())

    duplicates = pd.DataFrame(
        removal_records,
        columns=["path", "kind", "kept_instead", "distance", "md5", "class_name", "kept_class"],
    )
    duplicates.to_csv(DUPLICATES_CSV, index=False)
    clean.to_csv(CLEAN_CSV, index=False)

    summary = {
        "input_ok_images": int(len(usable)),
        "removed_total": int(len(duplicates)),
        "removed_exact": int((duplicates["kind"] == "exact").sum()) if len(duplicates) else 0,
        "removed_near": int((duplicates["kind"] == "near").sum()) if len(duplicates) else 0,
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "phash_hamming_threshold": exact_threshold,
        "scene_group_hamming_threshold": scene_threshold,
        "clean_images": int(len(clean)),
        "clean_per_class": clean["class_name"].value_counts().sort_index().to_dict(),
        "scene_groups_total": int(clean["scene_group"].nunique()),
        "scene_groups_with_multiple_images": multi_scene,
        "label_conflicts": [
            {
                "path": c["path"],
                "class_name": c["class_name"],
                "duplicate_of": c["kept_instead"],
                "duplicate_class": c["kept_class"],
            }
            for c in conflicts
        ],
    }
    report_path = STAGE3_REPORTS_DIR / "dedup_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Удалено дубликатов: {summary['removed_total']} "
        f"(точных {summary['removed_exact']}, визуальных {summary['removed_near']})"
    )
    print(f"Осталось изображений: {summary['clean_images']}")
    print(f"Групп сцен: {summary['scene_groups_total']} "
          f"(из них с >1 изображением: {multi_scene})")
    if conflicts:
        print(f"ВНИМАНИЕ: конфликтов разметки — {len(conflicts)}, см. отчёт")
    print(f"\nСписок удалённых: {relative_to_root(DUPLICATES_CSV)}")
    print(f"Чистый набор:     {relative_to_root(CLEAN_CSV)}")
    print(f"Сводка:           {relative_to_root(report_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
