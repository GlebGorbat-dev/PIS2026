from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.paths import PROJECT_ROOT, STAGE5_REPORTS_DIR, STAGE9_REPORTS_DIR, ensure_dirs, relative_to_root

# Пороги подобраны по распределению яркости, резкости и цвета на test
# и зафиксированы здесь, чтобы разбор этапа 9 повторялся той же командой.
POOR_LIGHT_MEAN = 95.0
POOR_LIGHT_DARK = 0.20
BLUR_VARIANCE = 700.0
SOIL_FRACTION = 0.18
LITTLE_LEAF = 0.20
SOIL_WITH_LITTLE_LEAF = 0.12
SMALL_PUSTULE = 0.035
SMALL_YELLOW = 0.07

DISEASES = ("yellow_rust", "brown_rust", "septoria")
SIMILAR_DISEASES = frozenset(DISEASES)

PRIMARY_ORDER = (
    "similar_diseases",
    "healthy_looks_like_disease",
    "confused_with_mildew",
    "called_healthy",
    "mildew_missed",
    "other",
)
PRIMARY_RU = {
    "similar_diseases": "Похожие заболевания",
    "healthy_looks_like_disease": "Здоровый лист похож на болезнь",
    "confused_with_mildew": "Принято за мучнистую росу",
    "called_healthy": "Болезнь принята за здоровый лист",
    "mildew_missed": "Мучнистая роса не распознана",
    "other": "Прочее",
}
CONDITION_RU = {
    "poor_light": "Плохое освещение",
    "blur": "Смазанный кадр",
    "complex_background": "Сложный фон",
    "small_symptom": "Симптом занимает малую часть кадра",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
BAR = "#2a78d6"


def measure_image(path: Path) -> dict[str, float]:
    """Грубые признаки кадра: яркость, резкость, доля листа, пятен, почвы и бледных пикселей."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((224, 224))
    pixels = np.asarray(rgb).astype(np.float32)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    padded = np.pad(luminance, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
    )
    peak = pixels.max(axis=2)
    saturation = np.where(peak > 1, (peak - pixels.min(axis=2)) / np.maximum(peak, 1), 0)
    leaf = (green > red + 8) & (green > blue + 8) & (luminance > 30) & (luminance < 210)
    pustule = (red > 130) & (red > green + 15) & (red > blue + 25) & (green > 40) & (green < 190) & (luminance < 200)
    yellow = (red > 140) & (green > 120) & (blue < 110) & (red + green > 2.2 * blue) & (luminance > 80) & (luminance < 210)
    pale = (luminance > 165) & (saturation < 0.22)
    soil = (
        (red > 70)
        & (green > 40)
        & (blue < red - 5)
        & (green < red + 15)
        & (saturation > 0.15)
        & (saturation < 0.55)
        & (luminance > 40)
        & (luminance < 150)
        & ~leaf
        & ~pustule
    )
    return {
        "mean_luminance": round(float(luminance.mean()), 2),
        "dark_fraction": round(float((luminance < 40).mean()), 4),
        "blur": round(float(laplacian.var()), 1),
        "leaf_fraction": round(float(leaf.mean()), 4),
        "pustule_fraction": round(float(pustule.mean()), 4),
        "yellow_fraction": round(float(yellow.mean()), 4),
        "pale_fraction": round(float(pale.mean()), 4),
        "soil_fraction": round(float(soil.mean()), 4),
        "pot_fraction": round(float((luminance < 35).mean()), 4),
    }


def primary_category(true_name: str, predicted: str) -> str:
    """Одна главная причина ошибки. Верный ответ помечается как correct."""
    if true_name == predicted:
        return "correct"
    if true_name == "healthy":
        return "healthy_looks_like_disease"
    if true_name == "mildew":
        return "mildew_missed"
    if predicted == "mildew":
        return "confused_with_mildew"
    if predicted == "healthy":
        return "called_healthy"
    if true_name in SIMILAR_DISEASES and predicted in SIMILAR_DISEASES:
        return "similar_diseases"
    return "other"


def condition_flags(features: dict[str, float], true_name: str) -> list[str]:
    """Условия съёмки. Считаются и для верных кадров, чтобы сравнить доли."""
    flags: list[str] = []
    if features["mean_luminance"] < POOR_LIGHT_MEAN or features["dark_fraction"] >= POOR_LIGHT_DARK:
        flags.append("poor_light")
    if features["blur"] < BLUR_VARIANCE:
        flags.append("blur")
    little_leaf = features["leaf_fraction"] < LITTLE_LEAF and features["soil_fraction"] >= SOIL_WITH_LITTLE_LEAF
    if features["soil_fraction"] >= SOIL_FRACTION or little_leaf:
        flags.append("complex_background")
    sparse = features["pustule_fraction"] < SMALL_PUSTULE and features["yellow_fraction"] < SMALL_YELLOW
    if true_name != "healthy" and sparse:
        flags.append("small_symptom")
    return flags


def true_class_rank(row: pd.Series, class_names: list[str]) -> int | None:
    columns = [f"p_{name}" for name in class_names]
    if any(column not in row.index for column in columns):
        return None
    order = sorted(class_names, key=lambda name: float(row[f"p_{name}"]), reverse=True)
    return order.index(row["class_name"]) + 1


def attach_analysis(frame: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    measured = []
    for row in frame.itertuples(index=False):
        features = measure_image(image_root / row.path)
        features["primary"] = primary_category(row.class_name, row.predicted)
        features["conditions"] = ",".join(condition_flags(features, row.class_name))
        measured.append(features)
    extra = pd.DataFrame(measured)
    return pd.concat([frame.reset_index(drop=True), extra], axis=1)


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return round(float(mask.mean()), 4)


def build_summary(frame: pd.DataFrame, class_names: list[str]) -> dict:
    errors = frame.loc[~frame["correct"]].copy()
    correct = frame.loc[frame["correct"]]
    pair_counts = Counter(f"{row.class_name} -> {row.predicted}" for row in errors.itertuples())
    primary_counts = {name: int((errors["primary"] == name).sum()) for name in PRIMARY_ORDER}
    primary_counts = {name: count for name, count in primary_counts.items() if count}

    conditions = {}
    for name in CONDITION_RU:
        on_error = errors["conditions"].fillna("").map(lambda value, key=name: key in value.split(",") if value else False)
        on_correct = correct["conditions"].fillna("").map(
            lambda value, key=name: key in value.split(",") if value else False
        )
        conditions[name] = {
            "title": CONDITION_RU[name],
            "errors": int(on_error.sum()),
            "error_share": _rate(on_error),
            "correct_share": _rate(on_correct),
        }

    ranks = errors.apply(lambda row: true_class_rank(row, class_names), axis=1) if len(errors) else pd.Series(dtype=float)
    rank_known = ranks.dropna()
    by_class = (
        frame.groupby("class_name")["pot_fraction"].median().round(4).to_dict() if "pot_fraction" in frame else {}
    )
    return {
        "images": int(len(frame)),
        "errors": int(len(errors)),
        "accuracy": round(float(frame["correct"].mean()), 4) if len(frame) else None,
        "primary": [
            {"id": name, "title": PRIMARY_RU[name], "count": count, "share_of_errors": round(count / len(errors), 4)}
            for name, count in primary_counts.items()
        ],
        "pairs": [{"pair": pair, "count": count} for pair, count in pair_counts.most_common()],
        "conditions": conditions,
        "true_class_in_top2": round(float((rank_known <= 2).mean()), 4) if len(rank_known) else None,
        "true_class_in_top2_count": int((rank_known <= 2).sum()) if len(rank_known) else 0,
        "high_confidence_errors": int((errors["confidence"] >= 0.9).sum()) if "confidence" in errors else None,
        "pot_fraction_median_by_class": by_class,
    }


def save_category_chart(summary: dict, path: Path) -> None:
    rows = summary["primary"]
    labels = [row["title"] for row in rows][::-1]
    counts = [row["count"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(labels, counts, color=BAR)
    for index, count in enumerate(counts):
        ax.text(count + 0.15, index, str(count), va="center", color=INK, fontsize=9)
    ax.set_xlabel("ошибок на test", color=INK)
    ax.set_title("Главные категории ошибок", color=INK)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def save_contact_sheet(frame: pd.DataFrame, image_root: Path, path: Path, thumb: int = 220, cols: int = 4) -> None:
    if frame.empty:
        return
    rows = (len(frame) + cols - 1) // cols
    cell_h = thumb + 34
    sheet = Image.new("RGB", (cols * thumb, rows * cell_h), (252, 252, 251))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(frame.itertuples()):
        image = Image.open(image_root / row.path).convert("RGB")
        image.thumbnail((thumb - 8, thumb - 8))
        column, line = index % cols, index // cols
        x, y = column * thumb + 4, line * cell_h + 4
        sheet.paste(image, (x, y))
        draw.text((column * thumb + 6, line * cell_h + thumb - 2), f"{row.class_name} → {row.predicted}", fill=(20, 20, 20))
        draw.text((column * thumb + 6, line * cell_h + thumb + 14), Path(row.path).name[:26], fill=(90, 90, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Разобрать ошибки модели по категориям на фиксированном test.")
    parser.add_argument("--predictions", type=Path, default=STAGE5_REPORTS_DIR / "predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=STAGE9_REPORTS_DIR)
    args = parser.parse_args(argv)

    if not args.predictions.exists():
        print(f"Нет {args.predictions}. Сначала: python -m src.model.evaluate")
        return 1

    ensure_dirs()
    frame = pd.read_csv(args.predictions)
    required = {"path", "class_name", "predicted", "correct"}
    missing = required - set(frame.columns)
    if missing:
        print(f"В {args.predictions} нет колонок: {', '.join(sorted(missing))}")
        return 1
    frame["correct"] = frame["correct"].astype(bool)

    print(f"Предсказания: {relative_to_root(args.predictions)}  снимков: {len(frame)}")
    analyzed = attach_analysis(frame, PROJECT_ROOT)
    class_names = [column[2:] for column in analyzed.columns if column.startswith("p_")]
    summary = build_summary(analyzed, class_names)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "path",
        "class_name",
        "predicted",
        "confidence",
        "correct",
        "primary",
        "conditions",
        "mean_luminance",
        "dark_fraction",
        "blur",
        "leaf_fraction",
        "pustule_fraction",
        "yellow_fraction",
        "pale_fraction",
        "soil_fraction",
        "pot_fraction",
    ]
    analyzed[keep].to_csv(out_dir / "conditions.csv", index=False)
    errors = analyzed.loc[~analyzed["correct"]].sort_values(["primary", "class_name", "predicted"])
    errors[keep].to_csv(out_dir / "errors.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_category_chart(summary, out_dir / "categories.png")
    sheets = out_dir / "sheets"
    for name in PRIMARY_ORDER:
        group = errors.loc[errors["primary"] == name]
        if group.empty:
            continue
        save_contact_sheet(group, PROJECT_ROOT, sheets / f"{name}.jpg")

    print(f"Ошибок: {summary['errors']} из {summary['images']}")
    for row in summary["primary"]:
        print(f"  {row['count']:3d}  {row['title']}")
    print(f"Истинный класс в топ-2: {summary['true_class_in_top2_count']} из {summary['errors']}")
    print(f"Отчёт: {relative_to_root(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
