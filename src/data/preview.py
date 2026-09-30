from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.data.split import SPLIT_NAMES
from src.paths import (
    DUPLICATES_CSV,
    PREVIEWS_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
    STAGE3_REPORTS_DIR,
    ensure_dirs,
    load_dataset_config,
    relative_to_root,
)

Image.MAX_IMAGE_PIXELS = None

LABEL_STRIP_PX = 16
BACKGROUND = (250, 250, 250)


def make_thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.Resampling.LANCZOS)


def contact_sheet(
    frame: pd.DataFrame,
    title: str,
    thumb_px: int,
    cols: int,
) -> Image.Image:
    cell = thumb_px + LABEL_STRIP_PX
    rows = (len(frame) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, max(rows, 1) * cell), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(frame.itertuples()):
        column, line = index % cols, index // cols
        x, y = column * cell, line * cell
        try:
            sheet.paste(make_thumbnail(PROJECT_ROOT / row.path, thumb_px), (x, y))
        except (OSError, ValueError):
            draw.rectangle([x, y, x + thumb_px, y + thumb_px], fill=(220, 120, 120))
            draw.text((x + 4, y + 4), "ошибка чтения", fill=(255, 255, 255))
        caption = getattr(row, "split", "") or ""
        draw.text(
            (x + 3, y + thumb_px + 3),
            f"{caption} {Path(row.path).name[:18]}",
            fill=(60, 60, 60),
        )

    print(f"  {title}: {len(frame)} миниатюр, сетка {cols}x{rows}")
    return sheet


def build_html(
    per_class: dict[str, dict],
    sheets: dict[str, str],
    split_counts: pd.DataFrame,
    duplicates: pd.DataFrame,
    config: dict,
) -> str:
    source = config["source"]
    classes = config["classes"]

    rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(classes[name]['ru'])}</td>"
        f"<td>{stats['total']}</td><td>{stats['train']}</td>"
        f"<td>{stats['val']}</td><td>{stats['test']}</td></tr>"
        for name, stats in sorted(per_class.items())
    )

    galleries = "\n".join(
        f"<section><h3>{html.escape(name)} — {html.escape(classes[name]['ru'])}</h3>"
        f"<img src='{html.escape(rel)}' alt='{html.escape(name)}'></section>"
        for name, rel in sorted(sheets.items())
    )

    dup_rows = "\n".join(
        f"<tr><td>{html.escape(str(r.path))}</td><td>{html.escape(str(r.kind))}</td>"
        f"<td>{html.escape(str(r.kept_instead))}</td><td>{r.distance}</td></tr>"
        for r in duplicates.head(200).itertuples()
    ) or "<tr><td colspan='4'>дубликатов не найдено</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Этап 3 — визуальная проверка датасета</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; margin-bottom: 28px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: left; }}
  th {{ background: #eef3ea; }}
  section {{ margin-bottom: 36px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; }}
  details {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Этап 3 — датасет заболеваний пшеницы</h1>
<div class="meta">
  Источник: <a href="{html.escape(source['record_url'])}">{html.escape(source['name'])}</a>,
  DOI {html.escape(source['doi'])}, лицензия {html.escape(source['license'])}.<br>
  Всего изображений после удаления дубликатов: <b>{int(split_counts.values.sum())}</b>.
</div>

<h2>Классы и выборки</h2>
<table>
<tr><th>Класс</th><th>Название</th><th>Всего</th><th>train</th><th>val</th><th>test</th></tr>
{rows}
</table>

<h2>Контактные листы</h2>
<p>Под каждой миниатюрой — выборка и имя файла. Ищем: чужой класс в сетке,
кадры без листа, сильный пересвет или смаз.</p>
{galleries}

<details>
<summary>Удалённые дубликаты ({len(duplicates)})</summary>
<table>
<tr><th>Удалён</th><th>Тип</th><th>Оставлен вместо него</th><th>dHash-расстояние</th></tr>
{dup_rows}
</table>
</details>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать визуальный отчёт по датасету.")
    parser.add_argument("--samples-per-class", type=int, default=None)
    args = parser.parse_args(argv)

    all_csv = SPLITS_DIR / "all.csv"
    if not all_csv.exists():
        print("Нет data/processed/splits/all.csv. Сначала: python -m src.data.split")
        return 1

    ensure_dirs()
    config = load_dataset_config()
    preview_config = config["preview"]
    samples = args.samples_per_class or int(preview_config["samples_per_class"])
    thumb_px = int(preview_config["thumbnail_px"])
    cols = int(preview_config["grid_cols"])
    seed = int(config["split"]["random_seed"])

    frame = pd.read_csv(all_csv)
    duplicates = (
        pd.read_csv(DUPLICATES_CSV)
        if DUPLICATES_CSV.exists()
        else pd.DataFrame(columns=["path", "kind", "kept_instead", "distance"])
    )

    print("Сборка контактных листов...")
    sheets: dict[str, str] = {}
    per_class: dict[str, dict] = {}
    for class_name, class_frame in frame.groupby("class_name"):
        per_split = max(1, samples // len(SPLIT_NAMES))
        chunks = []
        for split_name in SPLIT_NAMES:
            subset = class_frame[class_frame["split"] == split_name]
            if len(subset):
                chunks.append(subset.sample(n=min(len(subset), per_split), random_state=seed))
        sample = pd.concat(chunks).sort_values(["split", "path"])
        sheet = contact_sheet(sample, class_name, thumb_px, cols)
        sheet_path = PREVIEWS_DIR / f"{class_name}.jpg"
        sheet.save(sheet_path, quality=88, optimize=True)
        sheets[class_name] = sheet_path.relative_to(STAGE3_REPORTS_DIR).as_posix()

        counts = class_frame["split"].value_counts()
        per_class[class_name] = {
            "total": int(len(class_frame)),
            "train": int(counts.get("train", 0)),
            "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)),
        }

    split_counts = pd.crosstab(frame["class_name"], frame["split"])
    html_path = STAGE3_REPORTS_DIR / "dataset_review.html"
    html_path.write_text(
        build_html(per_class, sheets, split_counts, duplicates, config), encoding="utf-8"
    )

    stats_path = STAGE3_REPORTS_DIR / "preview_summary.json"
    stats_path.write_text(
        json.dumps(
            {"samples_per_class": samples, "per_class": per_class, "sheets": sheets},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nКонтактные листы: {relative_to_root(PREVIEWS_DIR)}")
    print(f"HTML-отчёт:       {relative_to_root(html_path)}")
    print("Откройте HTML в браузере и просмотрите каждый класс.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
