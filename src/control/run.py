from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.app import storage
from src.app.main import app
from src.data.download import file_md5
from src.data.preview import make_thumbnail
from src.model.predict import load_checkpoint
from src.paths import (
    CONTROL_DIR,
    PROJECT_ROOT,
    STAGE5_REPORTS_DIR,
    STAGE7_REPORTS_DIR,
    ensure_dirs,
    load_control_config,
    load_dataset_config,
    relative_to_root,
)

VERDICT = re.compile(r'class="verdict-name">[^<]*</span>\s*<code>([a-z_]+)</code>')
RANKED = re.compile(
    r'<span class="label">([^<]+)</span>.*?<span class="value">([\d.]+)%</span>', re.DOTALL
)
UI_CLASSES = re.compile(r"<li><code>([a-z_]+)</code>")

CORRECT_RGB = (12, 163, 12)
WRONG_RGB = (208, 59, 59)
BORDER_PX = 5
CAPTION_PX = 18
HEADER_PX = 28


@dataclass(frozen=True)
class Answer:
    class_name: str
    confidence: float
    ranked: list[str]
    low_confidence: bool


def parse_answer(page: str, ru_to_code: dict[str, str]) -> Answer:
    verdict = VERDICT.search(page)
    ranked = RANKED.findall(page)
    if verdict is None or not ranked:
        raise ValueError("на странице нет блока с результатом")
    return Answer(
        class_name=verdict.group(1),
        confidence=float(ranked[0][1]) / 100,
        ranked=[ru_to_code[label.strip()] for label, _ in ranked],
        low_confidence="Модель не уверена" in page,
    )


def ask_program(client: TestClient, path: Path, ru_to_code: dict[str, str]) -> tuple[Answer | None, str]:
    upload = client.post("/upload", files={"photo": (path.name, path.read_bytes(), "application/octet-stream")})
    if upload.status_code != 200 or "Выбранное изображение" not in upload.text:
        return None, f"загрузка отклонена ({upload.status_code})"
    response = client.post("/predict")
    client.post("/reset")
    if response.status_code != 200:
        return None, f"распознавание не удалось ({response.status_code})"
    try:
        return parse_answer(response.text, ru_to_code), ""
    except (ValueError, KeyError) as error:
        return None, f"ответ не разобран: {error}"


def wilson_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = {0.9: 1.6449, 0.95: 1.96, 0.99: 2.5758}[level]
    share = successes / total
    denominator = 1 + z * z / total
    center = (share + z * z / (2 * total)) / denominator
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def check_class(
    name: str,
    rows: pd.DataFrame,
    everything: pd.DataFrame,
    support: dict[str, bool],
    recall_floor: float,
    level: float,
) -> dict:
    total = len(rows)
    answered = int(rows["answered"].sum())
    correct = int(rows["correct"].sum())
    low, high = wilson_interval(correct, total, level)
    predicted_as = everything[everything["predicted"] == name]
    confusions = rows.loc[rows["answered"] & ~rows["correct"], "predicted"].value_counts()

    checks = {
        "declared_everywhere": all(support.values()),
        "files_intact": bool(rows["md5_ok"].all()),
        "all_answered": answered == total,
        "recognised_above_chance": low > recall_floor,
    }
    return {
        "images": total,
        "answered": answered,
        "correct": correct,
        "recall": round(correct / total, 4) if total else 0.0,
        "recall_ci": [round(low, 4), round(high, 4)],
        "precision": round(float(predicted_as["correct"].mean()), 4) if len(predicted_as) else None,
        "top2": round(float(rows["top2_hit"].mean()), 4) if total else 0.0,
        "mean_confidence": round(float(rows["confidence"].mean()), 4) if answered else None,
        "low_confidence_share": round(float(rows["low_confidence"].mean()), 4) if answered else None,
        "main_confusion": (
            {"class_name": confusions.index[0], "images": int(confusions.iloc[0])}
            if len(confusions)
            else None
        ),
        "support": support,
        "checks": checks,
        "passed": all(checks.values()),
    }


def control_sheet(name: str, rows: pd.DataFrame, thumb_px: int, cols: int) -> Image.Image:
    cell_w = thumb_px + 2 * BORDER_PX
    cell_h = cell_w + CAPTION_PX
    lines = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, HEADER_PX + max(lines, 1) * cell_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    correct = int(rows["correct"].sum())
    draw.text((6, 8), f"{name}: {correct}/{len(rows)} correct", fill=(30, 30, 30))

    for index, row in enumerate(rows.itertuples()):
        x = (index % cols) * cell_w
        y = HEADER_PX + (index // cols) * cell_h
        color = CORRECT_RGB if row.correct else WRONG_RGB
        draw.rectangle([x, y, x + cell_w - 1, y + cell_w - 1], fill=color)
        sheet.paste(make_thumbnail(PROJECT_ROOT / row.path, thumb_px), (x + BORDER_PX, y + BORDER_PX))
        mark = "OK" if row.correct else "ERR"
        caption = f"{mark}: {row.predicted} {row.confidence:.2f}" if row.answered else f"ERR: {row.error[:20]}"
        draw.text((x + 3, y + cell_w + 3), caption, fill=(40, 40, 40))
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверить все заявленные классы через программу.")
    parser.parse_args(argv)

    ensure_dirs()
    config = load_control_config()
    classes = load_dataset_config()["classes"]
    declared = list(classes.keys())
    ru_to_code = {meta["ru"]: code for code, meta in classes.items()}
    level = float(config["confidence_level"])
    floor = float(config["recall_floor"])

    missing = [name for name in declared if not (CONTROL_DIR / f"{name}.csv").exists()]
    if missing:
        print(f"Нет контрольных наборов для {missing}. Сначала: python -m src.control.build")
        return 1

    model_classes = load_checkpoint()["class_names"]
    stage5 = STAGE5_REPORTS_DIR / "predictions.csv"
    stage5_answers = (
        pd.read_csv(stage5).set_index("path")["predicted"].to_dict() if stage5.exists() else {}
    )

    records = []
    with tempfile.TemporaryDirectory() as uploads:
        storage.UPLOADS_DIR = Path(uploads)
        with TestClient(app) as client:
            ui_classes = UI_CLASSES.findall(client.get("/").text)
            for name in declared:
                control = pd.read_csv(CONTROL_DIR / f"{name}.csv")
                print(f"  {name:12} {len(control):3d} снимков ...", end="", flush=True)
                for row in control.itertuples():
                    path = PROJECT_ROOT / row.path
                    md5_ok = path.exists() and file_md5(path) == row.md5
                    answer, error = ask_program(client, path, ru_to_code) if md5_ok else (None, "файл изменён или отсутствует")
                    records.append(
                        {
                            "path": row.path,
                            "class_name": name,
                            "md5_ok": md5_ok,
                            "answered": answer is not None,
                            "predicted": answer.class_name if answer else "",
                            "confidence": round(answer.confidence, 4) if answer else float("nan"),
                            "low_confidence": bool(answer and answer.low_confidence),
                            "correct": bool(answer and answer.class_name == name),
                            "top2_hit": bool(answer and name in answer.ranked[:2]),
                            "stage5_predicted": stage5_answers.get(row.path, ""),
                            "error": error,
                        }
                    )
                done = [r for r in records if r["class_name"] == name]
                print(f" верно {sum(r['correct'] for r in done)}")

    results = pd.DataFrame(records)
    per_class = {}
    for name in declared:
        support = {
            "config": True,
            "model": name in model_classes,
            "interface": name in ui_classes,
        }
        per_class[name] = check_class(
            name, results[results["class_name"] == name], results, support, floor, level
        )

    compared = results[results["stage5_predicted"] != ""]
    agreement = (
        round(float((compared["predicted"] == compared["stage5_predicted"]).mean()), 4)
        if len(compared)
        else None
    )
    summary = {
        "control_sets": {name: relative_to_root(CONTROL_DIR / f"{name}.csv") for name in declared},
        "declared_classes": declared,
        "model_classes": list(model_classes),
        "interface_classes": ui_classes,
        "undeclared_in_model": sorted(set(model_classes) - set(declared)),
        "recall_floor": floor,
        "confidence_level": level,
        "images": len(results),
        "answered": int(results["answered"].sum()),
        "correct": int(results["correct"].sum()),
        "agreement_with_stage5": agreement,
        "per_class": per_class,
        "passed": all(item["passed"] for item in per_class.values())
        and not set(model_classes) - set(declared),
    }

    STAGE7_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(STAGE7_REPORTS_DIR / "control_results.csv", index=False)
    (STAGE7_REPORTS_DIR / "control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sheets_dir = STAGE7_REPORTS_DIR / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for name in declared:
        sheet = control_sheet(
            name,
            results[results["class_name"] == name],
            int(config["sheet_thumbnail_px"]),
            int(config["sheet_cols"]),
        )
        sheet.save(sheets_dir / f"{name}.jpg", quality=85)

    print(f"\n{'класс':12} {'верно':>7} {'recall':>7} {'95% ДИ':>13} {'top-2':>6}  итог")
    for name, item in per_class.items():
        low, high = item["recall_ci"]
        status = "PASS" if item["passed"] else "FAIL: " + ", ".join(k for k, v in item["checks"].items() if not v)
        print(
            f"{name:12} {item['correct']:3d}/{item['images']:<3d} {item['recall']:7.3f} "
            f"{low:6.3f}–{high:5.3f} {item['top2']:6.3f}  {status}"
        )
    print(f"\nОтветов программы: {summary['answered']} из {summary['images']}")
    if agreement is not None:
        print(f"Совпадение с предсказаниями этапа 5: {agreement:.3f}")
    print(f"Итог: {'все классы прошли проверку' if summary['passed'] else 'есть непрошедшие классы'}")
    print(f"Отчёт: {relative_to_root(STAGE7_REPORTS_DIR)}")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
