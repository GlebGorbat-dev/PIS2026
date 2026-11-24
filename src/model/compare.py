from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import STAGE5_REPORTS_DIR, ensure_dirs, relative_to_root

FOCUS_PAIRS = (
    ("septoria", "yellow_rust"),
    ("brown_rust", "yellow_rust"),
    ("septoria", "brown_rust"),
    ("yellow_rust", "septoria"),
    ("brown_rust", "septoria"),
)


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["correct"] = frame["correct"].astype(bool)
    return frame


def recall_by_class(frame: pd.DataFrame) -> dict[str, float | None]:
    recalls = {}
    for name, group in frame.groupby("class_name"):
        recalls[str(name)] = round(float(group["correct"].mean()), 4) if len(group) else None
    return recalls


def pair_count(frame: pd.DataFrame, true_name: str, predicted: str) -> int:
    return int(((frame["class_name"] == true_name) & (frame["predicted"] == predicted)).sum())


def paired_accuracy_delta(
    baseline: pd.DataFrame, candidate: pd.DataFrame, rounds: int = 1000, seed: int = 42
) -> dict:
    """Разница accuracy на одних и тех же кадрах. Интервал — bootstrap по снимкам."""
    merged = baseline.merge(candidate, on="path", suffixes=("_base", "_new"))
    delta = merged["correct_new"].astype(int).to_numpy() - merged["correct_base"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    samples = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(rounds)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "fixed": int((~merged["correct_base"] & merged["correct_new"]).sum()),
        "regressed": int((merged["correct_base"] & ~merged["correct_new"]).sum()),
        "delta": round(float(delta.mean()), 4),
        "delta_95ci": [round(float(low), 4), round(float(high), 4)],
        "rounds": rounds,
        "seed": seed,
    }


def compare_frames(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    names = sorted(set(baseline["class_name"]) | set(candidate["class_name"]))
    base_recall = recall_by_class(baseline)
    new_recall = recall_by_class(candidate)
    pairs = []
    for true_name, predicted in FOCUS_PAIRS:
        before = pair_count(baseline, true_name, predicted)
        after = pair_count(candidate, true_name, predicted)
        pairs.append(
            {
                "pair": f"{true_name} -> {predicted}",
                "baseline": before,
                "candidate": after,
                "delta": after - before,
            }
        )
    return {
        "baseline_images": int(len(baseline)),
        "candidate_images": int(len(candidate)),
        "same_paths": bool(baseline["path"].tolist() == candidate["path"].tolist()) if "path" in baseline and "path" in candidate else None,
        "baseline_accuracy": round(float(baseline["correct"].mean()), 4),
        "candidate_accuracy": round(float(candidate["correct"].mean()), 4),
        "baseline_errors": int((~baseline["correct"]).sum()),
        "candidate_errors": int((~candidate["correct"]).sum()),
        "recall": [
            {
                "class_name": name,
                "baseline": base_recall.get(name),
                "candidate": new_recall.get(name),
            }
            for name in names
        ],
        "focus_pairs": pairs,
        "paired_accuracy": paired_accuracy_delta(baseline, candidate) if "path" in baseline.columns else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнить две модели на одном и том же наборе предсказаний.")
    parser.add_argument("--baseline", type=Path, default=STAGE5_REPORTS_DIR / "predictions.csv")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.baseline.exists() or not args.candidate.exists():
        print("Нет одного из файлов предсказаний.")
        return 1

    ensure_dirs()
    report = compare_frames(load_predictions(args.baseline), load_predictions(args.candidate))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"accuracy {report['baseline_accuracy']:.3f} -> {report['candidate_accuracy']:.3f}  "
        f"ошибок {report['baseline_errors']} -> {report['candidate_errors']}"
    )
    for row in report["focus_pairs"]:
        print(f"  {row['pair']:28} {row['baseline']:3d} -> {row['candidate']:3d}")
    print(f"Сравнение: {relative_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
