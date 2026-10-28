from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.data.download import file_md5
from src.model.dataset import LeafCsvDataset, class_to_index
from src.model.predict import load_checkpoint
from src.paths import (
    PROJECT_ROOT,
    SPLITS_DIR,
    STAGE4_CHECKPOINT,
    STAGE5_REPORTS_DIR,
    ensure_dirs,
    relative_to_root,
)

CALIBRATION_BINS = 10

# Категориальная палитра (порядок слотов фиксирован) и синяя последовательная шкала.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES_DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
SEQUENTIAL_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
BASELINE = "#c3c2b7"


def run_inference(bundle: dict, csv_path: Path, batch_size: int = 32) -> tuple[pd.DataFrame, np.ndarray]:
    names = bundle["class_names"]
    dataset = LeafCsvDataset(csv_path, PROJECT_ROOT, class_to_index(names), bundle["transform"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = bundle["model"]
    chunks = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(bundle["device"]))
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return dataset.frame, np.concatenate(chunks)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, names: list[str]) -> dict:
    labels = list(range(len(names)))
    y_pred = probabilities.argmax(axis=1)
    onehot = label_binarize(y_true, classes=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    predicted_counts = np.bincount(y_pred, minlength=len(names))
    true_counts = np.bincount(y_true, minlength=len(names))
    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "average_precision": round(float(average_precision_score(onehot[:, i], probabilities[:, i])), 4),
            "support": int(support[i]),
            "predicted_count": int(predicted_counts[i]),
            "count_error": int(predicted_counts[i] - true_counts[i]),
        }

    def averaged(kind: str) -> dict:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=kind, zero_division=0
        )
        return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}

    confidence = probabilities.max(axis=1)
    correct = y_pred == y_true
    count_errors = predicted_counts - true_counts
    return {
        "images": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "top2_accuracy": round(float(top_k_accuracy_score(y_true, probabilities, k=2, labels=labels)), 4),
        "macro": averaged("macro"),
        "weighted": averaged("weighted"),
        "mAP": round(float(np.mean([per_class[n]["average_precision"] for n in names])), 4),
        "roc_auc_macro_ovr": round(
            float(roc_auc_score(onehot, probabilities, average="macro")), 4
        ),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=labels)), 4),
        "expected_calibration_error": round(expected_calibration_error(y_true, probabilities), 4),
        "mean_confidence_correct": round(float(confidence[correct].mean()), 4) if correct.any() else None,
        "mean_confidence_wrong": round(float(confidence[~correct].mean()), 4) if (~correct).any() else None,
        "count_error": {
            "mae_per_class": round(float(np.abs(count_errors).mean()), 4),
            "misassigned_share": round(float(np.abs(count_errors).sum() / 2 / len(y_true)), 4),
        },
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, rounds: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    accuracy, macro_f1 = [], []
    for _ in range(rounds):
        sample = rng.integers(0, len(y_true), len(y_true))
        accuracy.append(accuracy_score(y_true[sample], y_pred[sample]))
        macro_f1.append(
            f1_score(y_true[sample], y_pred[sample], labels=labels, average="macro", zero_division=0)
        )

    def interval(values: list[float]) -> list[float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return [round(float(low), 4), round(float(high), 4)]

    return {"rounds": rounds, "seed": seed, "accuracy_95ci": interval(accuracy), "macro_f1_95ci": interval(macro_f1)}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def save_confusion_matrix(matrix: np.ndarray, names: list[str], path: Path) -> None:
    rows = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
    image = ax.imshow(share, cmap=cmap, vmin=0.0, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "#ffffff" if share[i, j] > 0.55 else INK
            ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("предсказание", color=INK)
    ax.set_ylabel("истина", color=INK)
    ax.set_title("Матрица ошибок на test (доля от строки)", color=INK)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def save_pr_curves(y_true: np.ndarray, probabilities: np.ndarray, names: list[str], metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=SURFACE)
    _style_axes(ax)
    for i, name in enumerate(names):
        precision, recall, _ = precision_recall_curve(y_true == i, probabilities[:, i])
        ap = metrics["per_class"][name]["average_precision"]
        ax.plot(
            recall,
            precision,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            linestyle=SERIES_DASHES[i % len(SERIES_DASHES)],
            linewidth=2,
            label=f"{name}  AP={ap:.3f}",
        )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", color=INK)
    ax.set_ylabel("precision", color=INK)
    ax.set_title(f"Precision–recall на test, mAP={metrics['mAP']:.3f}", color=INK)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="lower left", frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Измерить качество модели на фиксированной выборке.")
    parser.add_argument("--weights", type=Path, default=STAGE4_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out-dir", type=Path, default=STAGE5_REPORTS_DIR)
    parser.add_argument("--bootstrap", type=int, default=1000, help="число повторов bootstrap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    ensure_dirs()
    csv_path = SPLITS_DIR / f"{args.split}.csv"
    if not csv_path.exists():
        print(f"Нет {csv_path}. Сначала: python -m src.data.split")
        return 1

    bundle = load_checkpoint(args.weights)
    names = bundle["class_names"]
    print(f"Веса: {relative_to_root(args.weights)}  устройство: {bundle['device']}")
    print(f"Выборка: {relative_to_root(csv_path)}")

    frame, probabilities = run_inference(bundle, csv_path)
    mapping = class_to_index(names)
    y_true = frame["class_name"].map(mapping).to_numpy()
    y_pred = probabilities.argmax(axis=1)

    metrics = compute_metrics(y_true, probabilities, names)
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    report = {
        "split": args.split,
        "split_csv": relative_to_root(csv_path),
        "split_csv_md5": file_md5(csv_path),
        "weights": relative_to_root(args.weights),
        "weights_md5": file_md5(args.weights),
        "architecture": payload.get("architecture"),
        "best_val_accuracy_at_training": round(float(payload.get("best_val_accuracy", 0.0)), 4),
        "device": str(bundle["device"]),
        "classes": names,
        **metrics,
        "bootstrap": bootstrap_intervals(y_true, y_pred, len(names), args.bootstrap, args.seed),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "path": frame["path"],
            "class_name": frame["class_name"],
            "predicted": [names[i] for i in y_pred],
            "confidence": probabilities.max(axis=1).round(4),
            "correct": y_pred == y_true,
        }
    )
    for i, name in enumerate(names):
        predictions[f"p_{name}"] = probabilities[:, i].round(4)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_matrix(np.array(metrics["confusion_matrix"]), names, out_dir / "confusion_matrix.png")
    save_pr_curves(y_true, probabilities, names, metrics, out_dir / "pr_curves.png")

    print(f"\nИзображений: {metrics['images']}")
    print(f"accuracy={metrics['accuracy']:.3f}  balanced={metrics['balanced_accuracy']:.3f}  mAP={metrics['mAP']:.3f}")
    print(
        f"macro precision={metrics['macro']['precision']:.3f}  "
        f"recall={metrics['macro']['recall']:.3f}  f1={metrics['macro']['f1']:.3f}"
    )
    print(f"{'класс':12} {'prec':>6} {'rec':>6} {'f1':>6} {'AP':>6} {'n':>4} {'pred':>5}")
    for name in names:
        row = metrics["per_class"][name]
        print(
            f"{name:12} {row['precision']:6.3f} {row['recall']:6.3f} {row['f1']:6.3f} "
            f"{row['average_precision']:6.3f} {row['support']:4d} {row['predicted_count']:5d}"
        )
    print(f"\nОтчёт: {relative_to_root(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
