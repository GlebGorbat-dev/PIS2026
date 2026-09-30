from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.architecture import build_model, pick_device
from src.model.dataset import (
    LeafCsvDataset,
    build_transforms,
    class_names_from_config,
    class_to_index,
)
from src.paths import (
    PROJECT_ROOT,
    SPLITS_DIR,
    STAGE4_CHECKPOINT,
    STAGE4_REPORTS_DIR,
    ensure_dirs,
    load_dataset_config,
    load_train_config,
    relative_to_root,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(train_csv: Path, names: list[str], device: torch.device) -> torch.Tensor:
    counts = pd.read_csv(train_csv)["class_name"].value_counts()
    total = int(counts.sum())
    n_classes = len(names)
    values = [total / (n_classes * int(counts.get(name, 1))) for name in names]
    return torch.tensor(values, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def save_curves(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Обучить первую модель классификации.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    ensure_dirs()
    dataset_config = load_dataset_config()
    train_config = load_train_config(args.config)
    names = class_names_from_config(dataset_config)
    mapping = class_to_index(names)
    seed = int(train_config["seed"])
    set_seed(seed)

    device = pick_device()
    print(f"Устройство: {device}")
    print(f"Архитектура: {train_config['architecture']}, freeze_backbone={train_config['freeze_backbone']}")
    print(f"Классы: {names}")

    train_set = LeafCsvDataset(
        SPLITS_DIR / "train.csv",
        PROJECT_ROOT,
        mapping,
        build_transforms(train_config, train=True),
    )
    val_set = LeafCsvDataset(
        SPLITS_DIR / "val.csv",
        PROJECT_ROOT,
        mapping,
        build_transforms(train_config, train=False),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config["num_workers"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config["num_workers"]),
    )

    model = build_model(
        len(names),
        pretrained=bool(train_config["pretrained"]),
        freeze_backbone=bool(train_config["freeze_backbone"]),
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total}, обучаемых: {trainable}")

    if train_config["use_class_weights"]:
        weights = class_weights(SPLITS_DIR / "train.csv", names, device)
        print(f"Веса классов: {dict(zip(names, [round(float(w), 3) for w in weights]))}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    history: list[dict] = []
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    patience = int(train_config["early_stopping_patience"])
    stale = 0

    epochs = int(train_config["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(row)
        print(
            f"Эпоха {epoch:02d}/{epochs}  "
            f"train loss={train_loss:.3f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.3f} acc={val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Ранняя остановка на эпохе {epoch}, лучшая val acc={best_acc:.3f} (эпоха {best_epoch})")
                break

    if best_state is None:
        print("Обучение не сохранило ни одного чекпоинта.")
        return 1

    checkpoint = {
        "model_state": best_state,
        "class_names": names,
        "config": train_config,
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "history": history,
        "architecture": train_config["architecture"],
    }
    STAGE4_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, STAGE4_CHECKPOINT)

    STAGE4_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curves = STAGE4_REPORTS_DIR / "train_curves.png"
    save_curves(history, curves)
    summary = {
        "architecture": train_config["architecture"],
        "freeze_backbone": train_config["freeze_backbone"],
        "pretrained": train_config["pretrained"],
        "device": str(device),
        "classes": names,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_acc, 4),
        "epochs_ran": len(history),
        "checkpoint": relative_to_root(STAGE4_CHECKPOINT),
        "history": history,
    }
    (STAGE4_REPORTS_DIR / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nЛучшая val accuracy: {best_acc:.3f} (эпоха {best_epoch})")
    print(f"Веса: {relative_to_root(STAGE4_CHECKPOINT)}")
    print(f"Сводка: {relative_to_root(STAGE4_REPORTS_DIR / 'train_summary.json')}")
    print(f"Графики: {relative_to_root(curves)}")
    print("Тестовая выборка на этом этапе не использовалась.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
