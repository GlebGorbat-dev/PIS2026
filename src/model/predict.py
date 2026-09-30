from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.model.architecture import build_model, pick_device
from src.model.dataset import build_transforms
from src.paths import STAGE4_CHECKPOINT, load_dataset_config

_CACHE: tuple | None = None


@dataclass(frozen=True)
class Prediction:
    class_name: str
    class_ru: str
    confidence: float
    probabilities: dict[str, float]


def _to_image(source: Image.Image | Path | str | bytes) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as image:
            return image.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def load_checkpoint(path: Path | None = None, device: torch.device | None = None):
    global _CACHE
    checkpoint_path = path or STAGE4_CHECKPOINT
    device = device or pick_device()
    cache_key = (str(checkpoint_path), str(device))
    if _CACHE is not None and _CACHE[0] == cache_key:
        return _CACHE[1]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Нет весов {checkpoint_path}. Сначала: python -m src.model.train"
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = list(payload["class_names"])
    train_config = payload["config"]
    model = build_model(
        len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    transform = build_transforms(train_config, train=False)
    bundle = {
        "model": model,
        "class_names": class_names,
        "config": train_config,
        "transform": transform,
        "device": device,
        "labels": load_dataset_config()["classes"],
    }
    _CACHE = (cache_key, bundle)
    return bundle


def predict(
    source: Image.Image | Path | str | bytes,
    *,
    weights_path: Path | None = None,
) -> Prediction:
    bundle = load_checkpoint(weights_path)
    tensor = bundle["transform"](_to_image(source)).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    class_name = bundle["class_names"][index]
    probs = {
        name: float(probabilities[i].item())
        for i, name in enumerate(bundle["class_names"])
    }
    return Prediction(
        class_name=class_name,
        class_ru=bundle["labels"][class_name]["ru"],
        confidence=float(probabilities[index].item()),
        probabilities=probs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Предсказать класс по фотографии.")
    parser.add_argument("image", type=Path, help="путь к изображению")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.image.exists():
        print(f"Нет файла {args.image}")
        return 1
    result = predict(args.image, weights_path=args.weights)
    print(f"{result.class_name} ({result.class_ru})  уверенность {result.confidence:.3f}")
    for name, value in sorted(result.probabilities.items(), key=lambda item: -item[1]):
        print(f"  {name:12} {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

