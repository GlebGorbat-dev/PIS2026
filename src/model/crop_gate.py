"""Проверка, что на снимке выбранная культура, до диагноза болезни.

ImageNet не знает классы «лист пшеницы» и «лист ячменя»: полевой лист обычно
получает низкую вероятность или одну из устойчивых ложных меток. Уверенный
чужой класс (банан, початок кукурузы при выбранной пшенице и т.п.) означает,
что болезни этой культуры на снимке нет.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.model.architecture import pick_device
from src.paths import CROP_GATE_WEIGHTS, CROP_GENITIVE, CROP_TITLES

# Ниже этой вероятности ImageNet не настаивает на метке. Так выглядят листья
# зерновых после приведения к 224 px. Порог стоит над ложными метками,
# которые реальные листья пшеницы набирают лишь изредка, и под «banana» (0,98).
CONFIDENCE_FLOOR = 0.60

# Метки, которые MobileNet-ImageNet ставит настоящим листьям пшеницы
# с вероятностью от 0,60. Отклонять их нельзя: иначе отсеются больные листья.
LEAF_LOOKALIKES = frozenset({"ear", "pineapple", "vine snake", "cardoon"})

# Единственная культура, которую ImageNet узнаёт напрямую.
# Совпадение с выбранной культурой пропускает кадр дальше; в тексте отказа
# распознанный объект всё равно не называется.
LABEL_CROP = {"corn": "maize"}

_LOCK = threading.Lock()
_BUNDLE: dict | None = None


def _imagenet_transform() -> transforms.Compose:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    return weights.transforms()


def _categories() -> list[str]:
    return list(MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"])


def load() -> dict:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not CROP_GATE_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Нет весов проверки культуры {CROP_GATE_WEIGHTS}."
            )
        device = pick_device()
        payload = torch.load(CROP_GATE_WEIGHTS, map_location=device, weights_only=False)
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(payload["model_state"])
        model.to(device)
        model.eval()
        _BUNDLE = {
            "model": model,
            "transform": _imagenet_transform(),
            "categories": _categories(),
            "device": device,
        }
        return _BUNDLE


def _to_image(source: Image.Image | Path | str | bytes) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as image:
            return image.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def classify(source: Image.Image | Path | str | bytes) -> tuple[str, float]:
    bundle = load()
    tensor = bundle["transform"](_to_image(source)).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return bundle["categories"][index], float(probabilities[index].item())


def decide(label: str, probability: float, crop: str) -> str | None:
    """Сообщение об отказе или None, если снимок можно отдать модели болезни."""
    if crop not in CROP_TITLES:
        raise KeyError(crop)
    if probability < CONFIDENCE_FLOOR or label in LEAF_LOOKALIKES:
        return None
    detected = LABEL_CROP.get(label)
    if detected == crop:
        return None
    return _message(crop)


def check_crop(source: Image.Image | Path | str | bytes, crop: str) -> str | None:
    label, probability = classify(source)
    return decide(label, probability, crop)


def _message(crop: str) -> str:
    selected = CROP_TITLES[crop].lower()
    leaf = CROP_GENITIVE[crop]
    return (
        f"На снимке не {selected}. Это не лист {leaf}, "
        "поэтому болезнь не определяется. Загрузите фотографию выбранной культуры."
    )
