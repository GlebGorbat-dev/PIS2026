from __future__ import annotations

import numpy as np
from PIL import Image

from src.app.settings import MAX_MEAN_LUMINANCE, MIN_FOCUS, MIN_MEAN_LUMINANCE

PREVIEW_PX = 224


def measure(image: Image.Image) -> dict[str, float]:
    """Яркость и резкость кадра на превью 224×224 — том же размере, что видит модель."""
    preview = image.convert("RGB").resize((PREVIEW_PX, PREVIEW_PX))
    pixels = np.asarray(preview).astype(np.float32)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    padded = np.pad(luminance, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
    )
    return {
        "mean_luminance": round(float(luminance.mean()), 2),
        "focus": round(float(laplacian.var()), 2),
    }


def problem(features: dict[str, float]) -> str | None:
    """Сообщение о плохом кадре. Сначала свет, потом резкость: чёрный кадр не называют размытым."""
    luminance = features["mean_luminance"]
    if luminance < MIN_MEAN_LUMINANCE:
        return (
            "Снимок слишком тёмный: лист почти не виден. "
            "Сфотографируйте растение при более ярком свете, без сильной тени."
        )
    if luminance > MAX_MEAN_LUMINANCE:
        return (
            "Снимок слишком светлый: детали листа потеряны. "
            "Уберите вспышку в упор и не снимайте против солнца."
        )
    if features["focus"] < MIN_FOCUS:
        return (
            "Снимок слишком размытый: болезнь на листе не разобрать. "
            "Держите камеру неподвижно и сфокусируйтесь на листе."
        )
    return None
