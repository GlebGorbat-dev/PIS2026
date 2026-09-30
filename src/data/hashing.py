"""Хэши для поиска дубликатов.

md5 ловит побайтово одинаковые файлы, dHash — визуально одинаковые кадры,
которые пересохранили с другим качеством, размером или повернули цвета.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

DHASH_SIDE = 8  # итоговый хэш — DHASH_SIDE * DHASH_SIDE = 64 бита


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image, side: int = DHASH_SIDE) -> int:
    """Разностный хэш: сравнивает яркость соседних пикселей по горизонтали.

    Устойчив к изменению масштаба и небольшой коррекции яркости, поэтому
    подходит для поиска near-duplicate внутри одного класса.
    """
    small = image.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    """Число различающихся бит двух хэшей."""
    return int(left ^ right).bit_count()


def dhash_to_hex(value: int, side: int = DHASH_SIDE) -> str:
    return f"{value:0{side * side // 4}x}"


def hex_to_dhash(text: str) -> int:
    return int(text, 16)
