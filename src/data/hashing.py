from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

DHASH_SIDE = 8


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image, side: int = DHASH_SIDE) -> int:
    small = image.convert("L").resize((side + 1, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def dhash_to_hex(value: int, side: int = DHASH_SIDE) -> str:
    return f"{value:0{side * side // 4}x}"


def hex_to_dhash(text: str) -> int:
    return int(text, 16)
