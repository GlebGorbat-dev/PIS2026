from __future__ import annotations

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MIN_SIDE_PX = 64

ALLOWED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}

FORMAT_EXTENSIONS: dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WEBP": ".webp",
}

UPLOAD_COOKIE = "current_upload"
COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60
