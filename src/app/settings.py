from __future__ import annotations

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MIN_SIDE_PX = 64
# Пороги отсекают заведомо плохой кадр и не задевают 999 полевых снимков пшеницы:
# у них яркость 50–176, резкость (дисперсия лапласиана на 224 px) от 78.
MIN_MEAN_LUMINANCE = 40.0
MAX_MEAN_LUMINANCE = 220.0
MIN_FOCUS = 40.0

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
CROP_COOKIE = "current_crop"
DEFAULT_CROP = "wheat"
COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60

# Два порога на вероятности лучшего класса. Выше 0,6 ответ показывается как есть.
# От 0,4 до 0,6 программа предупреждает и даёт сравнить два варианта.
# Ниже 0,4 случай не распознан: диагноз не называется.
LOW_CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_THRESHOLD = 0.4
