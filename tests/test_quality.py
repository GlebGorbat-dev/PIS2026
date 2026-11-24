from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from src.app.quality import measure, problem
from src.app.settings import MAX_MEAN_LUMINANCE, MIN_FOCUS, MIN_MEAN_LUMINANCE


def _leaf() -> Image.Image:
    image = Image.new("RGB", (320, 240), (80, 120, 60))
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 8):
        draw.line((x, 0, x, 240), fill=(30, 90, 40))
    return image


def test_readable_leaf_has_no_quality_problem():
    features = measure(_leaf())
    assert MIN_MEAN_LUMINANCE <= features["mean_luminance"] <= MAX_MEAN_LUMINANCE
    assert features["focus"] >= MIN_FOCUS
    assert problem(features) is None


def test_dark_frame_is_reported_before_blur():
    message = problem({"mean_luminance": MIN_MEAN_LUMINANCE - 1, "focus": 0})
    assert message is not None
    assert "тёмн" in message


def test_bright_frame_is_reported():
    message = problem({"mean_luminance": MAX_MEAN_LUMINANCE + 1, "focus": 1000})
    assert message is not None
    assert "светл" in message


def test_blurry_frame_is_reported_when_the_light_is_fine():
    blurred = _leaf().filter(ImageFilter.GaussianBlur(12))
    features = measure(blurred)
    assert features["focus"] < MIN_FOCUS
    message = problem(features)
    assert message is not None
    assert "размыт" in message
