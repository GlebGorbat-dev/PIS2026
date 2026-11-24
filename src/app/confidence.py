from __future__ import annotations

from src.app.settings import LOW_CONFIDENCE_THRESHOLD, UNKNOWN_THRESHOLD


def confidence_band(confidence: float) -> str:
    """Одна из трёх веток сценария: confident, low, unknown."""
    if confidence < UNKNOWN_THRESHOLD:
        return "unknown"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    return "confident"
