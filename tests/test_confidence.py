from __future__ import annotations

from src.app.confidence import confidence_band
from src.app.settings import LOW_CONFIDENCE_THRESHOLD, UNKNOWN_THRESHOLD


def test_three_bands_meet_at_the_thresholds():
    assert confidence_band(LOW_CONFIDENCE_THRESHOLD) == "confident"
    assert confidence_band(LOW_CONFIDENCE_THRESHOLD - 0.01) == "low"
    assert confidence_band(UNKNOWN_THRESHOLD) == "low"
    assert confidence_band(UNKNOWN_THRESHOLD - 0.01) == "unknown"


def test_thresholds_are_ordered():
    assert 0 < UNKNOWN_THRESHOLD < LOW_CONFIDENCE_THRESHOLD < 1
