"""Quality tiers: manual score wins, aesthetic fallback respects calibrated cuts."""

import pytest

from diffusion_trainer.dataset.quality_tags import DEFAULT_QUALITY_TAGS, calibrate_thresholds, quality_tier

THRESHOLDS = [0.25, 0.40, 0.60, 0.81]


def test_manual_score_maps_directly_and_ignores_aesthetic() -> None:
    # A rated post never consults the fallback, even with a contradicting aesthetic score.
    assert quality_tier(1, 0.99, THRESHOLDS) == 0
    assert quality_tier(3, None, THRESHOLDS) == 2
    assert quality_tier(5, 0.01, THRESHOLDS) == 4


def test_unrated_post_falls_back_to_aesthetic_cuts() -> None:
    assert quality_tier(0, 0.10, THRESHOLDS) == 0
    assert quality_tier(0, 0.30, THRESHOLDS) == 1
    assert quality_tier(0, 0.50, THRESHOLDS) == 2
    assert quality_tier(0, 0.70, THRESHOLDS) == 3
    assert quality_tier(0, 0.95, THRESHOLDS) == 4


def test_no_signal_yields_no_tier() -> None:
    assert quality_tier(0, None, THRESHOLDS) is None


def test_calibration_recovers_separable_cuts() -> None:
    # Tiers occupy disjoint score bands; the fitted cuts must land between them.
    bands = [[base + offset / 100 for offset in range(10)] for base in (0.0, 0.2, 0.4, 0.6, 0.8)]
    samples = [(tier + 1, value) for tier, band in enumerate(bands) for value in band]
    thresholds = calibrate_thresholds(samples)
    assert len(thresholds) == len(DEFAULT_QUALITY_TAGS) - 1
    assert thresholds == sorted(thresholds)
    for tier, cut in enumerate(thresholds):
        assert max(bands[tier]) < cut <= min(bands[tier + 1])
    # Round-trip: every calibration sample classifies back into its own tier.
    assert all(quality_tier(0, score, thresholds) == manual - 1 for manual, score in samples)


def test_calibration_tolerates_overlap_and_stays_monotone() -> None:
    samples = [(1, 0.30), (1, 0.10), (2, 0.05), (2, 0.35), (3, 0.30), (3, 0.55), (4, 0.50), (4, 0.75), (5, 0.70), (5, 0.95)]
    thresholds = calibrate_thresholds(samples)
    assert thresholds == sorted(thresholds)


def test_calibration_requires_every_tier() -> None:
    with pytest.raises(ValueError, match="tiers"):
        calibrate_thresholds([(1, 0.1), (2, 0.2), (3, 0.3)])
