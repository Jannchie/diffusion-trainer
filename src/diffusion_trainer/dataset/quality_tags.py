"""Quality-tier assignment: manual star scores first, aesthetic-score fallback.

Tiers are 0..4 (worst..best), rendered as one mutually exclusive quality tag
per image (NAI-style ``worst quality`` .. ``best quality`` by default, which
inherits the quality-word prior of NAI-lineage base models). Manual star
scores (1..5) map directly to tiers; unrated images (score 0) fall back to a
continuous aesthetic score (e.g. SILVA) cut at four ascending thresholds.

Thresholds should be *calibrated* from the rated population rather than fixed
by hand: :func:`calibrate_thresholds` picks, for each adjacent tier pair, the
cut that minimizes disagreement with the curator's manual scores, so the
fallback reproduces the same standards on the unrated remainder.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence

DEFAULT_QUALITY_TAGS: tuple[str, ...] = ("worst quality", "bad quality", "normal quality", "good quality", "best quality")
QUALITY_CATEGORY = "quality"
NUM_QUALITY_TIERS = len(DEFAULT_QUALITY_TAGS)


def quality_tier(manual_score: int, aesthetic_score: float | None, thresholds: Sequence[float]) -> int | None:
    """Tier 0..4 for one image, or None when neither signal exists.

    ``manual_score`` follows the curator scale (0 = unrated, 1..5 = rated);
    a rated image never consults the aesthetic fallback.
    """
    if manual_score >= 1:
        return min(manual_score, NUM_QUALITY_TIERS) - 1
    if aesthetic_score is None:
        return None
    return bisect_right(thresholds, aesthetic_score)


def _best_cut(lower: Sequence[float], upper: Sequence[float]) -> float:
    """Cut between two adjacent tiers minimizing misclassified rated samples.

    Classification must match ``quality_tier``'s ``bisect_right``: scores
    ``>= cut`` land in the upper tier. A sample is therefore misclassified
    when a lower-tier score lands at/above the cut or an upper-tier score
    lands below it.
    """
    lower_sorted = sorted(lower)
    upper_sorted = sorted(upper)
    best_cut = best_err = float("inf")
    for candidate in sorted({*lower_sorted, *upper_sorted}):
        err = (len(lower_sorted) - bisect_left(lower_sorted, candidate)) + bisect_left(upper_sorted, candidate)
        if err < best_err:
            best_err = err
            best_cut = candidate
    return best_cut


def calibrate_thresholds(samples: Iterable[tuple[int, float]]) -> list[float]:
    """Fit four ascending thresholds from (manual_score 1..5, aesthetic_score) pairs.

    Each threshold separates one adjacent tier pair; cuts are clamped to be
    non-decreasing so the resulting tier function stays monotone even when a
    sparsely populated tier would locally invert the order.
    """
    by_tier: dict[int, list[float]] = {}
    for manual_score, aesthetic_score in samples:
        if 1 <= manual_score <= NUM_QUALITY_TIERS:
            by_tier.setdefault(manual_score - 1, []).append(aesthetic_score)
    missing = [tier for tier in range(NUM_QUALITY_TIERS) if not by_tier.get(tier)]
    if missing:
        msg = f"Cannot calibrate aesthetic thresholds: no rated samples for tiers {missing}"
        raise ValueError(msg)
    thresholds: list[float] = []
    floor = float("-inf")
    for tier in range(NUM_QUALITY_TIERS - 1):
        floor = max(_best_cut(by_tier[tier], by_tier[tier + 1]), floor)
        thresholds.append(floor)
    return thresholds
