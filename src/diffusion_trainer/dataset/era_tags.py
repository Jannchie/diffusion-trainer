"""Era (year) tag assignment from post publication dates.

Anime art style drifts heavily across years; tagging each image's era turns
that drift into a controllable conditioning axis (Animagine-style
``newest``/``early``/``oldest`` modifiers) instead of an averaged-out
confound — the same philosophy as the quality axis in
:mod:`diffusion_trainer.dataset.quality_tags`. Images without a publication
date simply get no era tag ("era average", consistent with how a missing
quality signal is handled).
"""

from collections.abc import Sequence
from dataclasses import dataclass

ERA_CATEGORY = "year"
DEFAULT_ERA_SPEC = "oldest:2014,early:2017,mid:2020,recent:2023,newest"


@dataclass(frozen=True)
class EraBound:
    """One era bucket: years up to ``max_year`` inclusive (None = open end)."""

    tag: str
    max_year: int | None


def parse_era_spec(spec: str) -> list[EraBound]:
    """Parse ``"oldest:2014,early:2017,...,newest"`` into ascending era bounds.

    Each entry is ``tag:max_year``; only the final entry may omit the year to
    capture everything newer. An empty spec disables era tagging.
    """
    entries = [part.strip() for part in spec.split(",") if part.strip()]
    bounds: list[EraBound] = []
    previous_year: int | None = None
    for index, entry in enumerate(entries):
        tag, separator, year_text = entry.partition(":")
        tag = tag.strip()
        if not tag:
            msg = f"Era spec entry {entry!r} has no tag name"
            raise ValueError(msg)
        if not separator:
            if index != len(entries) - 1:
                msg = f"Only the last era spec entry may omit the year bound, got {entry!r}"
                raise ValueError(msg)
            bounds.append(EraBound(tag, None))
            break
        year = int(year_text)
        if previous_year is not None and year <= previous_year:
            msg = f"Era spec years must be strictly ascending, got {year} after {previous_year}"
            raise ValueError(msg)
        previous_year = year
        bounds.append(EraBound(tag, year))
    return bounds


def era_tag_for_year(year: int | None, bounds: Sequence[EraBound]) -> str | None:
    """Era tag for a publication year, or None when unknown/uncovered."""
    if year is None:
        return None
    for bound in bounds:
        if bound.max_year is None or year <= bound.max_year:
            return bound.tag
    return None
