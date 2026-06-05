"""Era spec parsing and year-to-era mapping."""

import pytest

from diffusion_trainer.dataset.era_tags import DEFAULT_ERA_SPEC, EraBound, era_tag_for_year, parse_era_spec


def test_default_spec_parses_with_open_end() -> None:
    bounds = parse_era_spec(DEFAULT_ERA_SPEC)
    assert [bound.tag for bound in bounds] == ["oldest", "early", "mid", "recent", "newest"]
    assert bounds[-1].max_year is None
    assert all(bound.max_year is not None for bound in bounds[:-1])


def test_year_maps_to_expected_bucket() -> None:
    bounds = parse_era_spec(DEFAULT_ERA_SPEC)
    assert era_tag_for_year(2008, bounds) == "oldest"
    assert era_tag_for_year(2014, bounds) == "oldest"  # boundary is inclusive
    assert era_tag_for_year(2015, bounds) == "early"
    assert era_tag_for_year(2020, bounds) == "mid"
    assert era_tag_for_year(2023, bounds) == "recent"
    assert era_tag_for_year(2026, bounds) == "newest"


def test_unknown_year_or_empty_spec_yields_no_tag() -> None:
    bounds = parse_era_spec(DEFAULT_ERA_SPEC)
    assert era_tag_for_year(None, bounds) is None
    assert parse_era_spec("") == []
    assert era_tag_for_year(2020, []) is None


def test_bounded_spec_leaves_newer_years_untagged() -> None:
    bounds = [EraBound("vintage", 2010)]
    assert era_tag_for_year(2005, bounds) == "vintage"
    assert era_tag_for_year(2020, bounds) is None


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="last era spec entry"):
        parse_era_spec("oldest,early:2017")
    with pytest.raises(ValueError, match="ascending"):
        parse_era_spec("a:2020,b:2015")
    with pytest.raises(ValueError, match="no tag name"):
        parse_era_spec(":2020")
