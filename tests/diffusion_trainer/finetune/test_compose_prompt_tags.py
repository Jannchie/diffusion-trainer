"""Category-aware prompt assembly: pinned prefix, shuffled tail, legacy fallback."""

import random

from diffusion_trainer.config import BaseConfig
from diffusion_trainer.finetune.base import TagCompositionRules, compose_prompt_tags

ORDER = ("quality", "artist", "copyright", "character", "general")

TAGS = ["good quality", "wlop", "genshin_impact", "ganyu_(genshin_impact)", "1girl", "solo", "long_hair"]
CATEGORIES = ["quality", "artist", "copyright", "character", "general", "general", "general"]


def _rules(*, shuffled: frozenset[str] = frozenset(), droppable: frozenset[str] = frozenset(), dropout: float = 0.0) -> TagCompositionRules:
    return TagCompositionRules(category_order=ORDER, shuffled_categories=shuffled, droppable_categories=droppable, single_tag_dropout=dropout)


def test_categories_concatenate_in_configured_order() -> None:
    assert compose_prompt_tags(TAGS, CATEGORIES, _rules()) == TAGS


def test_unlisted_categories_never_enter_the_prompt() -> None:
    composed = compose_prompt_tags([*TAGS, "highres"], [*CATEGORIES, "meta"], _rules())
    assert "highres" not in composed


def test_shuffle_randomizes_only_listed_categories() -> None:
    random.seed(0)
    rules = _rules(shuffled=frozenset({"general"}))
    results = {tuple(compose_prompt_tags(TAGS, CATEGORIES, rules)) for _ in range(50)}
    assert len(results) > 1  # the general tail does shuffle
    for result in results:
        assert list(result[:4]) == TAGS[:4]  # quality/artist/copyright/character stay pinned
        assert sorted(result[4:]) == sorted(TAGS[4:])


def test_dropout_spares_pinned_categories() -> None:
    random.seed(0)
    rules = _rules(droppable=frozenset({"general"}), dropout=0.99)
    for _ in range(50):
        composed = compose_prompt_tags(TAGS, CATEGORIES, rules)
        assert composed[:4] == TAGS[:4]
        assert set(composed[4:]) <= set(TAGS[4:])


def test_legacy_dataset_without_categories_behaves_flat() -> None:
    # Empty/misaligned categories: every tag is "general" -> full shuffle scope.
    random.seed(0)
    assert compose_prompt_tags(TAGS, [], _rules()) == TAGS
    rules = _rules(shuffled=frozenset({"general"}))
    results = {tuple(compose_prompt_tags(TAGS, [], rules)) for _ in range(50)}
    assert len(results) > 1
    assert any(result[0] != TAGS[0] for result in results)  # even the first tag moves


def test_rules_from_config_fold_global_switches() -> None:
    config = BaseConfig(model_path="x", dataset_path="y", shuffle_tags=False, single_tag_dropout=0.0)
    rules = TagCompositionRules.from_config(config)
    assert rules.shuffled_categories == frozenset()
    assert rules.droppable_categories == frozenset()

    config = BaseConfig(model_path="x", dataset_path="y", shuffle_tags=True, single_tag_dropout=0.1)
    rules = TagCompositionRules.from_config(config)
    assert rules.category_order == ("quality", "year", "artist", "copyright", "character", "general")
    assert rules.shuffled_categories == frozenset({"general"})
    assert rules.droppable_categories == frozenset({"general"})
