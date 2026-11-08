"""Prompt sets: loading, filtering, and expansion into training previews."""

from pathlib import Path

import pytest

from diffusion_trainer.config import MOSSImageConfig, SampleOptions
from diffusion_trainer.config.prompt_set import load_prompt_set, to_sample_options
from diffusion_trainer.finetune.utils import get_sample_options_hash

REPO_SET = Path(__file__).parents[2] / "configs" / "eval" / "anime_prompts_v1.toml"

SET = """
[defaults]
steps = 50
guidance_scale = 3.0
negative_prompt = ""
width = 1024
height = 1024

[[prompts]]
name = "a"
category = "text"
seed = 1
prompt = "sign that says HI"

[[prompts]]
name = "b"
category = "photo"
seed = 2
width = 832
height = 1216
prompt = "a photo"
"""


@pytest.fixture
def prompt_set(tmp_path: Path) -> Path:
    path = tmp_path / "set.toml"
    path.write_text(SET, encoding="utf-8")
    return path


def test_defaults_fill_and_entries_override(prompt_set: Path) -> None:
    a, b = load_prompt_set(prompt_set)
    assert (a.width, a.height, a.steps) == (1024, 1024, 50)
    assert (b.width, b.height) == (832, 1216)


def test_only_matches_names_and_categories_and_steps_override(prompt_set: Path) -> None:
    assert [p.name for p in load_prompt_set(prompt_set, only=["photo"])] == ["b"]
    assert [p.name for p in load_prompt_set(prompt_set, only=["a"])] == ["a"]
    assert {p.steps for p in load_prompt_set(prompt_set, steps=30)} == {30}


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.toml"
    path.write_text(SET.replace('name = "b"', 'name = "a"'), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_prompt_set(path)


def test_seeds_expand_into_one_prompt_per_seed(tmp_path: Path) -> None:
    path = tmp_path / "seeds.toml"
    path.write_text(SET.replace("seed = 2\n", "seeds = [7, 8]\n"), encoding="utf-8")
    prompts = load_prompt_set(path)
    assert [(p.name, p.seed) for p in prompts] == [("a", 1), ("b_s7", 7), ("b_s8", 8)]
    assert [p.name for p in load_prompt_set(path, only=["photo"])] == ["b_s7", "b_s8"]


def test_config_appends_the_set_after_inline_previews(prompt_set: Path) -> None:
    inline = {"prompt": "inline", "negative_prompt": "", "steps": 30, "seed": 0}
    config = MOSSImageConfig(
        model_path="x",
        dataset_path="x",
        preview_sample_options=[inline],  # type: ignore[list-item]
        preview_prompt_set=str(prompt_set),
        preview_prompt_set_only=["text"],
        preview_prompt_set_steps=30,
    )
    assert [o.name for o in config.preview_sample_options] == [None, "a"]
    assert config.preview_sample_options[1] == to_sample_options(load_prompt_set(prompt_set, only=["a"], steps=30))[0]


def test_unnamed_options_keep_their_pre_name_hash() -> None:
    # Pinned so adding SampleOptions.name never renames existing preview files.
    option = SampleOptions(prompt="p", negative_prompt="", steps=30, seed=47, width=1024, height=1024, guidance_scale=3.0)
    assert get_sample_options_hash(option) == "c872777e"


@pytest.mark.parametrize("name", ["anime_prompts_v1.toml", "aesthetic_v1.toml"])
def test_repo_prompt_sets_load(name: str) -> None:
    prompts = load_prompt_set(REPO_SET.parent / name)
    assert len(prompts) >= 40
    assert all(p.width % 16 == 0 and p.height % 16 == 0 for p in prompts)


EXT = """
include = ["base.toml"]

[defaults]
steps = 30
guidance_scale = 3.0
negative_prompt = ""
width = 832
height = 1216

[[prompts]]
name = "{name}"
category = "artist"
seed = 3
prompt = "@x"
"""


def test_include_extends_another_set_and_keeps_its_defaults(tmp_path: Path) -> None:
    (tmp_path / "base.toml").write_text(SET)
    (tmp_path / "ext.toml").write_text(EXT.format(name="c"))
    prompts = load_prompt_set(tmp_path / "ext.toml")
    assert [p.name for p in prompts] == [*(p.name for p in load_prompt_set(tmp_path / "base.toml")), "c"]
    assert prompts[0].steps == 50  # the included set's own defaults
    assert prompts[-1].width == 832


def test_include_rejects_names_clashing_with_the_included_set(tmp_path: Path) -> None:
    (tmp_path / "base.toml").write_text(SET)
    (tmp_path / "ext.toml").write_text(EXT.format(name="a"))
    with pytest.raises(ValueError, match="Duplicate"):
        load_prompt_set(tmp_path / "ext.toml")
