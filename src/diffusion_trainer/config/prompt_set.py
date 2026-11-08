"""Named prompt sets shared by training previews and offline evaluation.

A prompt set is a TOML file with a ``[defaults]`` table and ``[[prompts]]``
entries (see ``configs/eval/anime_prompts_v1.toml``). Training configs point
at one with ``preview_prompt_set``; ``scripts/eval_moss_prompts.py`` renders
the same file, so in-training previews and post-hoc grids stay comparable.
A set can ``include = ["other.toml"]`` (paths relative to itself) to extend
another set without copying it; included prompts keep their own defaults.
"""

import tomllib
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path

from diffusion_trainer.config import SampleOptions


@dataclass(frozen=True)
class EvalPrompt:
    name: str
    category: str
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance_scale: float
    width: int
    height: int


def _expand_entry(entry: dict) -> list[EvalPrompt]:
    """An entry with ``seeds = [...]`` becomes one prompt per seed, named ``<name>_s<seed>``."""
    seeds = entry.pop("seeds", None)
    if seeds is None:
        return [EvalPrompt(**entry)]
    return [EvalPrompt(**{**entry, "name": f"{entry['name']}_s{seed}", "seed": seed}) for seed in seeds]


def _read_prompts(path: Path) -> list[EvalPrompt]:
    """``path``'s prompts after those of every set it includes."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    included = [prompt for name in data.get("include", []) for prompt in _read_prompts(path.parent / name)]
    return included + [prompt for entry in data.get("prompts", []) for prompt in _expand_entry({**defaults, **entry})]


def load_prompt_set(path: str | PathLike, *, only: list[str] | None = None, steps: int | None = None) -> list[EvalPrompt]:
    """Read a prompt set; ``only`` keeps entries whose name or category is listed, ``steps`` overrides every entry."""
    prompts = _read_prompts(Path(path))
    names = [p.name for p in prompts]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        msg = f"Duplicate prompt names in {path}: {sorted(duplicates)}"
        raise ValueError(msg)
    if only:
        prompts = [p for p in prompts if p.name in only or p.category in only]
    if steps:
        prompts = [replace(p, steps=steps) for p in prompts]
    return prompts


def to_sample_options(prompts: list[EvalPrompt]) -> list[SampleOptions]:
    return [
        SampleOptions(
            name=p.name,
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            steps=p.steps,
            seed=p.seed,
            width=p.width,
            height=p.height,
            guidance_scale=p.guidance_scale,
        )
        for p in prompts
    ]
