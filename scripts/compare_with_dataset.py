"""Dataset-grounded character & artist comparison for SD 1.5 checkpoints.

This complements ``scripts/compare_models.py`` (hand-written prompts) by
grounding every prompt in a REAL training row. It samples rows from the
exported dataset (``hf://user/repo`` or a local export dir), decodes each row's
VAE latent back into the training-target image, rebuilds the prompt from that
row's category-ordered tags exactly as training composes them (shuffle/dropout
off for reproducibility), and renders ``[original | <model> ...]`` panels.

Tags are grouped into two sections so each learning axis is judged against the
source distribution it was trained on:

- ``artist``  -- the top artist tags (style learning).
- ``character`` -- the top character tags (identity learning).

The "original" panel is a VAE reconstruction of the stored latent (the exported
dataset keeps latents, not source pixels), i.e. the exact target the model saw.

Usage:
    uv run python scripts/compare_with_dataset.py --config configs/compare/v0.8-dataset.toml

Outputs under ``output_dir``: ``originals/<key>.png`` (decoded targets),
``singles/<model>/<name>.png`` (raw model outputs), ``panels/<section>/<name>.png``
(labeled side-by-side) and ``index.html`` (grouped by section and tag). Existing
files are skipped, so reruns resume where they left off.
"""

import argparse
import tarfile
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

# Same-directory import: Python puts the script's folder on sys.path[0], so the
# sibling comparison tool's conditioning/compositing helpers are reused verbatim.
from compare_models import CompareConfig, ModelSpec, PromptSpec, compose_comparison, generate_one, slugify
from huggingface_hub import hf_hub_download
from PIL import Image
from pyarrow import parquet as pq
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn

from diffusion_trainer.dataset.sharing import METADATA_FILENAME, SHARD_MEMBER_RE, read_verified_member
from diffusion_trainer.dataset.streaming import HfShardSource, LocalShardSource, ShardSource
from diffusion_trainer.finetune.base import TagCompositionRules, compose_prompt_tags, escape_attention_syntax
from diffusion_trainer.finetune.utils import load_sd15_pipeline

console = Console()

SECTIONS = ("artist", "character")


@dataclass(frozen=True)
class SelectionConfig:
    """How many tags/rows to sample per learning axis and how to compose prompts."""

    dataset: str
    n_artists: int
    n_characters: int
    samples_per_tag: int
    min_tag_count: int
    category_order: tuple[str, ...]

    def count_for(self, section: str) -> int:
        return self.n_artists if section == "artist" else self.n_characters


@dataclass(frozen=True)
class Selection:
    """One chosen (section, tag, sample) -> training row."""

    section: str
    tag: str
    index: int
    row: dict

    @property
    def key(self) -> str:
        return self.row["key"]

    @property
    def name(self) -> str:
        """Stable output basename for this cell."""
        return f"{slugify(self.tag)}-{self.index}-{self.key[:8]}"


def load_configs(path: Path) -> tuple[CompareConfig, SelectionConfig]:
    """Parse the TOML into the shared CompareConfig plus dataset-selection settings."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    models = [ModelSpec(**entry) for entry in raw["models"]]
    compare = CompareConfig(
        output_dir=Path(raw["output_dir"]),
        models=models,
        prompts=[],  # prompts are built per selected row, not listed in the config
        seeds=[int(raw.get("seed", 47))],
        steps=int(raw.get("steps", 25)),
        guidance_scale=float(raw.get("guidance_scale", 7.5)),
        negative_prompt=str(raw.get("negative_prompt", "")),
        hires_scale=float(raw.get("hires_scale", 0.0)),
        hires_strength=float(raw.get("hires_strength", 0.45)),
        hires_steps=int(raw.get("hires_steps", 20)),
    )
    selection = SelectionConfig(
        dataset=str(raw["dataset"]),
        n_artists=int(raw.get("n_artists", 16)),
        n_characters=int(raw.get("n_characters", 16)),
        samples_per_tag=int(raw.get("samples_per_tag", 2)),
        min_tag_count=int(raw.get("min_tag_count", 8)),
        category_order=tuple(raw.get("tag_category_order", ["quality", "year", "artist", "copyright", "character", "general"])),
    )
    return compare, selection


def load_dataset(dataset: str) -> tuple[list[dict], ShardSource]:
    """Load ``metadata.parquet`` and a shard resolver for an ``hf://`` repo or local export dir."""
    if dataset.startswith("hf://"):
        repo_id, _, revision = dataset[len("hf://"):].partition("@")
        metadata_path = hf_hub_download(repo_id, METADATA_FILENAME, repo_type="dataset", revision=revision or None)
        # Pin the resolved commit so cached shards resolve without a network round-trip.
        commit_sha = Path(metadata_path).parent.name
        rows = pq.read_table(metadata_path).to_pylist()
        return rows, HfShardSource(repo_id, commit_sha)
    export_dir = Path(dataset)
    rows = pq.read_table(export_dir / METADATA_FILENAME).to_pylist()
    return rows, LocalShardSource(export_dir)


def is_usable(row: dict) -> bool:
    """A row we can both decode (has a shard) and compose category-ordered prompts from."""
    tags = row.get("tags") or []
    categories = row.get("tag_categories") or []
    return bool(row.get("shard")) and bool(row.get("train_resolution")) and bool(tags) and len(categories) == len(tags)


def category_tags(row: dict, category: str) -> list[str]:
    """Tags of ``row`` that belong to ``category`` (order-preserving)."""
    tags = row.get("tags") or []
    categories = row.get("tag_categories") or []
    return [tag for tag, cat in zip(tags, categories, strict=True) if cat == category]


def index_tags(rows: list[dict], category: str) -> dict[str, list[dict]]:
    """Map each tag in ``category`` to the usable rows carrying it."""
    index: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not is_usable(row):
            continue
        for tag in dict.fromkeys(category_tags(row, category)):  # de-dup within a row
            index[tag].append(row)
    return index


def top_tags(tag_index: dict[str, list[dict]], count: int, min_count: int) -> list[tuple[str, list[dict]]]:
    """The ``count`` most frequent tags meeting ``min_count`` (ties broken by name)."""
    items = [(tag, rows) for tag, rows in tag_index.items() if len(rows) >= min_count]
    items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    return items[:count]


def row_score(row: dict) -> int:
    """Prefer clean, high-quality single-subject rows for an unambiguous style/identity read."""
    tags = set(row.get("tags") or [])
    score = 0
    score += 5 if "best quality" in tags else (3 if "good quality" in tags else 0)
    if "solo" in tags:
        score += 3
    if {"1girl", "1boy"} & tags:
        score += 2
    return score


def pick_rows(rows: list[dict], k: int, used_keys: set[str]) -> list[dict]:
    """Top-``k`` rows by representativeness, avoiding keys already chosen (with fallback)."""
    ranked = sorted(rows, key=lambda r: (-row_score(r), r["key"]))
    picked = [row for row in ranked if row["key"] not in used_keys][:k]
    if len(picked) < k:  # not enough unique rows: allow reuse to fill the quota
        picked = ranked[:k]
    for row in picked:
        used_keys.add(row["key"])
    return picked


def build_selections(rows: list[dict], selection: SelectionConfig) -> list[Selection]:
    """Choose the (section, tag, sample) rows for every learning axis."""
    selections: list[Selection] = []
    for section in SECTIONS:
        tag_index = index_tags(rows, section)
        chosen = top_tags(tag_index, selection.count_for(section), selection.min_tag_count)
        if not chosen:
            console.log(f"[yellow]no {section} tags reached min_tag_count={selection.min_tag_count}; skipping section")
        used_keys: set[str] = set()
        for tag, tag_rows in chosen:
            for index, row in enumerate(pick_rows(tag_rows, selection.samples_per_tag, used_keys)):
                selections.append(Selection(section=section, tag=tag, index=index, row=row))
        console.log(f"{section}: selected {len(chosen)} tags x {selection.samples_per_tag} samples")
    return selections


def build_prompt(row: dict, rules: TagCompositionRules) -> str:
    """Compose this row's prompt the way training does (category order, parens escaped)."""
    composed = compose_prompt_tags(list(row.get("tags") or []), list(row.get("tag_categories") or []), rules)
    caption = (row.get("caption") or "").strip()
    text = f"{caption}, " + ", ".join(composed) if caption else ", ".join(composed)
    return escape_attention_syntax(text)


def extract_latents(tar_path: Path, keys: set[str], rows_by_key: dict[str, dict]) -> dict[str, np.ndarray]:
    """Pull the requested members' latents out of one shard tar (checksum-verified)."""
    found: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            match = SHARD_MEMBER_RE.match(member.name)
            if match is None or not member.isfile():
                continue
            key = match.group(3)
            if key not in keys:
                continue
            data = read_verified_member(tar, member, rows_by_key[key])
            if data is None:
                continue
            with np.load(BytesIO(data)) as npz:
                if "latents" in npz:
                    found[key] = npz["latents"]
            if len(found) == len(keys):
                break
    return found


@torch.no_grad()
def decode_original(pipeline: object, latents: np.ndarray) -> Image.Image:
    """Reconstruct the training target from a stored (unscaled) VAE latent.

    Export stores raw encoder samples (``scaling_factor`` is NOT pre-applied),
    so the latent decodes directly without the diffusion-space rescale. The VAE
    is the fp32 copy loaded for clean decoding (fp16 NaNs on anime VAEs at 640+).
    """
    vae = pipeline.vae  # type: ignore[attr-defined]
    tensor = torch.from_numpy(np.asarray(latents)).unsqueeze(0).to(device=vae.device, dtype=torch.float32)
    image = vae.decode(tensor).sample
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]  # type: ignore[attr-defined]


def decode_originals(
    selections: list[Selection],
    shard_source: ShardSource,
    decode_pipeline: object,
    originals_dir: Path,
) -> None:
    """Download each needed shard once and cache the decoded original PNGs."""
    originals_dir.mkdir(parents=True, exist_ok=True)
    rows_by_key = {sel.key: sel.row for sel in selections}
    pending = {key for key in rows_by_key if not (originals_dir / f"{key}.png").exists()}
    if not pending:
        console.log("all originals already decoded")
        return

    keys_by_shard: dict[str, set[str]] = defaultdict(set)
    for key in pending:
        keys_by_shard[rows_by_key[key]["shard"]].add(key)

    progress = Progress(
        SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("decoding originals", total=len(pending))
        for shard, keys in keys_by_shard.items():
            tar_path = Path(shard_source(shard))
            for key, latents in extract_latents(tar_path, keys, rows_by_key).items():
                decode_original(decode_pipeline, latents).save(originals_dir / f"{key}.png")
                progress.advance(task)


def write_index(compare: CompareConfig, cells: list[tuple[Selection, Path]]) -> Path:
    """Contact sheet grouped by section then tag; panels embedded by relative path."""
    column_names = " | ".join(["original", *(model.name for model in compare.models)])
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>dataset comparison</title>",
        "<style>body{background:#181818;color:#ddd;font-family:sans-serif;margin:24px}",
        "h1{font-size:20px}h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:40px;text-transform:capitalize}",
        "h3{color:#9cf;margin:24px 0 4px}img{max-width:100%;height:auto;display:block;margin:6px 0 24px}</style>",
        "</head><body>",
        f"<h1>dataset-grounded comparison &mdash; {escape(column_names)}</h1>",
        f"<p>{len(cells)} panels | steps {compare.steps} | cfg {compare.guidance_scale} | "
        f"hires x{compare.hires_scale}@{compare.hires_strength}</p>",
    ]
    current_section: str | None = None
    current_tag: str | None = None
    for selection, panel_path in cells:
        if selection.section != current_section:
            current_section = selection.section
            current_tag = None
            parts.append(f"<h2>{escape(selection.section)}</h2>")
        if selection.tag != current_tag:
            current_tag = selection.tag
            parts.append(f"<h3>{escape(selection.tag)}</h3>")
        parts.append(f"<img src='{escape(panel_path.relative_to(compare.output_dir).as_posix())}' loading='lazy'>")
    parts.append("</body></html>")
    index_path = compare.output_dir / "index.html"
    index_path.write_text("\n".join(parts), encoding="utf-8")
    return index_path


def render_panels(
    selections: list[Selection],
    compare: CompareConfig,
    pipelines: dict[str, object],
    originals_dir: Path,
    rules: TagCompositionRules,
) -> list[tuple[Selection, Path]]:
    """Generate every model output and compose the [original | model...] panels."""
    seed = compare.seeds[0]
    panels_dir = compare.output_dir / "panels"
    singles_dir = compare.output_dir / "singles"
    cells: list[tuple[Selection, Path]] = []

    progress = Progress(
        SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("rendering panels", total=len(selections))
        for selection in selections:
            panel_path = panels_dir / selection.section / f"{selection.name}.png"
            cells.append((selection, panel_path))
            if panel_path.exists():
                progress.advance(task)
                continue

            row = selection.row
            resolution = row["train_resolution"]
            prompt = PromptSpec(
                prompt=build_prompt(row, rules),
                category=f"{selection.section}: {selection.tag}",
                negative_prompt=compare.negative_prompt,
                width=int(resolution[0]),
                height=int(resolution[1]),
            )

            outputs: list[tuple[str, Image.Image]] = []
            for model in compare.models:
                single_path = singles_dir / model.name / f"{selection.name}.png"
                if single_path.exists():
                    image = Image.open(single_path).convert("RGB")
                else:
                    image = generate_one(pipelines[model.name], model, prompt, compare, seed)
                    single_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(single_path)
                outputs.append((model.name, image))

            # Match the original to the generated size (hires fix upscales the
            # model outputs past the latent's native resolution).
            target_size = outputs[0][1].size
            original = Image.open(originals_dir / f"{row['key']}.png").convert("RGB").resize(target_size, Image.LANCZOS)

            panel_path.parent.mkdir(parents=True, exist_ok=True)
            compose_comparison([("original", original), *outputs], prompt, seed, compare).save(panel_path)
            progress.advance(task)
    return cells


def run(compare: CompareConfig, selection: SelectionConfig) -> None:
    """Select rows, decode originals, render every panel and write the index."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare.output_dir.mkdir(parents=True, exist_ok=True)

    rows, shard_source = load_dataset(selection.dataset)
    console.log(f"loaded {len(rows)} manifest rows from {selection.dataset}")
    selections = build_selections(rows, selection)
    if not selections:
        msg = "No rows selected; check min_tag_count / dataset categories."
        raise SystemExit(msg)
    console.log(f"selected {len(selections)} rows ({len({s.key for s in selections})} unique) across {len({s.row['shard'] for s in selections})} shards")

    pipelines: dict[str, object] = {}
    for model in compare.models:
        pipeline = load_sd15_pipeline(model.path, torch.float16)
        pipeline.to(device)
        pipeline.vae.to(torch.float32)  # fp16 anime-VAE decode NaNs at 640+; latents themselves are clean
        pipeline.set_progress_bar_config(disable=True)
        pipelines[model.name] = pipeline
        console.log(f"loaded [bold]{model.name}[/] from {model.path} "
                    f"(scheduler {type(pipeline.scheduler).__name__}, prediction {pipeline.scheduler.config.get('prediction_type')})")

    decode_originals(selections, shard_source, next(iter(pipelines.values())), compare.output_dir / "originals")

    rules = TagCompositionRules(
        category_order=selection.category_order,
        shuffled_categories=frozenset(),  # deterministic: original and models see identical conditioning
        droppable_categories=frozenset(),
        single_tag_dropout=0.0,
    )
    cells = render_panels(selections, compare, pipelines, compare.output_dir / "originals", rules)

    index_path = write_index(compare, cells)
    console.log(f"done: {len(cells)} panels -> {compare.output_dir / 'panels'}")
    console.log(f"contact sheet: {index_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Dataset comparison TOML (see configs/compare/v0.8-dataset.toml)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir from the config")
    args = parser.parse_args()

    compare, selection = load_configs(args.config)
    if args.output_dir is not None:
        compare = replace(compare, output_dir=args.output_dir)
    run(compare, selection)


if __name__ == "__main__":
    main()
