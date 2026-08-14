"""Input discovery for evaluation: image sets and their prompts.

Lives in the module (not the CLI script) so future eval entry points — e.g. an
in-training eval hook — share the same input contract.
"""

import json
from pathlib import Path

from diffusion_trainer.dataset.utils import retrieve_image_paths


def list_images(directory: Path, limit: int | None = None) -> list[Path]:
    """Sorted image paths under ``directory``, honoring the repo-wide extension list and hidden-file filtering."""
    paths = sorted(retrieve_image_paths(directory))
    if not paths:
        msg = f"no images found under {directory}"
        raise FileNotFoundError(msg)
    return paths[:limit]


def resolve_prompts(gen_images: list[Path], prompts_file: Path | None = None) -> dict[Path, str]:
    """Map each generated image to its prompt; iteration order follows ``gen_images``.

    A ``prompts_file`` JSON object ({filename: prompt}) wins over a sidecar
    ``<name>.txt`` next to the image. Images with neither are simply absent
    from the result (they still count toward distribution metrics).
    """
    mapping = json.loads(prompts_file.read_text()) if prompts_file else {}
    prompts: dict[Path, str] = {}
    for image in gen_images:
        if image.name in mapping:
            prompts[image] = mapping[image.name]
        elif (sidecar := image.with_suffix(".txt")).exists():
            prompts[image] = sidecar.read_text().strip()
    return prompts
