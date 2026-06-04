"""Import an exported dataset (local directory or HuggingFace dataset repo) for training.

Examples:
    uv run python scripts/import_dataset.py --source exports/suzume_xl --target-dir datasets/suzume_xl
    uv run python scripts/import_dataset.py --source someuser/suzume-xl-latents --target-dir datasets/suzume_xl

Afterwards point `dataset_path` in the training config to the target directory.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.logging import RichHandler

from diffusion_trainer.dataset.sharing import import_dataset

logger = logging.getLogger("diffusion_trainer.dataset")


def resolve_source(source: str, revision: str | None) -> Path:
    """Resolve a local directory as-is; otherwise treat the source as a HF dataset repo id."""
    local = Path(source)
    if local.is_dir():
        return local
    logger.info("Source %s is not a local directory, downloading from HuggingFace Hub...", source)
    return Path(snapshot_download(repo_id=source, repo_type="dataset", revision=revision))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an exported dataset into the local prepared-dataset layout")
    parser.add_argument("--source", type=str, required=True, help="Exported dataset directory, or a HuggingFace dataset repo id")
    parser.add_argument("--target-dir", type=str, required=True, help="Target dataset directory (used as dataset_path in training configs)")
    parser.add_argument("--revision", type=str, default=None, help="Optional repo revision when downloading from HuggingFace Hub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
    import_dataset(resolve_source(args.source, args.revision), args.target_dir)


if __name__ == "__main__":
    main()
