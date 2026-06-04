"""Export a prepared dataset directory into bucket-grouped tar shards for sharing.

Example:
    uv run python scripts/export_dataset.py --dataset-dir datasets/suzume_xl --output-dir exports/suzume_xl \
        --vae-name madebyollin/sdxl-vae-fp16-fix

The output directory can be uploaded to a HuggingFace dataset repo as-is:
    hf upload <user>/<repo> exports/suzume_xl --repo-type dataset
"""

import argparse
import logging

from rich.logging import RichHandler

from diffusion_trainer.dataset.sharing import DEFAULT_SHARD_SIZE_BYTES, export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a prepared dataset into bucket-grouped tar shards")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Prepared dataset directory (contains metadata.parquet and latents/)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the shareable dataset")
    parser.add_argument("--shard-size-mb", type=int, default=DEFAULT_SHARD_SIZE_BYTES >> 20, help="Target size of each tar shard in MiB")
    parser.add_argument("--name", type=str, default=None, help="Dataset name for the generated card (defaults to the output directory name)")
    parser.add_argument("--vae-name", type=str, default=None, help="VAE identifier recorded in the dataset card (strongly recommended)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
    export_dataset(
        args.dataset_dir,
        args.output_dir,
        shard_size_bytes=args.shard_size_mb << 20,
        dataset_name=args.name,
        vae_name=args.vae_name,
    )


if __name__ == "__main__":
    main()
