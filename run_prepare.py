"""Main data preparation script for diffusion training."""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from rich.logging import RichHandler

from diffusion_trainer.dataset import CreateParquetProcessor, LatentsGenerateProcessor, TaggingProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def get_default_dtype() -> torch.dtype:
    """Get the best default dtype based on hardware support."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for diffusion training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Input directory containing images to process",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Output directory for processed data (latents, tags, metadata)",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to VAE model (local file/directory or remote URL)",
    )

    # LatentsGenerateProcessor options
    parser.add_argument(
        "--vae_dtype",
        type=str,
        choices=["fp16", "fp32", "bf16", "float16", "float32", "bfloat16"],
        default=None,
        help="VAE data type (auto-detects best option if not specified)",
    )
    parser.add_argument(
        "--num_reader",
        type=int,
        default=4,
        help="Number of reader threads for latents generation",
    )
    parser.add_argument(
        "--num_writer",
        type=int,
        default=4,
        help="Number of writer threads for latents generation",
    )

    # TaggingProcessor options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for tagging",
    )
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=0.35,
        help="General tag threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=0.9,
        help="Character tag threshold (0.0-1.0)",
    )

    # CreateParquetProcessor options
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of worker threads for parquet creation",
    )

    # Pipeline control options
    parser.add_argument(
        "--skip_latents",
        action="store_true",
        help="Skip latents generation step",
    )
    parser.add_argument(
        "--skip_tagging",
        action="store_true",
        help="Skip tagging step",
    )
    parser.add_argument(
        "--skip_parquet",
        action="store_true",
        help="Skip parquet creation step",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip existing files (reprocess all)",
    )

    args = parser.parse_args()

    # Validate arguments
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error("Image directory '%s' does not exist.", args.image_path)
        sys.exit(1)

    if not image_path.is_dir():
        logger.error("Image path '%s' is not a directory.", args.image_path)
        return

    # Validate thresholds
    if not 0.0 <= args.general_threshold <= 1.0:
        logger.error("general_threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if not 0.0 <= args.character_threshold <= 1.0:
        logger.error("character_threshold must be between 0.0 and 1.0")
        sys.exit(1)

    # Setup paths
    target_path = Path(args.target_path)
    target_latents_path = target_path / "latents"
    target_tags_path = target_path / "tags"

    # Parse dtype
    vae_dtype = None
    if args.vae_dtype:
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        vae_dtype = dtype_map[args.vae_dtype]
    else:
        vae_dtype = get_default_dtype()
        logger.info("Auto-detected dtype: %s", vae_dtype)

    # Print configuration
    logger.info("🚀 Data Preparation Pipeline Configuration")
    logger.info("  📁 Image Directory: %s", image_path)
    logger.info("  📂 Target Directory: %s", target_path)
    logger.info("  🤖 VAE Model: %s", args.vae_path)
    logger.info("  📊 VAE Data Type: %s", vae_dtype)
    logger.info("  🏷️  General Threshold: %.2f", args.general_threshold)
    logger.info("  👤 Character Threshold: %.2f", args.character_threshold)
    logger.info("  ⏭️  Skip Existing: %s", not args.no_skip_existing)

    pipeline_start = time.time()

    # Step 1: Generate Latents
    if not args.skip_latents:
        logger.info("\n📊 Step 1: Generating latent vectors...")
        try:
            latent_generator = LatentsGenerateProcessor(
                vae_path=args.vae_path,
                img_path=str(image_path),
                target_path=str(target_latents_path),
                vae_dtype=vae_dtype,
                num_reader=args.num_reader,
                num_writer=args.num_writer,
            )
            latent_generator()
            logger.info("✅ Latents generation completed")
        except Exception:
            logger.exception("❌ Error during latents generation: %s")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping latents generation")

    # Step 2: Generate Tags
    if not args.skip_tagging:
        logger.info("\n🏷️  Step 2: Generating image tags...")
        try:
            tagger = TaggingProcessor(
                img_path=str(image_path),
                target_path=str(target_tags_path),
                num_workers=args.num_workers,
                general_threshold=args.general_threshold,
                character_threshold=args.character_threshold,
                skip_existing=not args.no_skip_existing,
            )
            tagger()
            logger.info("✅ Tagging completed")
        except Exception:
            logger.exception("❌ Error during tagging: %s")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping tagging")

    # Step 3: Create Parquet Metadata
    if not args.skip_parquet:
        logger.info("\n📦 Step 3: Creating metadata parquet...")
        try:
            # Update CreateParquetProcessor to handle our structure
            create_parquet = CreateParquetProcessor(target_path)
            create_parquet(max_workers=args.max_workers)
            logger.info("✅ Parquet creation completed")
        except Exception:
            logger.exception("❌ Error during parquet creation: %s")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping parquet creation")

    pipeline_end = time.time()
    elapsed = pipeline_end - pipeline_start

    logger.info("\n🎉 Data preparation pipeline completed successfully!")
    logger.info("⏱️  Total time: %.2f seconds", elapsed)
    logger.info("📁 Output directory: %s", target_path)

    # Summary of generated files
    if target_latents_path.exists():
        latent_count = sum(1 for _ in target_latents_path.glob("**/*.npz"))
        logger.info("  📊 Latent files: %d", latent_count)

    if target_tags_path.exists():
        tag_count = sum(1 for _ in target_tags_path.glob("**/*.txt"))
        logger.info("  🏷️  Tag files: %d", tag_count)

    parquet_file = target_path / "metadata.parquet"
    if parquet_file.exists():
        logger.info("  📦 Metadata file: %s", parquet_file)


if __name__ == "__main__":
    main()
