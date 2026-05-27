"""Module contains the dataset classes for the diffusion trainer."""

import pillow_avif  # noqa: F401

from .processors.create_parquet_processor import CreateParquetProcessor
from .processors.latents_generate_processor import LatentsGenerateProcessor
from .processors.tagging_processor import TaggingProcessor

__all__ = [
    "CreateParquetProcessor",
    "LatentsGenerateProcessor",
    "TaggingProcessor",
]
