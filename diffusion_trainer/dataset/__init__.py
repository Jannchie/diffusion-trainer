"""Module contains the dataset classes for the diffusion trainer."""

from .clean_prompt_processor import CleanPromptProcessor
from .latents_generate_processor import LatentsGenerateProcessor
from .tagging_processor import TaggingProcessor

__all__ = [
    "CleanPromptProcessor",
    "LatentsGenerateProcessor",
    "TaggingProcessor",
]
