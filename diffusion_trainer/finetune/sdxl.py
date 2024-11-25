"""Fintunner for Stable Diffusion XL model."""

import secrets
from logging import getLogger
from pathlib import Path

import torch
from rich.status import Status
from transformers import CLIPTokenizer

from diffusion_trainer.finetune.utils import prepare_accelerator
from diffusion_trainer.finetune.utils.sdxl import load_models_from_sdxl_checkpoint

logger = getLogger("diffusion_trainer.fineturn.sdxl")

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


def prepare_tokenizer() -> tuple[CLIPTokenizer, CLIPTokenizer]:
    """Prepare the tokenizer."""
    tokenizer_1: CLIPTokenizer = CLIPTokenizer.from_pretrained(TOKENIZER1_PATH)
    tokenizer_2: CLIPTokenizer = CLIPTokenizer.from_pretrained(TOKENIZER2_PATH)

    # TODO: Check if this is necessary
    tokenizer_2.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer
    return tokenizer_1, tokenizer_2


def load_model(path: Path, device: torch.device, dtype: torch.dtype) -> None:
    logger.info("Loading models from %s", path)
    with Status(f"Loading models from {path}"):
        (text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info) = load_models_from_sdxl_checkpoint(path, str(device), dtype)


class SDXLTuner:
    """Finetune Stable Diffusion XL model."""

    def __init__(self, *, model_path: str, seed: None | int = None) -> None:
        """Initialize."""
        self.model_path = Path(model_path)
        self.seed = seed if seed is not None else secrets.randbelow(1_000_000_000)
        self.tokenizer_1, self.tokenizer_2 = prepare_tokenizer()
        self.accelerator = prepare_accelerator()
        self.vae_dtype = torch.float16
        self.save_dtype = torch.float16
        self.weight_dtype = torch.float16
        load_model(self.model_path, self.accelerator.device, self.vae_dtype)

    def __call__(self) -> None:
        """Run the finetuning process."""
