"""Main."""

import logging

import torch
from rich.logging import RichHandler

from diffusion_trainer.dataset import CreateParquetProcessor, LatentsGenerateProcessor, TaggingProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

img_path = R"./test_dataset"
target_path = R"./cache"

vae_path = R"https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

latent_generator = LatentsGenerateProcessor(vae_path=vae_path, img_path=img_path, target_path=target_path, vae_dtype=torch.float16)
latent_generator()

tagger = TaggingProcessor(img_path, target_path, num_workers=4, skip_existing=True)
tagger()

create_parquet = CreateParquetProcessor(target_path)
create_parquet()
