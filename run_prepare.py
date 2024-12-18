"""Main."""

import logging

import torch
from rich.logging import RichHandler

from diffusion_trainer.dataset import CleanPromptProcessor, LatentsGenerateProcessor, TaggingProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


ds_path = R"E:\pictoria\server\demo"
target_path = R"E:\dataset-demo"
meta_path = R"E:\dataset-demo-sd\meta1.json"

vae_path = "madebyollin/sdxl-vae-fp16-fix"
# vae_path = R"C:\Users\Jannchie\Downloads\AOM3B2_orangemixs_fp16\vae"

latent_generator = LatentsGenerateProcessor(vae_path, ds_path, target_path, dtype=torch.float16)
latent_generator()
latent_generator.process_metadata(meta_path)

tagger = TaggingProcessor(meta_path, ds_path, num_workers=4, skip_existing=True)
tagger()

cleanner = CleanPromptProcessor(meta_path)
cleanner()
