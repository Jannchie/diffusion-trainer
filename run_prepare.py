"""Main."""

import logging

from rich.logging import RichHandler

from diffusion_trainer.dataset import CleanPromptProcessor, LatentsGenerateProcessor, TaggingProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

ds_path = R"../datasets/nijijourney-v6-520k-raw/train"
target_path = R"../datasets/nijijourney-v6-520k-raw/latents"
meta_path = R"../datasets/nijijourney-v6-520k-raw/meta_with_caption.json"

vae_path = "madebyollin/sdxl-vae-fp16-fix"
if True:
    import torch

    vae_path = R"C:\Users\Jannchie\Downloads\AOM3B2_orangemixs_fp16\vae"

    latent_generator = LatentsGenerateProcessor(vae_path, ds_path, target_path, vae_dtype=torch.float16)
    latent_generator()
    latent_generator.process_ss_meta(meta_path)

tagger = TaggingProcessor(meta_path, ds_path, num_workers=4, skip_existing=True)
tagger()

cleanner = CleanPromptProcessor(meta_path)
cleanner()
