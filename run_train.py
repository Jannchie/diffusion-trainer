"""Stable Diffusion XL Finetuner."""

import logging

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

if __name__ == "__main__":
    import tomllib
    from argparse import ArgumentParser
    from pathlib import Path

    from diffusion_trainer.config import SDXLConfig
    from diffusion_trainer.finetune.sdxl import SDXLTuner

    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    logger.info('Loading config from "%s"', args.config)

    config_path = Path(args.config)
    sdxl_config_dict = tomllib.load(config_path.open("rb"))
    sdxl_config = SDXLConfig(**sdxl_config_dict)
    tuner = SDXLTuner(sdxl_config)

    tuner.train()
