"""Stable Diffusion XL Finetuner."""

import logging

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(name)s :: %(message)s", datefmt="[%X]", handlers=[RichHandler()])

if __name__ == "__main__":
    import tomllib
    from argparse import ArgumentParser
    from pathlib import Path

    from diffusion_trainer.config import SDXLConfig
    from diffusion_trainer.finetune.sdxl import SDXLTuner

    logger = logging.getLogger(__name__)

    diffusers_logger = logging.getLogger("diffusers")
    diffusers_logger.setLevel(logging.ERROR)

    lycoris_logger = logging.getLogger("LyCORIS")
    lycoris_logger.setLevel(logging.ERROR)

    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/sd15.toml", help="Path to the config file.")
    parser.add_argument("--model_family", type=str, default="sd15", choices=["sd15", "sdxl"])
    args = parser.parse_args()

    logger.info('Loading config from "%s"', args.config)
    config_path = Path(args.config)
    config_dict = tomllib.load(config_path.open("rb"))

    model_family = args.model_family
    if model_family == "sdxl":
        config = SDXLConfig(**config_dict)
        tuner = SDXLTuner(config)
        tuner.train()
    elif model_family == "sd15":
        from diffusion_trainer.config import SD15Config
        from diffusion_trainer.finetune.sd15 import SD15Tuner

        config = SD15Config(**config_dict)
        tuner = SD15Tuner(config)
        tuner.train()
