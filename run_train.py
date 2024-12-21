"""Stable Diffusion XL Finetuner."""

import logging

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


if __name__ == "__main__":
    import tomllib
    from pathlib import Path

    from diffusion_trainer.dataset.dataset import DiffusionDataset
    from diffusion_trainer.finetune.sdxl import SDXLConfig, SDXLTuner

    config_path = Path("configs/sdxl_lokr.toml")
    sdxl_config_dict = tomllib.load(config_path.open("rb"))
    sdxl_config = SDXLConfig(**sdxl_config_dict)

    dataset = DiffusionDataset.from_metadata(sdxl_config.dataset_meta_path, ds_path=sdxl_config.dataset_path)

    tuner = SDXLTuner(config=sdxl_config)
    tuner(dataset=dataset)
