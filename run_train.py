"""Stable Diffusion XL Finetuner."""

import logging

from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

if __name__ == "__main__":
    import tomllib
    from pathlib import Path

    from diffusion_trainer.config import SDXLConfig
    from diffusion_trainer.dataset.dataset import DiffusionDataset
    from diffusion_trainer.dataset.latents_generate_processor import LatentsGenerateProcessor
    from diffusion_trainer.dataset.tagging_processor import TaggingProcessor
    from diffusion_trainer.finetune.sdxl import SDXLTuner
    from diffusion_trainer.finetune.utils import str_to_dtype

    logger = logging.getLogger("diffusion-trainer")
    config_path = Path("configs/sdxl_lokr.toml")
    sdxl_config_dict = tomllib.load(config_path.open("rb"))
    sdxl_config = SDXLConfig(**sdxl_config_dict)

    if sdxl_config.image_path is None:
        msg = "Please specify the image path in the config file."
        raise ValueError(msg)
    if sdxl_config.image_path and sdxl_config.skip_prepare_image is False:
        logger.info("Prepare image from %s", sdxl_config.image_path)
        if not sdxl_config.vae_path:
            msg = "Please specify the vae_path in the config file."
            raise ValueError(msg)

        if not sdxl_config.meta_path:
            sdxl_config.meta_path = (Path(sdxl_config.image_path) / "metadata").as_posix()
            logger.info("Metadata path not set. Using %s as metadata path.", sdxl_config.meta_path)
        vae_dtype = str_to_dtype(sdxl_config.vae_dtype)
        latents_processor = LatentsGenerateProcessor(
            vae_path=sdxl_config.vae_path,
            img_path=sdxl_config.image_path,
            meta_path=sdxl_config.meta_path,
            vae_dtype=vae_dtype,
        )
        latents_processor()

        tagging_processor = TaggingProcessor(img_path=sdxl_config.image_path, num_workers=1)
        tagging_processor()

    dataset = DiffusionDataset.from_filesystem(sdxl_config)

    tuner = SDXLTuner(sdxl_config)
    tuner.train(dataset)
    # if sdxl_config.ss_meta_path is None:
    #     msg = "Please specify the meta path in the config file."
    #     raise ValueError(msg)
    # dataset = DiffusionDataset.from_ss_meta(meta_dir=sdxl_config.ss_meta_path)
