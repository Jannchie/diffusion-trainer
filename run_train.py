"""Stable Diffusion XL Finetuner."""

import logging

from rich.logging import RichHandler

from diffusion_trainer.dataset.dataset import DiffusionDataset
from diffusion_trainer.finetune.sdxl import SDXLTuner

logging.basicConfig(level=logging.DEBUG, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

if __name__ == "__main__":
    model_path = R"E:/webui_forge_cu121_torch21/webui/models/Stable-diffusion/creative-xl-0.9-b2.safetensors"

    dataset = DiffusionDataset.from_metadata(R"E:\dataset-demo\meta.json")
    tuner = SDXLTuner(
        model_path=model_path,
        dataset=dataset,
    )
    tuner()
