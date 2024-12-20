import logging
from dataclasses import dataclass

import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset
from diffusion_trainer.shared import get_progress

logger = logging.getLogger(__name__)


@dataclass
class DiffusionBatchItem:
    img_latents: torch.Tensor
    crop_ltrb: torch.Tensor
    original_size: torch.Tensor
    train_resolution: torch.Tensor
    caption: list[str]
    tags: list[str]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
    dataset = DiffusionDataset.from_metadata(R"E:\dataset-demo\meta.json")
    batch_size = 16
    sampler = BucketBasedBatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=DiffusionDataset.collate_fn)  # type: ignore
    n_epochs = 10
    progress = get_progress()
    with progress:
        total_task = progress.add_task("Total Progress", total=n_epochs * len(data_loader))
        for epoch in range(n_epochs):
            for step, batch in enumerate(progress.track(data_loader, description=f"Epoch {epoch+1}")):
                if not isinstance(batch, DiffusionBatch):
                    msg = "Expected DiffusionBatch, got something else."
                    raise TypeError(msg)
                progress.update(total_task, advance=1)
                logger.debug(
                    "Step %d, Batch: %s %s %s",
                    step,
                    batch.original_size[0],
                    batch.crop_ltrb[0],
                    batch.train_resolution[0],
                )
