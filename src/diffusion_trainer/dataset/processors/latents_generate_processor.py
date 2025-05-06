"""Prepare latent vectors for the dataset."""

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from PIL import Image
from torchvision import transforms

from diffusion_trainer.dataset.utils import retrieve_image_paths, retrieve_npz_path
from diffusion_trainer.shared import get_progress, logger


@dataclass
class Args:
    """Dataclass to store arguments for the script."""

    img_path: str
    target_path: str
    vae_path: str
    meta_path: str


@dataclass
class WritePayload:
    """Dataclass to store the payload for writing the latent vectors."""

    save_path: Path
    latents: torch.Tensor
    crop_ltrb: tuple[int, int, int, int]
    original_size: tuple[int, int]
    resolution: tuple[int, int]


class SimpleLatentsProcessor:
    """ImageProcessor class to process images and save the latent vectors."""

    def __init__(self, model_name_or_path: str, dtype: torch.dtype | None = None, device: str | torch.device = "cuda") -> None:
        """Initialize the ImageProcessor class."""
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.vae = self.load_vae_model()
        self.predefined_resos = np.array(
            [
                (640, 1536),
                (768, 1344),
                (832, 1216),
                (896, 1152),
                (1024, 1024),
                (1152, 896),
                (1216, 832),
                (1344, 768),
                (1536, 640),
            ],
        )
        self.predefined_ars = np.array([w / h for w, h in self.predefined_resos])

    def load_vae_model(self) -> AutoencoderKL:
        """Load the VAE model."""
        path = Path(self.model_name_or_path)

        if path.suffix == ".safetensors":
            logger.info("Loading vae from file %s", path)
            vae = AutoencoderKL.from_single_file(self.model_name_or_path, torch_dtype=self.dtype)
        else:
            logger.info("Loading vae from folder %s", path)
            vae = AutoencoderKL.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype)
        vae = vae.to(self.device).eval()  # type: ignore
        logger.info("Loaded vae (%s) - dtype = %s, device = %s", self.model_name_or_path, vae.dtype, vae.device)
        return vae

    def select_reso(self, image_width: int, image_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Select the resolution for the image."""
        aspect_ratio = image_width / image_height
        ar_errors = self.predefined_ars - aspect_ratio
        predefined_bucket_id = np.abs(ar_errors).argmin()
        reso = self.predefined_resos[predefined_bucket_id]

        scale = reso[1] / image_height if aspect_ratio > reso[0] / reso[1] else reso[0] / image_width
        resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        return reso, resized_size

    def get_crop_ltrb(self, bucket_reso: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Get the crop left, top, right, and bottom values."""
        bucket_ar, image_ar = (
            bucket_reso[0] / bucket_reso[1],
            image_size[0] / image_size[1],
        )
        resized_width, resized_height = (bucket_reso[1] * image_ar, bucket_reso[1]) if bucket_ar > image_ar else (bucket_reso[0], bucket_reso[0] / image_ar)
        crop_left, crop_top = (bucket_reso[0] - int(resized_width)) // 2, (bucket_reso[1] - int(resized_height)) // 2
        return (
            crop_left,
            crop_top,
            crop_left + int(resized_width),
            crop_top + int(resized_height),
        )

    @staticmethod
    def process(image_path: str) -> np.ndarray:
        """Load the image from the path."""
        image = Image.open(image_path, "r").convert("RGB")
        return np.array(image)

    def resize_and_trim_image(
        self,
        image_np: np.ndarray,
        reso: tuple[int, int],
        resized_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize and trim the image."""
        image_np = cv2.resize(image_np, resized_size, interpolation=cv2.INTER_AREA)

        image_height, image_width = image_np.shape[:2]
        if image_width > reso[0]:
            trim_pos = (image_width - reso[0]) // 2
            image_np = image_np[:, trim_pos : trim_pos + reso[0]]
        if image_height > reso[1]:
            trim_pos = (image_height - reso[1]) // 2
            image_np = image_np[trim_pos : trim_pos + reso[1]]
        return image_np

    @torch.no_grad()
    def prepare_image_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """Prepare the image tensor."""
        np_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        image_tensor = np_to_tensor(image_np)
        image_channels = 3
        if type(image_tensor) is not torch.Tensor:
            msg = "Image tensor is not a torch.Tensor"
            raise TypeError(msg)
        if len(image_tensor.shape) == image_channels:
            image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.dtype).to(self.device)

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode the image using the VAE model."""
        vae_out = self.vae.encode(image_tensor)
        if not isinstance(vae_out, AutoencoderKLOutput):
            msg = "vae_out is not an instance of AutoencoderKLOutput"
            raise TypeError(msg)
        return vae_out.latent_dist.sample()[0]

    @staticmethod
    def save_encoded_image(
        payload: WritePayload,
    ) -> None:
        """Save the encoded image as a npz file."""
        new_npz = {
            "latents": payload.latents.cpu().numpy(),
            "crop_ltrb": payload.crop_ltrb,
            "original_size": payload.original_size,
            "train_resolution": payload.resolution,
        }
        payload.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(payload.save_path, **new_npz)

    @torch.no_grad()
    def process_by_pil(self, image_pil: Image.Image, save_npz_path: Path, save_img_path: str | None = None) -> None:
        """Process the image and save the latent vectors."""
        image_np = np.array(image_pil.convert("RGB"))
        self.process_by_np(image_np, save_npz_path, save_img_path)

    @torch.no_grad()
    def process_by_path(self, image_path: Path, save_npz_path: Path, save_img_path: str | None = None) -> None:
        """Process the image and save the latent vectors."""
        image_np = SimpleLatentsProcessor.process(image_path.as_posix())
        self.process_by_np(image_np, save_npz_path, save_img_path)

    @torch.no_grad()
    def process_by_np(self, image_np: np.ndarray, save_npz_path: Path, save_img_path: str | None = None) -> None:
        """Process the image and save the latent vectors."""
        original_size = image_np.shape[1], image_np.shape[0]
        reso, resized_size = self.select_reso(*original_size)
        image_np = self.resize_and_trim_image(image_np, reso, resized_size)

        if save_img_path is not None:
            Image.fromarray(image_np).save(save_img_path)

        crop_ltrb = self.get_crop_ltrb(np.array(reso), original_size)
        image_tensor = self.prepare_image_tensor(image_np)
        latents = self.encode_image(image_tensor)
        SimpleLatentsProcessor.save_encoded_image(WritePayload(save_npz_path, latents, crop_ltrb, original_size, reso))


from inch.processor import InchProcessor


class Processor:
    def process(self):
        InchProcessor()


class LatentsGenerateProcessor:
    """ImageProcessingPipeline class to process images and save the latent vectors."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        vae_path: str,
        img_path: str,
        target_path: str,
        vae_dtype: torch.dtype | None = None,
        num_reader: int = 4,
        num_writer: int = 4,
    ) -> None:
        """Initialize the ImageProcessingPipeline class."""
        self.ds_path = Path(img_path).absolute()
        self.meta_path = Path(target_path).absolute()
        self.num_reader = num_reader
        self.num_writer = num_writer

        self.reader_threads = []
        self.process_threads = []
        self.writer_threads = []

        self.read_queue = Queue[None | Path](maxsize=num_reader)
        self.process_queue = Queue[None | tuple[Path, np.ndarray]](maxsize=num_reader)
        self.write_queue = Queue[None | WritePayload](
            maxsize=num_writer,
        )

        self.progress_counter = 0
        self.progress_lock = threading.Lock()

        self.gpu_count = torch.cuda.device_count()
        self.processor_list = [SimpleLatentsProcessor(model_name_or_path=vae_path, dtype=vae_dtype, device=f"cuda:{i}") for i in range(self.gpu_count)]

        self.lock = threading.Lock()

    def read_image(
        self,
    ) -> None:
        """Read the image from the queue and process it."""
        skip_existing = True
        while image_path := self.read_queue.get():
            if image_path is None:
                break
            npz_save_path = self.get_npz_save_path(image_path)
            if skip_existing and npz_save_path.exists():
                try:
                    npz = np.load(npz_save_path)
                    reso = npz.get("train_resolution")
                    if reso is not None:
                        # TypeError: Object of type int64 is not JSON serializable
                        reso = reso.tolist()
                        with self.lock:  # Use a lock to ensure thread safety
                            self.progress_counter += 1
                        continue
                except Exception as e:
                    # Failed to read, reprocessing
                    logger.error(f"Failed to load {npz_save_path}, reprocessing...")
                    logger.exception(e)
            try:
                image_np = SimpleLatentsProcessor.process(image_path.as_posix())
            except Exception as e:
                logger.exception(e)
                with self.lock:
                    self.progress_counter += 1
                continue
            self.process_queue.put((npz_save_path, image_np))

    def get_npz_save_path(self, image_path: Path) -> Path:
        """Get the npz save path for the image."""
        base_path = self.ds_path
        relative_path = image_path.relative_to(base_path)
        target_path = self.meta_path / "latents" / relative_path
        return target_path.with_suffix(".npz")

    @torch.no_grad()
    def process_image(
        self,
        processor: SimpleLatentsProcessor,
    ) -> None:
        """Process the image from the queue and write the latent vectors."""
        while True:
            data = self.process_queue.get()
            if data is None:
                break  # Exit signal
            npz_save_path, image_np = data
            original_size = image_np.shape[1], image_np.shape[0]
            reso, resized_size = processor.select_reso(*original_size)
            image_np = processor.resize_and_trim_image(image_np, reso, resized_size)
            crop_ltrb = processor.get_crop_ltrb(np.array(reso), original_size)
            image_tensor = processor.prepare_image_tensor(image_np)
            try:
                latents = processor.encode_image(image_tensor)
                payload = WritePayload(
                    save_path=npz_save_path,
                    latents=latents,
                    crop_ltrb=crop_ltrb,
                    original_size=original_size,
                    resolution=reso,
                )
                self.write_queue.put(payload)
            except Exception as e:
                logger.exception(e)

    def write_npz(self) -> None:
        """Write the latent vectors to the npz file."""
        while True:
            data = self.write_queue.get()
            if data is None:
                break  # Exit signal

            SimpleLatentsProcessor.save_encoded_image(data)
            with self.lock:  # Use a lock to ensure thread safety
                self.progress_counter += 1

    def _setup_threads(self) -> None:
        self.reader_threads = [
            threading.Thread(
                target=self.read_image,
                args=(),
                daemon=True,
            )
            for _ in range(self.num_reader)
        ]

        self.process_threads = [
            threading.Thread(
                target=self.process_image,
                args=(processor,),
                daemon=True,
            )
            for processor in self.processor_list
        ]

        self.writer_threads = [
            threading.Thread(
                target=self.write_npz,
                args=(),
                daemon=True,
            )
            for _ in range(self.num_writer)
        ]

    def _start_threads(self) -> None:
        for t in self.reader_threads:
            t.start()
        for t in self.process_threads:
            t.start()
        for t in self.writer_threads:
            t.start()

    def _join_threads(self) -> None:
        for t in self.reader_threads:
            t.join()
        for t in self.process_threads:
            t.join()
        for t in self.writer_threads:
            t.join()

    def process_ss_meta(self, meta_path: Path | str) -> None:
        """Process the metadata and save it to the meta_path."""
        meta_path = Path(meta_path)
        if Path.exists(meta_path):
            try:
                with meta_path.open() as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to load metadata from {meta_path}, using empty metadata.")
                metadata = {}
        else:
            metadata = {}
        npz_path_list = retrieve_npz_path(self.meta_path, recursive=True)

        for path in npz_path_list:
            relative_path = path.relative_to(self.meta_path).with_suffix("").as_posix()
            if metadata.get(relative_path) is None:
                metadata[relative_path] = {}
            metadata[relative_path]["train_resolution"] = np.load(path)["train_resolution"].tolist()
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)

    def __call__(self) -> None:
        """Run the image processing pipeline."""
        self._setup_threads()
        self._start_threads()

        logger.info("Retrieving image paths...")
        t = time.time()
        image_path_list = list(retrieve_image_paths(self.ds_path, recursive=True))
        total_processes = len(image_path_list)
        t = time.time() - t
        logger.info(f"Finded {total_processes} images. Used time: {t:.4f} seconds.")

        with get_progress() as progress:
            task = progress.add_task("Generating Latents...", total=total_processes)

            for path in image_path_list:
                self.read_queue.put(path)
                progress.update(task, advance=1)

            while self.progress_counter < total_processes:
                with self.progress_lock:
                    progress.update(task, completed=self.progress_counter)

        for _ in range(self.num_writer):
            self.write_queue.put(None)
        for _ in range(self.num_reader):
            self.read_queue.put(None)
        for _ in range(len(self.processor_list)):
            self.process_queue.put(None)

        self._join_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    args = parser.parse_args()
    args = Args(**vars(args))

    img_path = args.img_path
    meta_path = args.meta_path
    vae_path = args.vae_path
    target_path = args.target_path

    pipeline = LatentsGenerateProcessor(vae_path=vae_path, img_path=img_path, target_path=target_path)

    pipeline.__call__()
    pipeline.process_ss_meta(meta_path)
