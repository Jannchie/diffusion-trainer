"""Prepare latent vectors for the dataset using SHA256-based directory structure."""

import argparse
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from PIL import Image
from torchvision import transforms

from diffusion_trainer.dataset.utils import retrieve_image_paths
from diffusion_trainer.shared import get_progress, logger


@dataclass
class WritePayload:
    """Payload for writing latent vectors."""

    save_path: Path
    latents: torch.Tensor
    crop_ltrb: tuple[int, int, int, int]
    original_size: tuple[int, int]
    resolution: tuple[int, int]


class SimpleLatentsProcessor:
    """Simple latents processor using SHA256-based storage."""

    def __init__(self, model_name_or_path: str, dtype: torch.dtype | None = None, device: str | torch.device = "cuda") -> None:
        """Initialize the processor."""
        self.model_name_or_path = resolve_vae_path(model_name_or_path)
        self.device = device
        self.dtype = dtype
        self.vae = self.load_vae_model()

        # Predefined resolutions for bucketing (same as original)
        self.predefined_resos = np.array([
            (640, 1536),
            (768, 1344),
            (832, 1216),
            (896, 1152),
            (1024, 1024),
            (1152, 896),
            (1216, 832),
            (1344, 768),
            (1536, 640),
        ])
        self.predefined_ars = np.array([w / h for w, h in self.predefined_resos])

    def load_vae_model(self) -> AutoencoderKL:
        """Load the VAE model."""
        path = Path(self.model_name_or_path)

        if path.suffix == ".safetensors":
            logger.info("Loading VAE from file %s", path)
            vae = AutoencoderKL.from_single_file(self.model_name_or_path, torch_dtype=self.dtype)
        else:
            logger.info("Loading VAE from folder %s", path)
            vae = AutoencoderKL.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype)

        vae = vae.to(self.device).eval()  # type: ignore
        logger.info("Loaded VAE (%s) - dtype = %s, device = %s", self.model_name_or_path, vae.dtype, vae.device)
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
        bucket_ar, image_ar = bucket_reso[0] / bucket_reso[1], image_size[0] / image_size[1]
        resized_width, resized_height = (bucket_reso[1] * image_ar, bucket_reso[1]) if bucket_ar > image_ar else (bucket_reso[0], bucket_reso[0] / image_ar)
        crop_left, crop_top = (bucket_reso[0] - int(resized_width)) // 2, (bucket_reso[1] - int(resized_height)) // 2
        return crop_left, crop_top, crop_left + int(resized_width), crop_top + int(resized_height)

    @staticmethod
    def process(image_path: str | Path) -> np.ndarray:
        """Load the image from the path."""
        image = Image.open(image_path, "r").convert("RGB")
        return np.array(image)

    def resize_and_trim_image(self, image_np: np.ndarray, reso: tuple[int, int], resized_size: tuple[int, int]) -> np.ndarray:
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
        if not isinstance(image_tensor, torch.Tensor):
            msg = "Expected torch.Tensor from transforms"
            raise TypeError(msg)
        if len(image_tensor.shape) == 3:
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

    @torch.no_grad()
    def process_by_pil(self, image: Image.Image, save_npz_path: Path) -> None:
        """Process a PIL image and save encoded latents to NPZ file."""
        # Convert PIL image to numpy array
        image_np = np.array(image.convert("RGB"))

        # Get original image size
        original_size = image_np.shape[1], image_np.shape[0]

        # Select resolution and resize image
        reso, resized_size = self.select_reso(*original_size)
        image_np = self.resize_and_trim_image(image_np, reso, resized_size)

        # Calculate crop parameters
        crop_ltrb = self.get_crop_ltrb(np.array(reso), original_size)

        # Prepare tensor and encode
        image_tensor = self.prepare_image_tensor(image_np)
        latents = self.encode_image(image_tensor)

        # Convert tensor to compatible dtype before converting to numpy
        latents_tensor = latents
        if latents_tensor.dtype == torch.bfloat16:
            # Convert bfloat16 to float32 (NumPy doesn't support bfloat16)
            latents_tensor = latents_tensor.float()
        # fp16 (float16) is supported by NumPy, so we keep it as-is

        latents_np = latents_tensor.cpu().numpy()

        # Prepare NPZ data
        npz_data = {
            "latents": latents_np,
            "crop_ltrb": crop_ltrb,
            "original_size": original_size,
            "train_resolution": reso,
        }

        # Create directory and save NPZ file
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz_path, **npz_data)


class LatentsGenerateProcessor:
    """Latents processor with SHA256-based directory structure (replaces original)."""

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
        """Initialize the processor (compatible with original interface)."""
        self.ds_path = Path(img_path).absolute()
        self.meta_path = Path(target_path).absolute()  # Keep original name for compatibility
        self.target_path = self.meta_path  # SHA256-based output
        self.num_reader = num_reader
        self.num_writer = num_writer

        # Create output directory
        self.target_path.mkdir(parents=True, exist_ok=True)

        # Threading components
        self.reader_threads = []
        self.process_threads = []
        self.writer_threads = []

        self.read_queue = Queue[Path | None](maxsize=num_reader * 2)
        self.process_queue = Queue[tuple[Path, np.ndarray] | None](maxsize=num_reader)
        self.write_queue = Queue[WritePayload | None](maxsize=num_writer)

        self.progress_counter = 0
        self.progress_lock = threading.Lock()

        # Create multiple GPU processors
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.processor_list = [
            SimpleLatentsProcessor(
                model_name_or_path=vae_path,
                dtype=vae_dtype,
                device=f"cuda:{i}" if torch.cuda.is_available() else "cpu",
            )
            for i in range(self.gpu_count)
        ]

        self.lock = threading.Lock()

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        hash_sha256 = hashlib.sha256()
        with image_path.open("rb") as f:
            while chunk := f.read(65536):  # 64KB chunks
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_npz_save_path(self, image_path: Path) -> Path:
        """Get NPZ save path with SHA256-based directory structure."""
        sha256_hash = self.calculate_sha256(image_path)
        dir1 = sha256_hash[:2]
        dir2 = sha256_hash[2:4]
        return self.target_path / dir1 / dir2 / f"{sha256_hash}.npz"

    def read_image(self) -> None:
        """Read images and check for existing files."""
        skip_existing = True
        while image_path := self.read_queue.get():
            if image_path is None:
                break

            try:
                npz_save_path = self.get_npz_save_path(image_path)

                # Check if output file already exists
                if skip_existing and npz_save_path.exists():
                    try:
                        # Verify the existing file is valid
                        npz = np.load(npz_save_path)
                        if "train_resolution" in npz:
                            with self.progress_lock:
                                self.progress_counter += 1
                            continue
                    except Exception:
                        logger.warning("Corrupted file %s, reprocessing...", npz_save_path)

                # Load and process image
                image_np = SimpleLatentsProcessor.process(image_path)
                self.process_queue.put((image_path, image_np))

            except Exception as e:
                logger.error("Error reading %s: %s", image_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def process_image(self, processor: SimpleLatentsProcessor) -> None:
        """Process images using VAE encoder."""
        while True:
            data = self.process_queue.get()
            if data is None:
                break

            image_path, image_np = data

            try:
                # Get original image size
                original_size = image_np.shape[1], image_np.shape[0]

                # Select resolution and resize image
                reso, resized_size = processor.select_reso(*original_size)
                image_np = processor.resize_and_trim_image(image_np, reso, resized_size)

                # Calculate crop parameters
                crop_ltrb = processor.get_crop_ltrb(np.array(reso), original_size)

                # Prepare tensor and encode
                image_tensor = processor.prepare_image_tensor(image_np)
                latents = processor.encode_image(image_tensor)

                # Get save path
                npz_save_path = self.get_npz_save_path(image_path)

                # Create payload for writing
                payload = WritePayload(
                    save_path=npz_save_path,
                    latents=latents,
                    crop_ltrb=crop_ltrb,
                    original_size=original_size,
                    resolution=reso,
                )

                self.write_queue.put(payload)

            except Exception as e:
                logger.error("Error processing %s: %s", image_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def write_npz(self) -> None:
        """Write encoded latents to NPZ files."""
        while True:
            payload = self.write_queue.get()
            if payload is None:
                break

            try:
                # Convert tensor to compatible dtype before converting to numpy
                latents_tensor = payload.latents
                if latents_tensor.dtype == torch.bfloat16:
                    # Convert bfloat16 to float32 (NumPy doesn't support bfloat16)
                    latents_tensor = latents_tensor.float()
                # fp16 (float16) is supported by NumPy, so we keep it as-is

                latents_np = latents_tensor.cpu().numpy()

                new_npz = {
                    "latents": latents_np,
                    "crop_ltrb": payload.crop_ltrb,
                    "original_size": payload.original_size,
                    "train_resolution": payload.resolution,
                }

                payload.save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(payload.save_path, **new_npz)

                with self.progress_lock:
                    self.progress_counter += 1

            except Exception as e:
                logger.error("Error writing %s: %s", payload.save_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def __call__(self) -> None:  # noqa: C901, PLR0912
        """Run the processing pipeline."""
        # Get all image paths
        logger.info("Scanning for images in %s", self.ds_path)
        image_paths = list(retrieve_image_paths(self.ds_path, recursive=True))
        total_images = len(image_paths)

        if total_images == 0:
            logger.info("No images found in %s", self.ds_path)
            return

        logger.info("Found %d images to process", total_images)

        # Create and start threads
        self.reader_threads = [threading.Thread(target=self.read_image, daemon=True) for _ in range(self.num_reader)]
        self.process_threads = [
            threading.Thread(target=self.process_image, args=(processor,), daemon=True)
            for processor in self.processor_list
        ]
        self.writer_threads = [threading.Thread(target=self.write_npz, daemon=True) for _ in range(self.num_writer)]

        # Start all threads
        for thread in self.reader_threads + self.process_threads + self.writer_threads:
            thread.start()

        # Process with progress tracking
        with get_progress() as progress:
            task = progress.add_task("Processing images...", total=total_images)

            # Add image paths to queue in batches
            batch_size = 1000
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                for image_path in batch:
                    self.read_queue.put(image_path)

                # Small delay to avoid overwhelming the queue
                if i > 0:
                    time.sleep(0.1)

            # Monitor progress
            last_count = 0
            stall_count = 0
            while self.progress_counter < total_images:
                current_count = self.progress_counter
                progress.update(task, completed=current_count)

                # Check for stalled processing
                if current_count == last_count:
                    stall_count += 1
                    if stall_count > 100:  # 10 seconds
                        logger.warning("Processing seems stalled at %d/%d", current_count, total_images)
                        stall_count = 0
                else:
                    stall_count = 0

                last_count = current_count
                time.sleep(0.1)

            # Signal threads to stop
            for _ in range(self.num_reader):
                self.read_queue.put(None)
            for _ in range(len(self.processor_list)):
                self.process_queue.put(None)
            for _ in range(self.num_writer):
                self.write_queue.put(None)

            # Wait for all threads to complete
            for thread in self.reader_threads + self.process_threads + self.writer_threads:
                thread.join()

            progress.update(task, completed=total_images)

        logger.info("Successfully processed %d images", total_images)


def is_remote_url(path: str) -> bool:
    """Check if the path is a remote URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def convert_hf_url_to_direct(url: str) -> str:
    """Convert Hugging Face blob URL to direct download URL."""
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def download_file_with_progress(url: str, target_path: Path) -> None:
    """Download a file from URL with progress bar."""
    download_url = convert_hf_url_to_direct(url)

    logger.info("Downloading %s to %s", url, target_path)

    with httpx.stream("GET", download_url, follow_redirects=True) as response:
        response.raise_for_status()

        with target_path.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)

    logger.info("Downloaded to %s", target_path)


def resolve_vae_path(vae_path: str) -> str:
    """Resolve VAE path, downloading if it's a remote URL."""
    if is_remote_url(vae_path):
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "diffusion_trainer" / "vae"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from URL
        parsed_url = urlparse(vae_path)
        filename = Path(parsed_url.path).name
        if not filename.endswith((".safetensors", ".ckpt", ".pt", ".pth")):
            filename = "vae_model.safetensors"

        cached_path = cache_dir / filename

        # Download if not cached
        if not cached_path.exists():
            logger.info("Downloading VAE model from %s", vae_path)
            download_file_with_progress(vae_path, cached_path)
        else:
            logger.info("Using cached VAE model at %s", cached_path)

        return str(cached_path)
    return vae_path


def get_default_dtype() -> torch.dtype:
    """Get the best default dtype based on hardware support."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latents processor with SHA256-based directory structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE model (local or remote URL)")
    parser.add_argument("--img_path", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--target_path", type=str, required=True, help="Output directory for latent files")
    parser.add_argument("--num_reader", type=int, default=4, help="Number of reader threads")
    parser.add_argument("--num_writer", type=int, default=4, help="Number of writer threads")
    parser.add_argument("--vae_dtype", type=str, choices=["fp16", "fp32", "bf16"], default=None, help="VAE dtype")

    args = parser.parse_args()

    # Parse dtype
    vae_dtype = None
    if args.vae_dtype:
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        vae_dtype = dtype_map[args.vae_dtype]
    else:
        vae_dtype = get_default_dtype()
        logger.info("Auto-detected dtype: %s", vae_dtype)

    processor = LatentsGenerateProcessor(
        vae_path=args.vae_path,
        img_path=args.img_path,
        target_path=args.target_path,
        vae_dtype=vae_dtype,
        num_reader=args.num_reader,
        num_writer=args.num_writer,
    )

    start_time = time.time()
    processor()
    end_time = time.time()

    elapsed = end_time - start_time
    logger.info("Processing completed in %.2f seconds", elapsed)
