"""Prepare latent vectors for the dataset using SHA256-based directory structure."""

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from PIL import Image
from torchvision import transforms

from diffusion_trainer.dataset.processors.base import ThreadedPipelineProcessor
from diffusion_trainer.dataset.utils import (
    LATENTS_META_FIELDS,
    calculate_file_sha256,
    load_latents_meta,
    retrieve_image_paths,
    sharded_path,
    write_latents_meta,
)
from diffusion_trainer.shared import logger
from diffusion_trainer.utils.dtype import get_default_dtype, str_to_dtype


@dataclass
class WritePayload:
    """Payload for writing latent vectors. Only built for items that encoded successfully."""

    save_path: Path
    latents: torch.Tensor
    crop_ltrb: tuple[int, int, int, int]
    original_size: tuple[int, int]
    resolution: tuple[int, int]


def meta_row(resolution: tuple[int, int], original_size: tuple[int, int], crop_ltrb: tuple[int, int, int, int]) -> dict[str, list[int]]:
    """Build a latents-metadata row (plain ints for parquet; crop values arrive as np.int64)."""
    return {
        "train_resolution": [int(v) for v in resolution],
        "original_size": [int(v) for v in original_size],
        "crop_ltrb": [int(v) for v in crop_ltrb],
    }


def latents_to_numpy(latents: torch.Tensor) -> np.ndarray:
    """Convert a latents tensor to a NumPy-compatible array.

    NumPy has no bfloat16, so bf16 is widened to fp32; fp16 stays as-is.
    """
    if latents.dtype == torch.bfloat16:
        latents = latents.float()
    return latents.cpu().numpy()


# Bucket tables keyed by base resolution. Every side is a multiple of 64 (the
# UNet needs latent dims divisible by 8) and every area stays within base².
PREDEFINED_RESOS: dict[int, tuple[tuple[int, int], ...]] = {
    512: (
        (320, 768),
        (384, 640),
        (448, 576),
        (512, 512),
        (576, 448),
        (640, 384),
        (768, 320),
    ),
    768: (
        (512, 1152),
        (576, 1024),
        (640, 896),
        (704, 832),
        (768, 768),
        (832, 704),
        (896, 640),
        (1024, 576),
        (1152, 512),
    ),
    1024: (
        (640, 1536),
        (768, 1344),
        (832, 1216),
        (896, 1152),
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (1536, 640),
    ),
}


class SimpleLatentsProcessor:
    """Simple latents processor using SHA256-based storage."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
        base_resolution: int = 1024,
    ) -> None:
        """Initialize the processor."""
        self.model_name_or_path = resolve_vae_path(model_name_or_path)
        self.device = device
        self.dtype = dtype
        self.vae = self.load_vae_model()

        if base_resolution not in PREDEFINED_RESOS:
            msg = f"Unsupported base_resolution {base_resolution}; choose one of {sorted(PREDEFINED_RESOS)}"
            raise ValueError(msg)
        self.predefined_resos = np.array(PREDEFINED_RESOS[base_resolution])
        self.predefined_ars = np.array([w / h for w, h in self.predefined_resos])

    def load_vae_model(self) -> AutoencoderKL:
        """Load the VAE model.

        Accepts a standalone VAE (file or folder) as well as a full pipeline
        repo, in which case the weights live under ``vae/``. Pointing at the
        pipeline repo is the normal way to prepare latents for a model whose VAE
        isn't published separately — e.g. the 16-channel one Lumina 2 uses.
        """
        path = Path(self.model_name_or_path)

        if path.suffix == ".safetensors":
            logger.info("Loading VAE from file %s", path)
            vae = AutoencoderKL.from_single_file(self.model_name_or_path, torch_dtype=self.dtype)
        else:
            logger.info("Loading VAE from folder %s", path)
            try:
                vae = AutoencoderKL.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype)
            except (OSError, ValueError):
                logger.info("No VAE config at the root of %s; retrying its vae/ subfolder", path)
                vae = AutoencoderKL.from_pretrained(self.model_name_or_path, subfolder="vae", torch_dtype=self.dtype)

        vae = vae.to(self.device).eval()  # type: ignore
        logger.info(
            "Loaded VAE (%s) - dtype = %s, device = %s, latent channels = %s",
            self.model_name_or_path,
            vae.dtype,
            vae.device,
            vae.config.get("latent_channels"),
        )
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
    def encode_np(self, image_np: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
        """Bucket, resize and encode an RGB image array.

        Returns ``(latents, crop_ltrb, original_size, train_resolution)``.
        """
        original_size = image_np.shape[1], image_np.shape[0]
        reso, resized_size = self.select_reso(*original_size)
        image_np = self.resize_and_trim_image(image_np, reso, resized_size)
        crop_ltrb = self.get_crop_ltrb(np.array(reso), original_size)
        image_tensor = self.prepare_image_tensor(image_np)
        latents = self.encode_image(image_tensor)
        return latents, crop_ltrb, original_size, (int(reso[0]), int(reso[1]))

    @torch.no_grad()
    def process_by_pil(self, image: Image.Image, save_npz_path: Path) -> dict[str, list[int]]:
        """Encode a PIL image to a latent-only NPZ file and return its metadata row.

        The returned row (``train_resolution`` / ``original_size`` / ``crop_ltrb``)
        must be persisted by the caller (e.g. via ``write_latents_meta``); the NPZ
        itself carries only the latents.
        """
        image_np = np.array(image.convert("RGB"))
        latents, crop_ltrb, original_size, reso = self.encode_np(image_np)

        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz_path, latents=latents_to_numpy(latents))

        return meta_row(reso, original_size, crop_ltrb)


class LatentsGenerateProcessor(ThreadedPipelineProcessor[Path, tuple[Path, np.ndarray], WritePayload]):
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
        skip_existing: bool = True,
        base_resolution: int = 1024,
    ) -> None:
        """Initialize the processor (compatible with original interface)."""
        self.ds_path = Path(img_path).absolute()
        self.meta_path = Path(target_path).absolute()  # Keep original name for compatibility
        self.target_path = self.meta_path  # SHA256-based output
        self.skip_existing = skip_existing
        self.target_path.mkdir(parents=True, exist_ok=True)

        # Single source of truth for per-image metadata (crop/size/resolution):
        # NPZ files carry only the latents, rows accumulate here and are flushed
        # to latents_meta.parquet when the run finishes.
        self.latents_meta = load_latents_meta(self.target_path)
        self.meta_lock = threading.Lock()

        # One VAE processor per GPU; each process thread gets its own (see make_process_worker).
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.processor_list = [
            SimpleLatentsProcessor(
                model_name_or_path=vae_path,
                dtype=vae_dtype,
                device=f"cuda:{i}" if torch.cuda.is_available() else "cpu",
                base_resolution=base_resolution,
            )
            for i in range(gpu_count)
        ]

        super().__init__(
            num_reader=num_reader,
            num_writer=num_writer,
            num_process_workers=len(self.processor_list),
            description="Processing images...",
            poll_interval=0.05,
            stall_ticks=200,
        )

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        return calculate_file_sha256(image_path)

    def get_npz_save_path(self, image_path: Path) -> Path:
        """Get NPZ save path with SHA256-based directory structure."""
        return sharded_path(self.target_path, self.calculate_sha256(image_path), "npz")

    # ----- ThreadedPipelineProcessor stages -----

    def get_items(self) -> list[Path]:
        logger.info("Scanning for images in %s", self.ds_path)
        return list(retrieve_image_paths(self.ds_path, recursive=True))

    def make_process_worker(self, index: int) -> object:
        return self.processor_list[index]

    def read_item(self, item: Path) -> tuple[Path, np.ndarray] | None:
        npz_save_path = self.get_npz_save_path(item)
        if self.skip_existing and npz_save_path.exists():
            key = npz_save_path.stem
            try:
                # A valid cache must carry the latents themselves, not just
                # metadata; a half-written file would otherwise be skipped and
                # only blow up later during training.
                with np.load(npz_save_path) as npz:
                    if "latents" in npz:
                        if key in self.latents_meta:
                            return None  # valid existing latents with known metadata
                        if all(field in npz for field in LATENTS_META_FIELDS):
                            # Legacy NPZ that still embeds its metadata: harvest it
                            # into latents_meta instead of re-encoding the image.
                            with self.meta_lock:
                                self.latents_meta[key] = {field: [int(v) for v in npz[field]] for field in LATENTS_META_FIELDS}
                            return None
                        # Latent-only NPZ whose metadata row was lost (e.g. crash
                        # before flush): fall through and re-encode.
            except Exception:
                logger.warning("Corrupted file %s, reprocessing...", npz_save_path)
        image_np = SimpleLatentsProcessor.process(item)
        return (item, image_np)

    def process_item(self, worker: object, loaded: tuple[Path, np.ndarray]) -> WritePayload:
        image_path, image_np = loaded
        processor = cast("SimpleLatentsProcessor", worker)
        latents, crop_ltrb, original_size, reso = processor.encode_np(image_np)
        return WritePayload(
            save_path=self.get_npz_save_path(image_path),
            latents=latents,
            crop_ltrb=crop_ltrb,
            original_size=original_size,
            resolution=reso,
        )

    def write_item(self, payload: WritePayload) -> None:
        payload.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(payload.save_path, latents=latents_to_numpy(payload.latents))
        with self.meta_lock:
            self.latents_meta[payload.save_path.stem] = meta_row(payload.resolution, payload.original_size, payload.crop_ltrb)

    def __call__(self) -> None:
        try:
            self.run()
        finally:
            # Flush even on failure so already-encoded items keep their metadata.
            if self.latents_meta:
                write_latents_meta(self.target_path, self.latents_meta)


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
    parser.add_argument("--base_resolution", type=int, default=1024, help="Bucket base resolution: 1024 for SDXL, 768/512 for SD 1.5")

    args = parser.parse_args()

    # Parse dtype
    if args.vae_dtype:
        vae_dtype = str_to_dtype(args.vae_dtype)
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
        base_resolution=args.base_resolution,
    )

    start_time = time.time()
    processor()
    end_time = time.time()

    elapsed = end_time - start_time
    logger.info("Processing completed in %.2f seconds", elapsed)
