"""Batch VAE encoding script for processing images and generating latent embeddings."""

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
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from torchvision import transforms

# Import from the project
from diffusion_trainer.dataset.utils import retrieve_image_paths
from diffusion_trainer.shared import logger

# Global console for consistent rich output
console = Console()


@dataclass
class ProcessingTask:
    """Task for processing pipeline."""

    image_path: Path
    sha256_hash: str | None = None
    output_path: Path | None = None


@dataclass
class WritePayload:
    """Payload for writing latent vectors."""

    save_path: Path
    latents: torch.Tensor
    crop_ltrb: tuple[int, int, int, int]
    original_size: tuple[int, int]
    resolution: tuple[int, int]


class BatchVAEProcessor:
    """Batch VAE processor for generating latent embeddings from images."""

    def __init__(  # noqa: PLR0913
        self,
        vae_path: str,
        output_dir: str,
        *,
        dtype: torch.dtype | None = None,
        device: str = "cuda",
        num_readers: int = 4,
        num_writers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        """Initialize the batch VAE processor."""
        # Resolve VAE path (download if remote URL)
        self.vae_path = resolve_vae_path(vae_path)
        self.output_dir = Path(output_dir)
        self.device = str(device)
        self.dtype = dtype
        self.num_readers = num_readers
        self.num_writers = num_writers
        self.skip_existing = skip_existing

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load VAE model
        self.vae = self.load_vae_model()

        # Predefined resolutions for bucketing (same as SimpleLatentsProcessor)
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

        # Threading components
        self.read_queue = Queue[ProcessingTask | None](maxsize=num_readers * 2)
        self.process_queue = Queue[tuple[ProcessingTask, np.ndarray] | None](maxsize=num_readers)
        self.write_queue = Queue[WritePayload | None](maxsize=num_writers)

        self.progress_counter = 0
        self.progress_lock = threading.Lock()
        self.progress = None
        self.progress_task_id = None

        # Create multiple GPU processors if available
        self.gpu_count = torch.cuda.device_count() if self.device == "cuda" else 1
        self.processors = []
        for i in range(self.gpu_count):
            proc_device = f"cuda:{i}" if self.device == "cuda" else self.device
            vae_copy = self.load_vae_model(proc_device)
            self.processors.append(vae_copy)

    def load_vae_model(self, device: str | None = None) -> AutoencoderKL:
        """Load the VAE model."""
        if device is None:
            device = self.device
        device_str = str(device)

        path = Path(self.vae_path)

        if path.suffix == ".safetensors":
            logger.info("Loading VAE from file %s", path)
            vae = AutoencoderKL.from_single_file(self.vae_path, torch_dtype=self.dtype)
        else:
            logger.info("Loading VAE from folder %s", path)
            vae = AutoencoderKL.from_pretrained(self.vae_path, torch_dtype=self.dtype)

        vae = vae.to(device_str).eval()  # type: ignore
        logger.info("Loaded VAE (%s) - dtype = %s, device = %s", self.vae_path, vae.dtype, vae.device)
        return vae

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        hash_sha256 = hashlib.sha256()
        with image_path.open("rb") as f:
            # Use larger chunk size for better performance
            while chunk := f.read(65536):  # 64KB chunks
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_output_path(self, sha256_hash: str) -> Path:
        """Get output path with two-level directory structure."""
        dir1 = sha256_hash[:2]
        dir2 = sha256_hash[2:4]
        return self.output_dir / dir1 / dir2 / f"{sha256_hash}.npz"

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
    def load_image(image_path: Path) -> np.ndarray:
        """Load image from path."""
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
    def prepare_image_tensor(self, image_np: np.ndarray, device: str) -> torch.Tensor:
        """Prepare the image tensor."""
        np_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        image_tensor = np_to_tensor(image_np)
        if not isinstance(image_tensor, torch.Tensor):
            msg = "Expected torch.Tensor from transforms"
            raise TypeError(msg)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.dtype).to(device)

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
        """Encode the image using the VAE model."""
        vae_out = vae.encode(image_tensor)
        if not isinstance(vae_out, AutoencoderKLOutput):
            msg = "vae_out is not an instance of AutoencoderKLOutput"
            raise TypeError(msg)
        return vae_out.latent_dist.sample()[0]

    @staticmethod
    def save_encoded_image(payload: WritePayload) -> None:
        """Save the encoded image as a npz file."""
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

    def read_image_worker(self) -> None:
        """Reader thread worker function."""
        while True:
            task = self.read_queue.get()
            if task is None:
                break

            try:
                # Calculate SHA256 and output path lazily
                if task.sha256_hash is None:
                    task.sha256_hash = self.calculate_sha256(task.image_path)
                if task.output_path is None:
                    task.output_path = self.get_output_path(task.sha256_hash)

                # Check if output file already exists
                if self.skip_existing and task.output_path.exists():
                    try:
                        # Verify the existing file is valid
                        npz = np.load(task.output_path)
                        if "train_resolution" in npz:
                            self._update_progress()
                            continue
                    except Exception:
                        # File is corrupted, reprocess
                        logger.warning("Corrupted file %s, reprocessing...", task.output_path)

                # Load image
                image_np = self.load_image(task.image_path)
                self.process_queue.put((task, image_np))

            except Exception as e:
                logger.error("Error reading %s: %s", task.image_path, e)
                self._update_progress()

    def process_image_worker(self, vae: AutoencoderKL) -> None:
        """Processor thread worker function."""
        while True:
            data = self.process_queue.get()
            if data is None:
                break

            task, image_np = data

            try:
                # Get original image size
                original_size = image_np.shape[1], image_np.shape[0]

                # Select resolution and resize image
                reso, resized_size = self.select_reso(*original_size)
                image_np = self.resize_and_trim_image(image_np, reso, resized_size)

                # Calculate crop parameters
                crop_ltrb = self.get_crop_ltrb(np.array(reso), original_size)

                # Prepare tensor and encode
                image_tensor = self.prepare_image_tensor(image_np, str(vae.device))
                latents = self.encode_image(image_tensor, vae)

                # Create payload for writing
                if task.output_path is None:
                    msg = "output_path should be set by reader"
                    raise ValueError(msg)  # noqa: TRY301
                payload = WritePayload(
                    save_path=task.output_path,
                    latents=latents,
                    crop_ltrb=crop_ltrb,
                    original_size=original_size,
                    resolution=reso,
                )

                self.write_queue.put(payload)

            except Exception as e:
                logger.error("Error processing %s: %s", task.image_path, e)
                self._update_progress()

    def write_npz_worker(self) -> None:
        """Writer thread worker function."""
        while True:
            payload = self.write_queue.get()
            if payload is None:
                break

            try:
                self.save_encoded_image(payload)
                self._update_progress()
            except Exception as e:
                logger.error("Error writing %s: %s", payload.save_path, e)
                self._update_progress()

    def setup_threads(self) -> tuple[list[threading.Thread], list[threading.Thread], list[threading.Thread]]:
        """Setup all worker threads."""
        reader_threads = [threading.Thread(target=self.read_image_worker, daemon=True) for _ in range(self.num_readers)]

        processor_threads = [threading.Thread(target=self.process_image_worker, args=(vae,), daemon=True) for vae in self.processors]

        writer_threads = [threading.Thread(target=self.write_npz_worker, daemon=True) for _ in range(self.num_writers)]

        return reader_threads, processor_threads, writer_threads

    def start_threads(self, reader_threads: list[threading.Thread], processor_threads: list[threading.Thread], writer_threads: list[threading.Thread]) -> None:
        """Start all worker threads."""
        for thread in reader_threads + processor_threads + writer_threads:
            thread.start()

    def join_threads(self, reader_threads: list[threading.Thread], processor_threads: list[threading.Thread], writer_threads: list[threading.Thread]) -> None:
        """Join all worker threads."""
        for thread in reader_threads + processor_threads + writer_threads:
            thread.join()

    def _update_progress(self) -> None:
        """Update progress counter and progress bar."""
        with self.progress_lock:
            self.progress_counter += 1
            if self.progress is not None and self.progress_task_id is not None:
                self.progress.update(self.progress_task_id, completed=self.progress_counter)

    def process_directory(self, input_dir: str) -> None:  # noqa: C901
        """Process all images in the input directory."""
        input_path = Path(input_dir)

        # Get all image paths
        console.print("Scanning for images...", style="blue")
        image_paths = list(retrieve_image_paths(input_path, recursive=True))
        total_images = len(image_paths)

        if total_images == 0:
            console.print("No images found in the input directory.", style="red")
            return

        console.print(f"Found {total_images} images to process.", style="green")

        # Setup and start threads
        reader_threads, processor_threads, writer_threads = self.setup_threads()
        self.start_threads(reader_threads, processor_threads, writer_threads)

        # Process with progress tracking
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Processing images...", total=total_images)

            # Set progress references for worker threads
            self.progress = progress
            self.progress_task_id = task_id

            # Add image paths to reader queue in batches
            console.print("Starting image processing...", style="blue")
            batch_size = 1000
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                for image_path in batch:
                    # Create task with just the image path, SHA256 calculated lazily
                    task = ProcessingTask(image_path=image_path)
                    self.read_queue.put(task)

                # Give some time for processing to avoid overwhelming the queue
                if i > 0:
                    time.sleep(0.1)

            # Monitor progress and check for stalls
            last_count = 0
            stall_count = 0
            while self.progress_counter < total_images:
                current_count = self.progress_counter

                # Check for stalled processing
                if current_count == last_count:
                    stall_count += 1
                    if stall_count > 100:  # 10 seconds of no progress
                        console.print(f"⚠️  Processing seems stalled at {current_count}/{total_images}", style="yellow")
                        stall_count = 0
                else:
                    stall_count = 0

                last_count = current_count
                time.sleep(0.1)

            # Signal threads to stop
            for _ in range(self.num_readers):
                self.read_queue.put(None)
            for _ in range(len(self.processors)):
                self.process_queue.put(None)
            for _ in range(self.num_writers):
                self.write_queue.put(None)

            # Wait for all threads to complete
            self.join_threads(reader_threads, processor_threads, writer_threads)

            # Clean up progress references
            self.progress = None
            self.progress_task_id = None

        console.print(f"✅ Successfully processed {total_images} images!", style="bold green")


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
    # Convert HF blob URLs to direct download URLs
    download_url = convert_hf_url_to_direct(url)

    with console.status(f"Downloading {url}..."), httpx.stream("GET", download_url, follow_redirects=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with target_path.open("wb") as f:
            if total_size > 0:
                with Progress() as progress:
                    task = progress.add_task("Downloading...", total=total_size)

                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
            else:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    console.print(f"✅ Downloaded to {target_path}", style="green")


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


def get_default_dtype() -> str:
    """Get the best default dtype based on hardware support."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str.lower() in dtype_map:
        return dtype_map[dtype_str.lower()]
    msg = f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}"
    raise ValueError(msg)


def main() -> None:  # noqa: PLR0915, C901
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch VAE encoding for generating latent embeddings from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated latent embeddings",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to VAE model (local file/directory or remote URL, e.g. https://huggingface.co/...)",
    )

    # Optional arguments
    parser.add_argument(
        "--num_readers",
        type=int,
        default=1,
        help="Number of reader threads for loading images",
    )
    parser.add_argument(
        "--num_writers",
        type=int,
        default=1,
        help="Number of writer threads for saving embeddings",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["fp16", "fp32", "bf16", "float16", "float32", "bfloat16"],
        help="Model precision/data type (auto-detects best option if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda, cpu, etc.)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip existing files (reprocess all images)",
    )

    args = parser.parse_args()

    # Validate arguments
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error("Input directory '%s' does not exist.", args.input_dir)
        return

    if not input_path.is_dir():
        logger.error("Input path '%s' is not a directory.", args.input_dir)
        return

    # Validate VAE path (allow remote URLs)
    if not is_remote_url(args.vae_path):
        vae_path = Path(args.vae_path)
        if not vae_path.exists():
            logger.error("VAE path '%s' does not exist.", args.vae_path)
            return

    # Parse dtype with auto-detection
    dtype_str = args.dtype if args.dtype is not None else get_default_dtype()
    if args.dtype is None:
        logger.info("Auto-detected dtype: %s", dtype_str)
    try:
        dtype = parse_dtype(dtype_str)
    except ValueError as e:
        logger.error("Error: %s", e)
        return

    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # Print configuration using both console and logger
    console.print("\n🚀 [bold blue]Batch VAE Encoding Configuration[/bold blue]")
    console.print(f"  📁 Input Directory: [green]{args.input_dir}[/green]")
    console.print(f"  📂 Output Directory: [green]{args.output_dir}[/green]")
    console.print(f"  🤖 VAE Model: [cyan]{args.vae_path}[/cyan]")
    console.print(f"  🔧 Device: [yellow]{device}[/yellow]")
    console.print(f"  📊 Data Type: [magenta]{dtype_str}[/magenta]")
    console.print(f"  👥 Reader Threads: {args.num_readers}")
    console.print(f"  ✍️  Writer Threads: {args.num_writers}")
    console.print(f"  ⏭️  Skip Existing: {not args.no_skip_existing}")

    if device.startswith("cuda"):
        gpu_count = torch.cuda.device_count()
        console.print(f"  🎮 GPU Count: [bold]{gpu_count}[/bold]")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            console.print(f"    • GPU {i}: [dim]{gpu_name}[/dim]")

    console.print()

    # Create processor and run
    try:
        processor = BatchVAEProcessor(
            vae_path=args.vae_path,
            output_dir=args.output_dir,
            dtype=dtype,
            device=device,
            num_readers=args.num_readers,
            num_writers=args.num_writers,
            skip_existing=not args.no_skip_existing,
        )

        start_time = time.time()
        processor.process_directory(args.input_dir)
        end_time = time.time()

        elapsed = end_time - start_time
        logger.info("🎉 Processing completed in %.2f seconds!", elapsed)

    except Exception as e:
        logger.error("Error during processing: %s", e)
        logger.exception("Processing failed")


if __name__ == "__main__":
    main()
