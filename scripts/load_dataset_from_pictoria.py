import io
from pathlib import Path

import httpx
from PIL import Image
from rich.progress import track

from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor
from diffusion_trainer.dataset.utils import load_latents_meta, sharded_path, write_latents_meta


def load_pictoria_dataset(
    base_url: str,
    output_dir: str,
    vae_path: str,
) -> None:
    processor = SimpleLatentsProcessor(vae_path)

    latents_dir = Path(output_dir).joinpath("latents")
    tags_dir = Path(output_dir).joinpath("tags")
    latents_meta = load_latents_meta(latents_dir)
    start = 0
    try:
        while start is not None:
            resp = httpx.get(
                f"{base_url}/v2/posts",
                params={
                    "limit": 1,
                    "start": start,
                },
            )
            data = resp.json()
            start = data["nextCursor"]
            posts = data["items"]
            post_ids = [post["id"] for post in posts]
            for post_id in track(post_ids, description="Downloading posts..."):
                resp = httpx.get(
                    f"{base_url}/v2/posts/{post_id}",
                )
                post = resp.json()
                post_info = {
                    "id": post["id"],
                    "file_path": post["filePath"],
                    "file_name": post["fileName"],
                    "extension": post["extension"],
                    "width": post["width"],
                    "height": post["height"],
                    "aspect_ratio": post["aspectRatio"],
                    "score": post["score"],
                    "source": post["source"],
                    "caption": post["caption"] or "",
                    "tags": [tag["tagInfo"]["name"] for tag in post["tags"]],
                    "sha256": post["sha256"],
                }

                sha256 = post_info["sha256"]
                tag_path = sharded_path(tags_dir, sha256, "txt")
                tag_path.parent.mkdir(parents=True, exist_ok=True)
                tag_path.write_text(", ".join(post_info["tags"]), encoding="utf-8")

                save_path = sharded_path(latents_dir, sha256, "npz")
                if save_path.exists() and sha256 in latents_meta:
                    continue
                img_resp = httpx.get(
                    f"{base_url}/v2/images/original/id/{post_info['id']}",
                )
                if img_resp.status_code != 200:
                    continue
                img = Image.open(io.BytesIO(img_resp.content))
                latents_meta[sha256] = processor.process_by_pil(img, save_npz_path=save_path)
    finally:
        if latents_meta:
            write_latents_meta(latents_dir, latents_meta)

    create_parquet_processor = CreateParquetProcessor(output_dir)
    create_parquet_processor()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load dataset from Pictoria")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL for Pictoria API")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for cached data")
    parser.add_argument(
        "--vae-path",
        type=str,
        required=True,
        help="Path to the VAE model",
    )

    args = parser.parse_args()

    load_pictoria_dataset(
        base_url=args.base_url,
        output_dir=args.output_dir,
        vae_path=args.vae_path,
    )
