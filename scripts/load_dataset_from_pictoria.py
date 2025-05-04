import io
from pathlib import Path

import httpx
from PIL import Image
from rich.progress import track

from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor


def load_pictoria_dataset(
    base_url: str,
    output_dir: str,
    vae_path: str,
) -> None:
    processor = SimpleLatentsProcessor(vae_path)

    Path(output_dir).joinpath("tags").mkdir(parents=True, exist_ok=True)
    latents_dir = Path(output_dir).joinpath("latents")
    tags_dir = Path(output_dir).joinpath("tags")
    resp = httpx.post(
        f"{base_url}/v2/posts/search",
        params={
            "limit": 999999999,
        },
        json={
            "score": [
                4,
                5,
            ],
        },
    )
    posts = resp.json()
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
            "md5": post["md5"],
        }

        with tags_dir.joinpath(f"{post_info['md5']}.txt").open("w", encoding="utf-8") as f:
            f.write(", ".join(post_info["tags"]))

        img_resp = httpx.get(
            f"{base_url}/v2/images/original/id/{post_info['id']}",
        )
        img = Image.open(io.BytesIO(img_resp.content))
        save_path = latents_dir / f"{post_info['md5']}.npz"
        if save_path.exists():
            continue
        processor.process_by_pil(img, save_npz_path=save_path)

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
