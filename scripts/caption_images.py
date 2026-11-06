"""Recaption a manifest's images with a VLM, following the MOSS-Image recipe.

Reads ``manifest.parquet`` (``key``, ``tags``, optional ``tag_categories`` /
``original_caption``) and the images under ``--images-dir`` named
``<key>.<ext>``, and appends one JSON line per image to ``--output``:

    {"key": ..., "model": ..., "caption": {"long", "medium", "short", "watermark", "borders"}}

Re-runs skip keys already in the output, so an interrupted run resumes and a
manifest that grew (new posts) only captions the new rows. Failed images are
logged and left for the next run.

Example (vLLM serving Qwen3.5-35B-A3B on localhost):
    uv run python scripts/caption_images.py \
        --manifest datasets/pictoria-5star/manifest.parquet \
        --images-dir datasets/pictoria-5star/caption_inputs \
        --output datasets/pictoria-5star/captions.jsonl \
        --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-35B-A3B-FP8 --concurrency 32
"""

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.captioning import CaptionClient, CaptionValidationError, build_reference, read_caption_records

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("caption_images")


def images_by_key(images_dir: Path) -> dict[str, Path]:
    """One directory scan up front instead of a glob per row."""
    return {path.stem: path for path in images_dir.iterdir() if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recaption images with an OpenAI-compatible VLM (MOSS-Image recipe)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of <key>.<ext> images (downscaled copies are fine)")
    parser.add_argument("--output", type=Path, required=True, help="captions.jsonl to append to")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking (faster, less accurate)")
    parser.add_argument("--limit", type=int, default=0, help="Only caption the first N pending rows (0 = all)")
    args = parser.parse_args()

    rows = pq.read_table(args.manifest).to_pylist()
    finished = read_caption_records(args.output).keys()
    images = images_by_key(args.images_dir)
    pending = [row for row in rows if row["key"] not in finished]
    if args.limit:
        pending = pending[: args.limit]
    logger.info("%d rows in manifest, %d already captioned, %d to do", len(rows), len(finished), len(pending))

    client = CaptionClient(endpoint=args.endpoint, model=args.model, max_tokens=args.max_tokens, enable_thinking=not args.no_thinking)
    lock = threading.Lock()
    failed = 0

    def work(row: dict, http: httpx.Client) -> None:
        image = images.get(row["key"])
        if image is None:
            msg = f"no image for {row['key']}"
            raise FileNotFoundError(msg)
        reference = build_reference(row["tags"], row.get("tag_categories"), row.get("original_caption") or "")
        record = client.caption(http, image, reference)
        line = json.dumps({"key": row["key"], "model": args.model, "caption": record.to_json_dict()}, ensure_ascii=False)
        with lock, args.output.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(limits=httpx.Limits(max_connections=args.concurrency)) as http, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, row, http): row["key"] for row in pending}
        for future in track(as_completed(futures), total=len(futures), description="Captioning..."):
            try:
                future.result()
            except (CaptionValidationError, httpx.HTTPError, FileNotFoundError, KeyError) as exc:
                failed += 1
                logger.warning("Failed %s: %s", futures[future], exc)
    logger.info("Done: %d captioned, %d failed (re-run to retry)", len(pending) - failed, failed)


if __name__ == "__main__":
    main()
