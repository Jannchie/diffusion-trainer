import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_trainer.shared import get_progress

progress = get_progress()

parser = argparse.ArgumentParser()
parser.add_argument("--ss_meta_path", type=str, required=True)
parser.add_argument("--ss_latent_path", type=str, required=True)
parser.add_argument("--trainer_meta_path", type=str, required=True)

args = parser.parse_args()

ss_meta_path = Path(args.ss_meta_path)
ss_latent_path = Path(args.ss_latent_path)
trainer_meta_path = Path(args.trainer_meta_path)

ss_meta = json.loads(ss_meta_path.read_text())

trainer_meta_path.mkdir(exist_ok=True)
(trainer_meta_path / "latents").mkdir(exist_ok=True)
(trainer_meta_path / "tags").mkdir(exist_ok=True)
(trainer_meta_path / "captions").mkdir(exist_ok=True)


def process_key(key: str) -> dict:
    item = {}
    with suppress(FileNotFoundError):
        npz_path = ss_latent_path / f"{key}.npz"
        npz = np.load(npz_path)

        target_npz_path = trainer_meta_path / "latents" / f"{key}.npz"
        target_npz_path.write_bytes(npz_path.read_bytes())
        item["key"] = key
        item["tags"] = ss_meta[key].get("tags", []).split(", ")
        item["caption"] = ss_meta[key].get("caption", "")
        item["train_resolution"] = npz.get("train_resolution").tolist()
        item["original_size"] = npz.get("original_size").tolist()
        item["crop_ltrb"] = npz.get("crop_ltrb").tolist()
    return item


with progress:
    with ThreadPoolExecutor() as executor:
        results = list(progress.track(executor.map(process_key, ss_meta.keys()), description="Converting metadata", total=len(ss_meta)))

    items = defaultdict(list)
    for result in results:
        if result:
            for k, v in result.items():
                items[k].append(v)

    table = pa.table(items)
    pq.write_table(table, trainer_meta_path / "metadata.parquet")
