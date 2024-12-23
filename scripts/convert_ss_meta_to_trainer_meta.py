import argparse
import json
from contextlib import suppress
from pathlib import Path

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

with progress:
    for key in progress.track(ss_meta.keys(), description="Converting metadata"):
        with suppress(FileNotFoundError):
            npz_path = ss_latent_path / f"{key}.npz"

            target_npz_path = trainer_meta_path / "latents" / f"{key}.npz"
            target_npz_path.write_bytes(npz_path.read_bytes())

            if tags := ss_meta[key].get("tags", None):
                target_tags_path = trainer_meta_path / "tags" / f"{key}.txt"
                target_tags_path.write_text(tags)

            if prompt := ss_meta[key].get("caption", None):
                target_prompt_path = trainer_meta_path / "caption" / f"{key}.txt"
                target_prompt_path.write_text(prompt)
