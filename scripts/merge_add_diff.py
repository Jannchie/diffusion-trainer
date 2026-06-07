"""Add-difference model merge: base + alpha * (plus - minus).

Transports a finetune delta (e.g. an artist-style anneal) onto a different
base checkpoint without retraining and without changing the base's
parameterization — the output keeps the base pipeline's scheduler config
(epsilon stays epsilon), tokenizer, and VAE; only UNet and text-encoder
weights move.

  uv run python scripts/merge_add_diff.py \
      --base models/AOM3B2_orangemixs_fp16 \
      --plus out/pictoria-v0.5/pictoria-v0.5 \
      --minus out/pictoria-v0.5-base/pictoria-v0.5-base \
      --alphas 0.5 1.0 1.5 \
      --out-dir out/merges --name aom-artist5
"""

import argparse
from pathlib import Path

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from rich.console import Console
from transformers.models.clip import CLIPTextModel

console = Console()


def add_difference(base: torch.nn.Module, plus: torch.nn.Module, minus: torch.nn.Module, alpha: float) -> None:
    """In-place: base += alpha * (plus - minus), accumulated in fp32."""
    base_sd = base.state_dict()
    plus_sd = plus.state_dict()
    minus_sd = minus.state_dict()
    if base_sd.keys() != plus_sd.keys() or base_sd.keys() != minus_sd.keys():
        only = (base_sd.keys() ^ plus_sd.keys()) | (base_sd.keys() ^ minus_sd.keys())
        msg = f"state_dict key mismatch: {sorted(only)[:5]}..."
        raise ValueError(msg)
    merged = {}
    for key, base_w in base_sd.items():
        delta = plus_sd[key].to(torch.float32) - minus_sd[key].to(torch.float32)
        merged[key] = (base_w.to(torch.float32) + alpha * delta).to(base_w.dtype)
    base.load_state_dict(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Pipeline dir whose identity (scheduler/VAE) is kept")
    parser.add_argument("--plus", required=True, help="Pipeline dir at the head of the delta")
    parser.add_argument("--minus", required=True, help="Pipeline dir at the tail of the delta")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--out-dir", default="out/merges")
    parser.add_argument("--name", default="merge")
    args = parser.parse_args()

    dtype = torch.float16
    console.log(f"loading base pipeline: {args.base}")
    base_pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=dtype, safety_checker=None)
    base_unet_sd = {k: v.clone() for k, v in base_pipe.unet.state_dict().items()}
    base_te_sd = {k: v.clone() for k, v in base_pipe.text_encoder.state_dict().items()}

    console.log(f"loading delta endpoints: {args.plus} / {args.minus}")
    plus_unet = UNet2DConditionModel.from_pretrained(args.plus, subfolder="unet", torch_dtype=dtype)
    minus_unet = UNet2DConditionModel.from_pretrained(args.minus, subfolder="unet", torch_dtype=dtype)
    plus_te = CLIPTextModel.from_pretrained(args.plus, subfolder="text_encoder", torch_dtype=dtype)
    minus_te = CLIPTextModel.from_pretrained(args.minus, subfolder="text_encoder", torch_dtype=dtype)

    for alpha in args.alphas:
        # Reset to the pristine base before applying this alpha
        base_pipe.unet.load_state_dict(base_unet_sd)
        base_pipe.text_encoder.load_state_dict(base_te_sd)
        add_difference(base_pipe.unet, plus_unet, minus_unet, alpha)
        add_difference(base_pipe.text_encoder, plus_te, minus_te, alpha)
        out = Path(args.out_dir) / f"{args.name}-a{alpha:g}"
        base_pipe.save_pretrained(out)
        console.log(f"alpha={alpha:g} -> {out}")


if __name__ == "__main__":
    main()
