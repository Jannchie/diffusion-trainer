"""Multi-seed variant comparison: for each prompt, a grid of rows=variant x
cols=seed, so per-variant seed-consistency is visible (not a cherry-picked seed).

    uv run python scripts/compare_lora_seeds.py --epoch ep40 --multiplier 1.0
"""
import argparse
from pathlib import Path

import torch
from diffusion_prompt_embedder import get_embeddings_sd15
from lycoris import create_lycoris_from_weights
from PIL import Image, ImageDraw, ImageFont

from diffusion_trainer.finetune.utils import load_sd15_pipeline

BASE = "out/pictoria-v0.8-eps/pictoria-v0.8-eps"
SWEEP = Path("out/gpt2img-sweep")
VARIANTS = ["base", "locon", "loha", "lokr"]  # the user's shortlist + locon
SEEDS = [10, 20, 30, 40]
NEG = "worst quality, bad quality, blurry, watermark, signature"

# (key, prompt, w, h)
PROMPTS = [
    ("char_night", "gpt_2_image, generated, best quality, 1girl, solo, rain, night, city_street, neon_lights, reflection, looking_at_viewer, jacket, bokeh", 640, 896),
    ("char_day_roof", "gpt_2_image, generated, best quality, 1girl, solo, rooftop, blue_sky, cumulonimbus_cloud, summer, sunlight, scenery, city, from_behind, school_uniform, wind", 640, 896),
    ("scene_night_street", "gpt_2_image, generated, scenery, no_humans, rain, night, city_street, neon_signs, reflection, puddle, lantern, cinematic, bokeh", 896, 640),
]


def render(pipe, prompt, w, h, seed):
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad(), torch.autocast(pipe.device.type):
        pe, npe = get_embeddings_sd15(pipe.tokenizer, pipe.text_encoder, prompt=prompt, neg_prompt=NEG, clip_skip=2, pad_last_block=True)
        latent = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, num_inference_steps=25, generator=gen, width=w, height=h, guidance_scale=7.5, guidance_rescale=0.0, output_type="latent").images
        sample = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
        return pipe.image_processor.postprocess(sample, output_type="pil")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="ep40")
    ap.add_argument("--multiplier", type=float, default=1.0)
    args = ap.parse_args()
    dtype = torch.bfloat16
    out_root = SWEEP / "_compare" / f"seeds_{args.epoch}_m{args.multiplier}"
    out_root.mkdir(parents=True, exist_ok=True)

    # results[variant][key][seed] = PIL
    results = {v: {k: {} for k, *_ in PROMPTS} for v in VARIANTS}
    for v in VARIANTS:
        pipe = load_sd15_pipeline(BASE, dtype)
        pipe.to("cuda")
        pipe.vae.to(torch.float32)
        if v != "base":
            wf = SWEEP / v / f"gpt2img-{v}" / f"gpt2img-{v}-{args.epoch}.safetensors"
            net, _ = create_lycoris_from_weights(1.0, wf.as_posix(), pipe.unet)
            net.merge_to(args.multiplier)
            net.to("cuda", dtype=dtype)
        for key, prompt, w, h in PROMPTS:
            for s in SEEDS:
                results[v][key][s] = render(pipe, prompt, w, h, s)
        del pipe
        torch.cuda.empty_cache()
        print("done", v)

    # One sheet per prompt: rows = variants, cols = seeds.
    CELL = 320
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for key, prompt, w, h in PROMPTS:
        thumbs = {(v, s): results[v][key][s].copy() for v in VARIANTS for s in SEEDS}
        for t in thumbs.values():
            t.thumbnail((CELL, CELL))
        cw = max(t.width for t in thumbs.values())
        ch = max(t.height for t in thumbs.values())
        LW, TH = 70, 26
        W = LW + len(SEEDS) * (cw + 6)
        H = TH + len(VARIANTS) * (ch + 6)
        sheet = Image.new("RGB", (W, H), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((4, 6), key, fill=(255, 230, 120), font=font)
        for ci, s in enumerate(SEEDS):
            d.text((LW + ci * (cw + 6) + 4, 6), f"seed {s}", fill=(210, 210, 220), font=font)
        for ri, v in enumerate(VARIANTS):
            y = TH + ri * (ch + 6)
            d.text((4, y + ch // 2), v, fill=(120, 220, 255), font=font)
            for ci, s in enumerate(SEEDS):
                sheet.paste(thumbs[(v, s)], (LW + ci * (cw + 6), y))
        sheet.save(out_root / f"_seeds_{key}.png")
        print("sheet", key)
    print("OUT", out_root)


if __name__ == "__main__":
    main()
