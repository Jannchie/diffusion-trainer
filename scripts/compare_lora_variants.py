"""Apply each gpt-image-2 LoRA variant to the v0.8-eps base and render a shared
prompt set for side-by-side comparison. Emits one contact sheet per prompt
(rows = base + 4 variants) plus a combined sheet.

    uv run python scripts/compare_lora_variants.py --epoch ep40 --multiplier 1.0
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
VARIANTS = ["lora", "locon", "loha", "lokr"]
NEG = "worst quality, bad quality, blurry, watermark, signature"

# (key, prompt, w, h, seed). Trigger = "gpt_2_image, generated".
PROMPTS = [
    ("char_night", "gpt_2_image, generated, best quality, 1girl, solo, rain, night, city_street, neon_lights, reflection, looking_at_viewer, jacket, bokeh", 640, 896, 111),
    ("char_day_roof", "gpt_2_image, generated, best quality, 1girl, solo, rooftop, blue_sky, cumulonimbus_cloud, summer, sunlight, scenery, city, from_behind, school_uniform, wind", 640, 896, 222),
    ("char_interior", "gpt_2_image, generated, best quality, 1girl, solo, sitting, train_interior, window, night, city_lights, reflection, rain, looking_to_the_side, coat", 896, 640, 333),
    ("scene_watercolor", "gpt_2_image, generated, watercolor_(medium), traditional_media, mountain, onsen, autumn_leaves, mist, river, no_humans, scenery, sunset", 896, 640, 444),
    ("ctrl_clean_1girl", "best quality, 1girl, solo, looking_at_viewer, long_hair, smile, white_background", 768, 768, 47),
    ("ctrl_artist_wlop", "best quality, wlop, 1girl, solo, long_hair, portrait, looking_at_viewer, jewelry, dramatic_lighting", 640, 896, 555),
]


def render_row(pipe, label, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for key, prompt, w, h, seed in PROMPTS:
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        with torch.no_grad(), torch.autocast(pipe.device.type):
            pe, npe = get_embeddings_sd15(pipe.tokenizer, pipe.text_encoder, prompt=prompt, neg_prompt=NEG, clip_skip=2, pad_last_block=True)
            latent = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, num_inference_steps=25, generator=gen, width=w, height=h, guidance_scale=7.5, guidance_rescale=0.0, output_type="latent").images
            # Decode in fp32 -- bf16 VAE decode NaNs to black frames on some
            # latents. Pass the raw [-1,1] sample straight to postprocess, which
            # denormalizes internally -- doing /2+0.5 here too double-denorms and
            # washes everything into mid-gray.
            sample = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
            img = pipe.image_processor.postprocess(sample, output_type="pil")[0]
        img.save(out_dir / f"{label}__{key}.png")
        imgs[key] = img
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="ep40")
    ap.add_argument("--multiplier", type=float, default=1.0)
    args = ap.parse_args()

    dtype = torch.bfloat16
    out_root = SWEEP / "_compare" / f"infer_{args.epoch}_m{args.multiplier}"
    rows = {}

    # Base (no LoRA) reference.
    pipe = load_sd15_pipeline(BASE, dtype)
    pipe.to("cuda")
    pipe.vae.to(torch.float32)  # fp32 VAE for NaN-free decode
    rows["base"] = render_row(pipe, "base", out_root)
    del pipe
    torch.cuda.empty_cache()

    # Each variant: fresh pipeline, merge the saved LyCORIS weights into the UNet.
    for v in VARIANTS:
        weight_file = SWEEP / v / f"gpt2img-{v}" / f"gpt2img-{v}-{args.epoch}.safetensors"
        if not weight_file.exists():
            print("MISSING", weight_file)
            continue
        pipe = load_sd15_pipeline(BASE, dtype)
        pipe.to("cuda")
        pipe.vae.to(torch.float32)  # fp32 VAE for NaN-free decode
        # multiplier=1.0 at load; apply the requested strength once via merge_to.
        net, _ = create_lycoris_from_weights(1.0, weight_file.as_posix(), pipe.unet)
        net.merge_to(args.multiplier)
        net.to("cuda", dtype=dtype)
        rows[v] = render_row(pipe, v, out_root)
        del pipe, net
        torch.cuda.empty_cache()
        print("done", v)

    # Per-prompt contact sheets: rows = base + variants.
    order = ["base", *[v for v in VARIANTS if v in rows]]
    CELL = 420
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for key, prompt, w, h, seed in PROMPTS:
        thumbs = []
        for label in order:
            im = rows[label][key].copy()
            im.thumbnail((CELL, CELL))
            thumbs.append((label, im))
        cw = max(t.width for _, t in thumbs)
        ch = max(t.height for _, t in thumbs)
        LW = 90
        sheet = Image.new("RGB", (LW + len(thumbs) * (cw + 8), 34 + ch), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), key, fill=(255, 230, 120), font=font)
        for i, (label, im) in enumerate(thumbs):
            x = LW + i * (cw + 8)
            d.text((x + 4, 8), label, fill=(210, 210, 220), font=font)
            sheet.paste(im, (x, 34))
        sheet.save(out_root / f"_sheet_{key}.png")
        print("sheet", key)
    print("OUT", out_root)


if __name__ == "__main__":
    main()
