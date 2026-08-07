"""Strength comparison: variant x merge-multiplier x seed, large cells.

Per prompt -> one sheet: rows = (variant @ strength) combos, cols = seed.
Includes a NO-TRIGGER dark prompt to probe how leakage scales with strength.

    uv run python scripts/compare_lora_strength.py --epoch ep40
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
VARIANTS = ["locon", "loha", "lokr"]
STRENGTHS = [0.5, 0.75, 1.0]
SEEDS = [10, 20, 30, 40]
NEG = "worst quality, bad quality, blurry, watermark, signature"
TRIG = "gpt_2_image, generated, best quality, "

PROMPTS = [
    ("ciloranko_dark", TRIG + "ciloranko, 1girl, solo, rain, night, city_street, neon_lights, reflection, looking_at_viewer, jacket", 640, 896),
    ("scene_night_street", TRIG + "scenery, no_humans, rain, night, city_street, neon_signs, reflection, puddle, lantern, cinematic, bokeh", 896, 640),
    ("watercolor_onsen", "gpt_2_image, generated, watercolor_(medium), traditional_media, mountain, onsen, autumn_leaves, mist, river, no_humans, scenery, sunset", 896, 640),
    ("cafe_day", TRIG + "1girl, solo, cafe, window, sunlight, coffee, looking_at_viewer, sweater, cozy, bokeh, warm_lighting", 640, 896),
    # NO trigger, dark scene -> leakage probe (should stay clean v0.8 at all strengths).
    ("LEAK_dark_notrigger", "best quality, 1girl, solo, night, dark, moonlight, looking_at_viewer, black_dress, indoors, long_hair", 640, 896),
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
    ap.add_argument("--cell", type=int, default=340)
    args = ap.parse_args()
    dtype = torch.bfloat16
    out_root = SWEEP / "_compare" / f"strength_{args.epoch}"
    img_dir = out_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # results[(variant,strength)][key][seed]
    results = {}
    for v in VARIANTS:
        wf = SWEEP / v / f"gpt2img-{v}" / f"gpt2img-{v}-{args.epoch}.safetensors"
        for st in STRENGTHS:
            pipe = load_sd15_pipeline(BASE, dtype)
            pipe.to("cuda")
            pipe.vae.to(torch.float32)
            net, _ = create_lycoris_from_weights(1.0, wf.as_posix(), pipe.unet)
            net.merge_to(st)
            net.to("cuda", dtype=dtype)
            cell = {}
            for key, prompt, w, h in PROMPTS:
                cell[key] = {}
                for s in SEEDS:
                    im = render(pipe, prompt, w, h, s)
                    im.save(img_dir / f"{v}_m{st}__{key}__s{s}.png")
                    cell[key][s] = im
            results[(v, st)] = cell
            del pipe, net
            torch.cuda.empty_cache()
            print("done", v, st, flush=True)

    combos = [(v, st) for v in VARIANTS for st in STRENGTHS]
    CELL = args.cell
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for key, prompt, w, h in PROMPTS:
        thumbs = {(c, s): results[c][key][s].copy() for c in combos for s in SEEDS}
        for t in thumbs.values():
            t.thumbnail((CELL, CELL))
        cw = max(t.width for t in thumbs.values())
        ch = max(t.height for t in thumbs.values())
        LW, TH = 110, 28
        sheet = Image.new("RGB", (LW + len(SEEDS) * (cw + 6), TH + len(combos) * (ch + 6)), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((4, 6), key, fill=(255, 230, 120), font=font)
        for ci, s in enumerate(SEEDS):
            d.text((LW + ci * (cw + 6) + 4, 6), f"seed {s}", fill=(210, 210, 220), font=font)
        for ri, c in enumerate(combos):
            y = TH + ri * (ch + 6)
            d.text((4, y + ch // 2 - 6), f"{c[0]}\n@{c[1]}", fill=(120, 220, 255), font=font)
            for ci, s in enumerate(SEEDS):
                sheet.paste(thumbs[(c, s)], (LW + ci * (cw + 6), y))
        sheet.save(out_root / f"_sheet_{key}.png")
        print("sheet", key, flush=True)
    print("OUT", out_root, flush=True)


if __name__ == "__main__":
    main()
