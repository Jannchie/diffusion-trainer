"""Large-format gallery: many artists/scenes x many seeds x variants.

Per prompt -> one big sheet (rows = variant, cols = seed, large cells).

    uv run python scripts/compare_lora_gallery.py --epoch ep40 --multiplier 1.0
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
VARIANTS = ["base", "locon", "loha", "lokr"]
SEEDS = [10, 20, 30, 40, 50]
NEG = "worst quality, bad quality, blurry, watermark, signature"
TRIG = "gpt_2_image, generated, best quality, "

# (key, prompt, w, h). Trigger prepended for all but lets artist/scene vary.
PROMPTS = [
    # --- artists x character ---
    ("artist_wlop", TRIG + "wlop, 1girl, solo, upper_body, long_hair, dramatic_lighting, looking_at_viewer, jewelry, intricate_detail", 640, 896),
    ("artist_ciloranko", TRIG + "ciloranko, 1girl, solo, rain, night, city_street, neon_lights, reflection, looking_at_viewer, jacket", 640, 896),
    ("artist_mika_pikazo", TRIG + "mika_pikazo, 1girl, solo, colorful, neon, cyberpunk, looking_at_viewer, upper_body, vivid", 640, 896),
    ("artist_guweiz", TRIG + "guweiz, 1girl, solo, rooftop, sunset, city, wind, school_uniform, from_side, backlighting", 640, 896),
    # --- scenes (mixed character / scenery) ---
    ("scene_night_street", TRIG + "scenery, no_humans, rain, night, city_street, neon_signs, reflection, puddle, lantern, cinematic, bokeh", 896, 640),
    ("scene_watercolor_onsen", "gpt_2_image, generated, watercolor_(medium), traditional_media, mountain, onsen, autumn_leaves, mist, river, no_humans, scenery, sunset", 896, 640),
    ("scene_isekai_market", TRIG + "fantasy, scenery, cliff, floating_island, sky, marketplace, crowd, sunlight, cinematic, no_humans, epic", 896, 640),
    ("scene_snow_shrine", TRIG + "scenery, winter, snow, shrine, torii, night, lantern, no_humans, cinematic, blue_hour, atmospheric", 896, 640),
    ("char_train_night", TRIG + "1girl, solo, sitting, train_interior, window, night, city_lights, reflection, rain, coat, looking_to_the_side", 896, 640),
    ("char_cafe_day", TRIG + "1girl, solo, cafe, window, sunlight, coffee, looking_at_viewer, sweater, cozy, bokeh, warm_lighting", 640, 896),
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
    ap.add_argument("--cell", type=int, default=512)
    args = ap.parse_args()
    dtype = torch.bfloat16
    out_root = SWEEP / "_compare" / f"gallery_{args.epoch}_m{args.multiplier}"
    img_dir = out_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

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
                im = render(pipe, prompt, w, h, s)
                im.save(img_dir / f"{v}__{key}__s{s}.png")  # full-res for close inspection
                results[v][key][s] = im
        del pipe
        torch.cuda.empty_cache()
        print("done", v, flush=True)

    CELL = args.cell
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for key, prompt, w, h in PROMPTS:
        thumbs = {(v, s): results[v][key][s].copy() for v in VARIANTS for s in SEEDS}
        for t in thumbs.values():
            t.thumbnail((CELL, CELL))
        cw = max(t.width for t in thumbs.values())
        ch = max(t.height for t in thumbs.values())
        LW, TH = 80, 30
        sheet = Image.new("RGB", (LW + len(SEEDS) * (cw + 6), TH + len(VARIANTS) * (ch + 6)), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((4, 6), key, fill=(255, 230, 120), font=font)
        for ci, s in enumerate(SEEDS):
            d.text((LW + ci * (cw + 6) + 4, 6), f"seed {s}", fill=(210, 210, 220), font=font)
        for ri, v in enumerate(VARIANTS):
            y = TH + ri * (ch + 6)
            d.text((4, y + ch // 2), v, fill=(120, 220, 255), font=font)
            for ci, s in enumerate(SEEDS):
                sheet.paste(thumbs[(v, s)], (LW + ci * (cw + 6), y))
        sheet.save(out_root / f"_sheet_{key}.png")
        print("sheet", key, flush=True)
    print("OUT", out_root, flush=True)


if __name__ == "__main__":
    main()
