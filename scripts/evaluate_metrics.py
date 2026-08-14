"""Distribution similarity (CMMD) and prompt adherence scores for generated images.

Both metrics run on one shared dual-encoder backbone (SigLIP2 by default; any
CLIP-family Hub checkpoint via ``--encoder``):

- CMMD between a reference image set (training data) and a generated set —
  the FID replacement from arXiv:2401.09603, stable from a few hundred images.
- Prompt adherence for every generated image that has a prompt: raw cosine
  (comparable across backbones) plus the backbone's native score (SigLIP match
  probability, or CLIPScore for CLIP backbones).

Prompts are discovered per generated image from a sidecar ``<name>.txt`` next
to it, or from ``--prompts_file`` (JSON object mapping filename to prompt,
which wins over sidecars). Images without a prompt still count toward CMMD.

Usage:
    uv run python scripts/evaluate_metrics.py \
        --ref_dir datasets/sample/images \
        --gen_dir out/eval/singles/my-lora \
        [--encoder google/siglip2-so400m-patch16-384] \
        [--output out/eval/metrics.json] [--no-pandm]

Results print as a table, write to ``--output`` JSON, and log to a pandm run
(project ``diffusion-trainer``, tag ``eval``) unless ``--no-pandm``.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandm
import torch
from rich.console import Console
from rich.table import Table

from diffusion_trainer.eval import DEFAULT_ENCODER, DualEncoder, compute_cmmd, list_images, resolve_prompts

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref_dir", type=Path, required=True, help="Reference (training) image directory.")
    parser.add_argument("--gen_dir", type=Path, required=True, help="Generated image directory.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="CLIP-family Hub checkpoint (CLIP / SigLIP / SigLIP2).")
    parser.add_argument("--prompts_file", type=Path, default=None, help="JSON object {filename: prompt}; overrides sidecar .txt files.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None, help="Cap per set (CMMD is stable from a few hundred).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON here.")
    parser.add_argument("--pandm", default=True, action=argparse.BooleanOptionalAction, help="Log metrics to a pandm run.")
    args = parser.parse_args()

    ref_images = list_images(args.ref_dir, args.max_images)
    gen_images = list_images(args.gen_dir, args.max_images)
    prompts = resolve_prompts(gen_images, args.prompts_file)
    console.print(f"encoder=[bold]{args.encoder}[/] ref={len(ref_images)} gen={len(gen_images)} with_prompt={len(prompts)}")

    encoder = DualEncoder(args.encoder, device=args.device)
    with console.status("encoding reference images..."):
        ref_embs = encoder.encode_images(ref_images, args.batch_size)
    with console.status("encoding generated images..."):
        gen_embs = encoder.encode_images(gen_images, args.batch_size)

    metrics: dict[str, float | int | str] = {
        "encoder": args.encoder,
        "encoder_family": encoder.family,
        "n_ref": len(ref_images),
        "n_gen": len(gen_images),
        "n_with_prompt": len(prompts),
        "cmmd": compute_cmmd(ref_embs, gen_embs),
    }

    if prompts:
        with console.status("encoding prompts..."):
            text_embs = encoder.encode_texts(list(prompts.values()))
        # resolve_prompts iterates gen_images in order, so rows stay aligned.
        paired_image_embs = gen_embs[[i for i, p in enumerate(gen_images) if p in prompts]]
        for name, values in encoder.pair_scores(paired_image_embs, text_embs).items():
            metrics[f"{name}_mean"] = values.mean().item()
            metrics[f"{name}_std"] = values.std().item()

    table = Table(title="evaluation metrics")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        console.print(f"wrote {args.output}")

    if args.pandm:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        run = pandm.init(
            project="diffusion-trainer",
            name=f"eval-{args.gen_dir.name}-{timestamp}",
            config={"ref_dir": str(args.ref_dir), "gen_dir": str(args.gen_dir), "encoder": args.encoder},
            tags=["eval", encoder.family],
            description=f"CMMD + prompt adherence for {args.gen_dir}",
        )
        run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.finish()
        console.print(f"logged to pandm run [bold]{run.id}[/]")


if __name__ == "__main__":
    main()
