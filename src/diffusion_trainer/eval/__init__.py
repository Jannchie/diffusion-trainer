"""Evaluation metrics for trained checkpoints.

- :mod:`.encoders` wraps CLIP-family dual encoders (CLIP / SigLIP / SigLIP2)
  behind one interface so every metric here is backbone-agnostic.
- :mod:`.cmmd` computes CMMD (CLIP Maximum Mean Discrepancy), the FID
  replacement from "Rethinking FID" (arXiv:2401.09603) — unbiased and stable
  from a few hundred samples, unlike FID.

Entry point: ``scripts/evaluate_metrics.py``.
"""

from diffusion_trainer.eval.cmmd import compute_cmmd
from diffusion_trainer.eval.data import list_images, resolve_prompts
from diffusion_trainer.eval.encoders import DEFAULT_ENCODER, DualEncoder

__all__ = ["DEFAULT_ENCODER", "DualEncoder", "compute_cmmd", "list_images", "resolve_prompts"]
