"""Backbone-agnostic dual-encoder wrapper for evaluation metrics.

Wraps any CLIP-family checkpoint on the Hub (CLIP, SigLIP, SigLIP2) behind one
interface: l2-normalized image/text embeddings plus the family's *native*
image-text score. Native matters because the families are calibrated
differently — CLIP was trained with a softmax contrastive loss, so its
convention is CLIPScore = 100 * max(cos, 0); SigLIP/SigLIP2 were trained with
a sigmoid pairwise loss, so sigmoid(cos * logit_scale + logit_bias) is an
actual match probability. Raw cosine is always returned too, as the
cross-backbone comparable number.
"""

import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from diffusion_trainer.utils.dtype import get_default_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DEFAULT_ENCODER = "google/siglip2-so400m-patch16-384"


def _load_image(image: "Path | str | Image.Image") -> Image.Image:
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")


class DualEncoder:
    """A CLIP-family model + processor pair exposing normalized embeddings and pair scores."""

    def __init__(self, model_id: str = DEFAULT_ENCODER, device: str | torch.device = "cuda", dtype: torch.dtype | None = None) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        if dtype is None:
            dtype = get_default_dtype() if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.family: str = self.model.config.model_type
        # Capability probe instead of family-name matching: the sigmoid-loss
        # family is exactly the set of models carrying a logit_bias, and that
        # same property decides tokenizer padding (their text towers were
        # trained on fixed max_length padding; dynamic padding silently
        # degrades their embeddings).
        self._logit_bias = getattr(self.model, "logit_bias", None)
        self.uses_sigmoid_loss: bool = self._logit_bias is not None

    def _embedding(self, features: "torch.Tensor | object") -> torch.Tensor:
        """Unwrap get_*_features output and guard the embedding space.

        transformers 5.x returns a ModelOutput whose pooler_output is the
        projected embedding the contrastive head scores with (verified against
        logits_per_image for both CLIP and SigLIP2); 4.x returned the tensor
        directly. When the config declares a projection_dim, verify it — a
        backbone whose pooler_output is the pre-projection state would
        otherwise yield plausible-looking but meaningless metrics.
        """
        emb = features if isinstance(features, torch.Tensor) else features.pooler_output  # type: ignore[union-attr]
        expected = getattr(self.model.config, "projection_dim", None)
        if expected is not None and emb.shape[-1] != expected:
            msg = f"{self.model_id}: embedding dim {emb.shape[-1]} != projection_dim {expected}; pooler_output is not the projected embedding"
            raise RuntimeError(msg)
        return emb

    @staticmethod
    def _finalize(emb: torch.Tensor) -> torch.Tensor:
        """The output contract in one place: l2-normalized float32 on CPU."""
        return torch.nn.functional.normalize(emb.float(), dim=-1).cpu()

    @torch.inference_mode()
    def encode_images(self, images: "Sequence[Path | str | Image.Image]", batch_size: int = 32, num_workers: int = 8) -> torch.Tensor:
        """Return (N, D) l2-normalized float32 embeddings on CPU.

        Decoding is the wall-clock bottleneck, not the forward pass, so images
        are decoded on a thread pool with two batches in flight — bounded
        memory, and the GPU never waits on PIL.
        """
        chunks = iter([images[i : i + batch_size] for i in range(0, len(images), batch_size)])
        feats = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            pending = deque([[pool.submit(_load_image, img) for img in chunk] for chunk in itertools.islice(chunks, 2)])
            while pending:
                batch = [future.result() for future in pending.popleft()]
                if (nxt := next(chunks, None)) is not None:
                    pending.append([pool.submit(_load_image, img) for img in nxt])
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats.append(self._finalize(self._embedding(self.model.get_image_features(**inputs))))
        return torch.cat(feats)

    @torch.inference_mode()
    def encode_texts(self, texts: "Sequence[str]", batch_size: int = 64) -> torch.Tensor:
        """Return (N, D) l2-normalized float32 embeddings on CPU.

        Note CLIP truncates at its 77-token window (tag prompts longer than
        that are cut — one reason to prefer the SigLIP2 backbone here).
        """
        tokenize_kwargs = {"padding": "max_length" if self.uses_sigmoid_loss else True, "truncation": True}
        feats = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(text=list(texts[start : start + batch_size]), return_tensors="pt", **tokenize_kwargs).to(self.device)
            feats.append(self._finalize(self._embedding(self.model.get_text_features(**inputs))))
        return torch.cat(feats)

    def pair_scores(self, image_embs: torch.Tensor, text_embs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-pair alignment scores for row-aligned (N, D) embedding matrices.

        Returns ``cosine`` always, plus ``siglip_prob`` (match probability) for
        sigmoid-loss backbones or ``clip_score`` for softmax-loss ones.
        """
        cos = (image_embs * text_embs).sum(dim=-1)
        scores = {"cosine": cos}
        if self._logit_bias is not None:
            scores["siglip_prob"] = torch.sigmoid(cos * self.model.logit_scale.exp().float().cpu() + self._logit_bias.float().cpu())
        else:
            scores["clip_score"] = 100.0 * cos.clamp(min=0)
        return scores
