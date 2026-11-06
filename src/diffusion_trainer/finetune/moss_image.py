"""Finetuner for MOSS-Image (OpenMOSS): 5B DiT + Qwen3.5-9B + FLUX.2 VAE.

A sibling of ``Lumina2Tuner`` — same ``process_batch`` -> ``train_each_batch``
-> ``train_on_latents`` skeleton, same LyCORIS wiring and preview path. What
differs, all verified against the official ``pipeline_moss_image.py`` and the
tech report (§A.2):

* the objective keeps diffusers' own conventions: time input ``1000 * sigma``,
  target ``noise - x0`` (see ``MOSSFlowMatchObjective``), and the default sigma
  density is the report's truncated logit-normal (``flow_match_timestep_sampling = "moss"``);
* the latent space is the FLUX.2 VAE's 32 channels, 2x2-patchified to 128 and
  normalized by the VAE's batch-norm running statistics — not a scale/shift pair;
* conditioning is Qwen3.5's last hidden state over the chat-templated prompt,
  encoded one prompt at a time (the pipeline does the same, to keep padding out
  of the Gated DeltaNet layers) and right-padded with a boolean mask;
* the DiT and the pipeline are remote code from the model repo, so gradient
  checkpointing is installed here rather than through diffusers.
"""

import functools
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch.utils.checkpoint import checkpoint

from diffusion_trainer.config import BaseConfig, MOSSImageConfig, SampleOptions
from diffusion_trainer.dataset.dataset import DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.objective import DiffusionObjective, MOSSFlowMatchObjective, ddpm_only_settings_in_use
from diffusion_trainer.finetune.utils import TrainableModel, load_moss_image_pipeline
from diffusion_trainer.finetune.utils.lora import MOSS_IMAGE_LORA_TARGETS, LoraTargets

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline
    from PIL import Image

logger = logging.getLogger("diffusion_trainer.finetune.moss_image")


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, 2H, 2W) -> (B, 4C, H, W)``, channel order ``(c, py, px)`` as in ``Flux2Pipeline``."""
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Inverse of ``patchify_latents``; the exact reshape ``MOSSImagePipeline`` decodes with."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def checkpointed(forward: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Recompute ``forward`` in backward instead of storing its activations.

    Only while autograd is recording: previews run the same blocks under
    ``no_grad``, where checkpointing would be pure overhead.
    """

    @functools.wraps(forward)
    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        if torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)  # type: ignore[return-value]
        return forward(*args, **kwargs)

    return wrapper


@torch.no_grad()
def encode_moss_prompts(  # noqa: PLR0913
    tokenizer: Any,  # noqa: ANN401
    text_encoder: torch.nn.Module,
    prompts: list[str],
    *,
    max_sequence_length: int,
    device: torch.device | str,
    empty_prompt_rows: dict[torch.dtype | None, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3.5 last hidden states over the chat-templated prompts, right-padded, plus a boolean mask.

    Mirrors ``MOSSImagePipeline.encode_prompt``: the user-turn chat template
    with ``enable_thinking=True`` and the generation prompt, one unpadded
    forward per prompt (padding must stay out of the Gated DeltaNet layers).
    ``empty_prompt_rows``, when given, memoizes the empty prompt per autocast
    dtype, so condition dropout and empty negatives skip the encoder.
    """
    device_type = torch.device(device).type
    autocast_key = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None
    rows = []
    for prompt in prompts:
        if not prompt and empty_prompt_rows is not None and autocast_key in empty_prompt_rows:
            rows.append(empty_prompt_rows[autocast_key])
            continue
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length).to(device)
        row = text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
        if not prompt and empty_prompt_rows is not None:
            empty_prompt_rows[autocast_key] = row
        rows.append(row)

    length = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
    for index, row in enumerate(rows):
        embeds[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = True
    return embeds, mask


class MOSSImageModels(NamedTuple):
    transformer: torch.nn.Module  # MOSSImageTransformer (remote code)
    text_encoder: torch.nn.Module  # Qwen3_5Model


@dataclass
class MOSSImageBatch:
    img_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor


class MOSSImageTuner(BaseTuner):
    """Finetune MOSS-Image-Base / -SFT."""

    @staticmethod
    def from_config(config: BaseConfig) -> "MOSSImageTuner":
        """Create a new instance from a configuration dictionary."""
        if not isinstance(config, MOSSImageConfig):
            msg = f"Expected MOSSImageConfig, got {type(config)}"
            raise TypeError(msg)
        return MOSSImageTuner(config)

    def __init__(self, config: MOSSImageConfig) -> None:
        self.config = config
        super().__init__(config)
        self.logger = logger
        # The empty prompt (condition dropout, default negative) always encodes
        # to the same states through the frozen encoder: keyed by autocast dtype.
        self._empty_prompt_rows: dict[torch.dtype | None, torch.Tensor] = {}
        self._warn_about_ddpm_only_settings()

    def _warn_about_ddpm_only_settings(self) -> None:
        """Flag config knobs that silently do nothing under rectified flow."""
        ignored = ddpm_only_settings_in_use(self.config) + (["use_enhanced_embeddings"] if self.config.use_enhanced_embeddings else [])
        if ignored:
            logger.warning("These settings do not apply to MOSS-Image and are IGNORED: %s.", ", ".join(ignored))

    # --- model wiring -----------------------------------------------------

    def get_pipeline(self) -> "DiffusionPipeline":
        return load_moss_image_pipeline(self.config.model_path, self.weight_dtype)

    def _setup_models(self) -> None:
        self.moss_models = MOSSImageModels(
            transformer=self.pipeline.transformer.to(self.device, dtype=self.model_dtype(self.config.unet_lr)),
            text_encoder=self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype),
        )
        self.models: list[Any] = list(self.moss_models)

        # The latent normalization is the VAE's batch norm, applied by hand in
        # the pipeline. Its statistics are constants; keep an fp32 device copy
        # so the per-step transform never touches the (CPU-resident) VAE.
        vae = self.pipeline.vae
        bn = vae.bn
        self._latent_mean = bn.running_mean.detach().float().to(self.device).view(1, -1, 1, 1)
        self._latent_std = (bn.running_var.detach().float() + vae.config.batch_norm_eps).sqrt().to(self.device).view(1, -1, 1, 1)

    @property
    def denoiser(self) -> torch.nn.Module:
        return self.pipeline.transformer

    def get_noise_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """``MOSSImageScheduler`` subclasses the flow-match Euler scheduler; reuse it as-is."""
        scheduler = self.pipeline.scheduler
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            msg = f"MOSS-Image expects a FlowMatchEulerDiscreteScheduler subclass, got {type(scheduler).__name__}"
            raise TypeError(msg)
        return scheduler

    def create_objective(self) -> DiffusionObjective:
        return MOSSFlowMatchObjective(self.config, self.get_noise_scheduler(), self.device)

    def _enable_gradient_checkpointing(self) -> None:
        """Checkpoint each joint DiT block.

        The remote-code DiT does not opt into diffusers' checkpointing API
        (``enable_gradient_checkpointing`` raises), so each ``_Block.forward``
        is wrapped instead — the same granularity Kohaku's "block" plan uses.
        The frozen text encoder runs under ``no_grad`` and needs none.
        """
        layers = cast("torch.nn.ModuleList", self.moss_models.transformer.layers)
        for block in layers:
            block.forward = checkpointed(block.forward)
        logger.info("Gradient checkpointing enabled on %d DiT blocks", len(layers))

    def _configure_full_finetune(self) -> None:
        if self.config.unet_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.moss_models.transformer, lr=self.config.unet_lr))

    @property
    def lora_targets(self) -> LoraTargets:
        return MOSS_IMAGE_LORA_TARGETS

    @property
    def lora_base_model_version(self) -> str:
        return "moss_image"

    @property
    def lora_metadata_resolution(self) -> str:
        return "1024,1024"

    # --- latent space -----------------------------------------------------

    def _apply_vae_scaling(self, latents: torch.Tensor) -> torch.Tensor:
        """Raw 32-channel FLUX.2 latents -> the DiT's normalized 128-channel space.

        Exactly what ``Flux2Pipeline`` does on encode: patchify 2x2, then
        subtract the batch-norm mean and divide by its std. Inverse of
        ``_decode_preview_latents``.
        """
        normalized = (patchify_latents(latents.float()) - self._latent_mean) / self._latent_std
        return normalized.to(latents.dtype)

    @torch.no_grad()
    def _decode_preview_latents(self, latents: torch.Tensor) -> "Image.Image":
        with self._vae_in_preview_mode() as vae:
            raw = unpatchify_latents(latents.to(torch.float32) * self._latent_std + self._latent_mean)
            image = vae.decode(raw.to(vae.device), return_dict=False)[0].clamp(-1, 1)
            return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    # --- conditioning -----------------------------------------------------

    def encode_prompts(self, prompts_str: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts to Qwen3.5 hidden states plus a boolean mask.

        Mirrors ``MOSSImagePipeline.encode_prompt``: the user-turn chat template
        with ``enable_thinking=True`` and the generation prompt, the LAST hidden
        state, one unpadded forward per prompt, right-padded afterwards. The
        empty string (condition dropout) goes through the same template, which
        is exactly the pipeline's default negative prompt.
        """
        return encode_moss_prompts(
            self.pipeline.tokenizer,
            self.moss_models.text_encoder,
            prompts_str,
            max_sequence_length=self.config.max_sequence_length,
            device=self.device,
            empty_prompt_rows=self._empty_prompt_rows,
        )

    def process_batch(self, batch: DiffusionBatch) -> MOSSImageBatch:
        prompts_str = self.create_prompts_str(batch)
        prompts_str = self.apply_condition_dropout_to_prompts(prompts_str)
        prompt_embeds, prompt_attention_mask = self.encode_prompts(prompts_str)

        return MOSSImageBatch(
            img_latents=batch.img_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

    # --- training step ----------------------------------------------------

    def train_each_batch(self, batch: MOSSImageBatch) -> None:
        tensors = self._move_tensors_to_device_and_dtype(
            img_latents=batch.img_latents,
            prompt_embeds=batch.prompt_embeds,
        )
        prompt_embeds = tensors["prompt_embeds"]
        # The DiT validates a binary mask and selects tokens with it; keep it bool.
        prompt_attention_mask = batch.prompt_attention_mask.to(self.device)

        def model_pred_fn(img_noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds, prompt_attention_mask)

        self.train_on_latents(
            tensors["img_latents"],
            model_pred_fn,
            extra_tensors=(prompt_embeds, prompt_attention_mask),
        )

    def get_model_pred(
        self,
        img_noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        runtime_transformer = self.get_runtime_model(self.moss_models.transformer)
        return runtime_transformer(
            hidden_states=img_noisy_latents,
            # Already 1000 * sigma (the objective's model_timesteps); the DiT
            # builds its sinusoid from fp32/fp64 times, so keep it fp32.
            timestep=timesteps.float(),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]

    # --- previews ---------------------------------------------------------

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        """Encode preview prompts through the exact training-time path (``clip_skip`` is unused)."""
        del clip_skip
        prompt_embeds, prompt_attention_mask = self.encode_prompts([prompt])
        negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompts([neg_prompt])
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def preview_pipeline_kwargs(self, sample_option: SampleOptions) -> dict[str, Any]:
        """MOSS takes no ``guidance_rescale`` and rejects ``None`` sizes (it has its own 1024 default).

        The pipeline's own guidance defaults stay in force: CFG with block 22
        skipped in the negative branch, sampled at shift 5.
        """
        kwargs = super().preview_pipeline_kwargs(sample_option)
        kwargs.pop("guidance_rescale")
        return {key: value for key, value in kwargs.items() if value is not None}

    @property
    def preview_autocast_dtype(self) -> torch.dtype:
        # fp16 (torch's CUDA autocast default) overflows MOSS's residual stream,
        # which peaks around 62k on the base model; the weights are bf16 anyway.
        return torch.bfloat16

    @property
    def supports_hires_preview(self) -> bool:
        # The remote-code pipeline has no img2img counterpart.
        return False

    # --- saving -----------------------------------------------------------

    def save_full_finetune_model(self, filename: str) -> None:
        """Save the DiT as a drop-in ``transformer/`` folder for the MOSS-Image bundle.

        Only the DiT trains (the text encoder is frozen and the VAE never
        trains), so re-saving the 18 GB encoder every checkpoint would be waste.
        ``pipeline_moss_image.py`` is copied alongside because the folder's
        config names its remote-code class: replace the bundle's
        ``transformer/`` with this one to load the finetune.
        """
        out = self.save_path / filename / "transformer"
        out.mkdir(parents=True, exist_ok=True)
        transformer = self.moss_models.transformer
        original_dtype = next(transformer.parameters()).dtype
        with torch.no_grad():
            if self.save_dtype != original_dtype:
                transformer.to(self.save_dtype)
            try:
                transformer.save_pretrained(out)  # type: ignore[operator]  # a diffusers ModelMixin at runtime
            finally:
                if self.save_dtype != original_dtype:
                    transformer.to(original_dtype)

        source = Path(self.config.model_path) / "transformer" / "pipeline_moss_image.py"
        if source.is_file():
            shutil.copy2(source, out / source.name)
        else:
            logger.warning("Could not find %s to copy next to the saved transformer; copy it from the model repo by hand.", source)
