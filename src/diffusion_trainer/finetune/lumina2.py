"""Finetuner for Lumina 2 (NextDiT) models, e.g. Neta Lumina.

Structurally a sibling of ``SD15Tuner``/``SDXLTuner``: same ``process_batch`` ->
``train_each_batch`` -> ``train_on_latents`` skeleton, same LyCORIS wiring, same
preview path. What differs is pushed into the pieces ``BaseTuner`` now delegates:

* the denoiser is ``pipeline.transformer`` (a DiT), not ``pipeline.unet``;
* the objective is rectified flow, not DDPM (see ``FlowMatchObjective`` for the
  sigma/timestep/target conventions, which are easy to get subtly backwards);
* conditioning is Gemma-2 hidden states with an attention mask, and the mask has
  to travel all the way to the DiT — dropping it makes the model attend to 200+
  pad tokens and quietly wrecks prompt adherence;
* the VAE is the 16-channel FLUX one, so latents carry a ``shift_factor``
  (handled in ``BaseTuner._apply_vae_scaling``).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from diffusion_trainer.config import BaseConfig, Lumina2Config, SampleOptions
from diffusion_trainer.dataset.dataset import DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.objective import DiffusionObjective, FlowMatchObjective
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    load_lumina2_pipeline,
)
from diffusion_trainer.finetune.utils.lora import LUMINA2_LORA_TARGETS, LoraTargets

if TYPE_CHECKING:
    from diffusers.pipelines.lumina2.pipeline_lumina2 import Lumina2Pipeline

logger = logging.getLogger("diffusion_trainer.finetune.lumina2")

# Lumina 2 joins the instruction preamble to the user prompt with this literal.
PROMPT_START_MARKER = " <Prompt Start> "


class Lumina2Models(NamedTuple):
    transformer: Lumina2Transformer2DModel
    text_encoder: torch.nn.Module  # Gemma2Model


@dataclass
class Lumina2Batch:
    img_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor


class Lumina2Tuner(BaseTuner):
    """Finetune Lumina 2 / Neta Lumina models."""

    @staticmethod
    def from_config(config: BaseConfig) -> "Lumina2Tuner":
        """Create a new instance from a configuration dictionary."""
        if not isinstance(config, Lumina2Config):
            msg = f"Expected Lumina2Config, got {type(config)}"
            raise TypeError(msg)
        return Lumina2Tuner(config)

    def __init__(self, config: Lumina2Config) -> None:
        self.config = config
        super().__init__(config)
        self.logger = logger
        self._warn_about_ddpm_only_settings()

    def _warn_about_ddpm_only_settings(self) -> None:
        """Flag config knobs that silently do nothing under rectified flow.

        These are all load-bearing in the SD 1.5 lineage, so a config copied
        from there looks like it configures something when it does not.
        """
        ignored = [
            name
            for name, active in (
                ("prediction_type", self.config.prediction_type is not None),
                ("rescale_betas_zero_snr", self.config.rescale_betas_zero_snr),
                ("snr_gamma", bool(self.config.snr_gamma)),
                ("use_debiased_estimation", self.config.use_debiased_estimation),
                ("timestep_bias_strategy", self.config.timestep_bias_strategy != "uniform"),
            )
            if active
        ]
        if ignored:
            logger.warning(
                "These settings belong to the DDPM lineage and are IGNORED under rectified flow: %s. Use the flow_match_* options instead.",
                ", ".join(ignored),
            )
        if self.config.use_enhanced_embeddings:
            logger.warning("use_enhanced_embeddings is ignored for Lumina 2: Gemma-2 conditioning has no A1111 attention-syntax parser.")

    # --- model wiring -----------------------------------------------------

    def get_pipeline(self) -> "Lumina2Pipeline":
        return load_lumina2_pipeline(self.config.model_path, self.weight_dtype)

    def _setup_models(self) -> None:
        self.lumina_models = Lumina2Models(
            transformer=self.pipeline.transformer.to(self.device, dtype=self.model_dtype(self.config.unet_lr)),
            text_encoder=self.pipeline.text_encoder.to(self.device, dtype=self.model_dtype(self.config.text_encoder_lr)),
        )
        self.models: list[Any] = list(self.lumina_models)

    @property
    def denoiser(self) -> torch.nn.Module:
        return self.pipeline.transformer

    def get_noise_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Reuse the pipeline's flow-match scheduler as-is.

        Unlike the DDPM path there is nothing to rewrite: prediction_type and
        zero-terminal-SNR are DDPM concepts, and the sigma schedule only matters
        at sampling time (training draws its own sigmas in the objective).
        """
        scheduler = self.pipeline.scheduler
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            msg = f"Lumina 2 expects a FlowMatchEulerDiscreteScheduler, got {type(scheduler).__name__}"
            raise TypeError(msg)
        return scheduler

    def create_objective(self) -> DiffusionObjective:
        return FlowMatchObjective(self.config, self.get_noise_scheduler(), self.device)

    def _configure_full_finetune(self) -> None:
        if self.config.unet_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.lumina_models.transformer, lr=self.config.unet_lr))
        if self.config.text_encoder_lr:
            logger.warning("Training Gemma-2 at lr=%s. It is a 2B LLM: this is easy to destabilize and rarely what you want.", self.config.text_encoder_lr)
            self.trainable_models_with_lr.append(TrainableModel(model=self.lumina_models.text_encoder, lr=self.config.text_encoder_lr))

    @property
    def lora_targets(self) -> LoraTargets:
        return LUMINA2_LORA_TARGETS

    @property
    def lora_base_model_version(self) -> str:
        return "lumina2"

    @property
    def lora_metadata_resolution(self) -> str:
        return "1024,1024"

    # --- conditioning -----------------------------------------------------

    def _build_prompts(self, prompts_str: list[str]) -> list[str]:
        """Prepend the instruction preamble exactly the way the pipeline does."""
        return [self.config.system_prompt + PROMPT_START_MARKER + prompt for prompt in prompts_str]

    def encode_prompts(self, prompts_str: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts to Gemma-2 hidden states plus their attention mask.

        Mirrors ``Lumina2Pipeline._get_gemma_prompt_embeds``: right-padded to
        ``max_sequence_length``, conditioned on ``hidden_states[-gemma_skip_layers]``
        (the penultimate layer by default, as Lumina 2 was trained), and the
        real attention mask is returned rather than discarded — Gemma is a
        causal LLM whose pad positions carry garbage, and the DiT cross-attends
        to this sequence directly.
        """
        tokenizer = self.pipeline.tokenizer
        text_inputs = tokenizer(
            self._build_prompts(prompts_str),
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.accelerator.device)
        attention_mask = text_inputs.attention_mask.to(self.accelerator.device)

        runtime_text_encoder = self.get_runtime_model(self.lumina_models.text_encoder)
        with self.text_encoder_grad_context(self.lumina_models.text_encoder):
            output = runtime_text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # hidden_states[0] is the embedding output; clamp so a misconfigured
        # skip depth cannot wrap around into it.
        skip = max(self.config.gemma_skip_layers, 1)
        index = max(-skip, -(len(output.hidden_states) - 1))
        return output.hidden_states[index], attention_mask

    def process_batch(self, batch: DiffusionBatch) -> Lumina2Batch:
        prompts_str = self.create_prompts_str(batch)
        prompts_str = self.apply_condition_dropout_to_prompts(prompts_str)
        prompt_embeds, prompt_attention_mask = self.encode_prompts(prompts_str)

        return Lumina2Batch(
            img_latents=batch.img_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

    # --- training step ----------------------------------------------------

    def train_each_batch(self, batch: Lumina2Batch) -> None:
        tensors = self._move_tensors_to_device_and_dtype(
            img_latents=batch.img_latents,
            prompt_embeds=batch.prompt_embeds,
        )
        prompt_embeds = tensors["prompt_embeds"]
        # The mask stays integer: casting it to bf16 with the other tensors
        # would feed the attention a float mask it does not expect.
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
        runtime_transformer = self.get_runtime_model(self.lumina_models.transformer)
        return runtime_transformer(
            hidden_states=img_noisy_latents,
            # Already mapped to Lumina's t=1-is-image convention by the objective.
            timestep=timesteps.to(dtype=img_noisy_latents.dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]

    # --- previews ---------------------------------------------------------

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        """Encode preview prompts through the exact training-time path.

        ``clip_skip`` is a CLIP concept and is ignored; the Gemma layer choice
        comes from ``gemma_skip_layers`` so previews cannot silently diverge
        from training. Returns the four tensors ``Lumina2Pipeline`` expects.
        """
        del clip_skip
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask = self.encode_prompts([prompt])
            negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompts([neg_prompt])
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def preview_pipeline_kwargs(self, sample_option: SampleOptions) -> dict[str, Any]:
        """Lumina 2 has no ``guidance_rescale``; it normalizes CFG differently.

        ``cfg_normalization`` (rescale the guided prediction back to the
        conditional's norm) and ``cfg_trunc_ratio`` (drop the unconditional
        branch over the last fraction of the schedule) are the pipeline's own
        defaults — the SD-side ``guidance_rescale`` field is simply unused.
        Subtracting from the base dict rather than restating it means a new
        sampling knob reaches this family automatically.
        """
        kwargs = super().preview_pipeline_kwargs(sample_option)
        kwargs.pop("guidance_rescale")
        return kwargs

    @property
    def supports_hires_preview(self) -> bool:
        # diffusers ships no Lumina 2 img2img pipeline, so AutoPipelineForImage2Image
        # cannot build one from this pipeline.
        return False
