import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SampleOptions:
    prompt: str
    negative_prompt: str
    steps: int
    seed: int
    width: int | None = field(default=None)
    height: int | None = field(default=None)
    clip_skip: int = field(default=2)
    guidance_scale: float = field(default=7.5)
    # ZTSNR + v-prediction models need CFG rescale (~0.7) to avoid washed-out
    # or overexposed samples (Lin et al., "Common Diffusion Noise Schedules
    # and Sample Steps are Flawed"). Keep 0.0 for epsilon models.
    guidance_rescale: float = field(default=0.0)
    # A1111-style hires fix for previews: lanczos-upscale the base output by
    # hires_scale and img2img it at hires_strength. Off when <= 1. Small faces
    # in full-body previews live below the 8x-VAE latent resolution at base
    # size, so single-pass previews understate the model's detail ceiling.
    hires_scale: float = field(default=0.0)
    hires_strength: float = field(default=0.45)
    hires_steps: int = field(default=20)


@dataclass
class BaseConfig:
    model_path: str = field(metadata={"help": "Path to the model."})
    dataset_path: str = field(metadata={"help": "Path to the dataset."})
    image_path: str | None = field(default=None, metadata={"help": "Path to the images."})
    skip_prepare_image: bool = field(default=True, metadata={"help": "Skip preparing the images."})

    vae_path: str | None = field(default=None, metadata={"help": "Path to the VAE."})
    vae_dtype: str = field(default="fp32", metadata={"help": "the VAE dtype."})

    ss_latent_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset latents."})
    ss_meta_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset metadata."})

    shuffle_tags: bool = field(default=False, metadata={"help": "Shuffle tags."})
    single_tag_dropout: float = field(default=0.0, metadata={"help": "Single tag dropout."})
    all_tags_dropout: float = field(default=0.0, metadata={"help": "All tags dropout."})
    caption_dropout: float = field(default=0.0, metadata={"help": "Caption dropout."})
    # Category-aware prompt assembly. Datasets prepared without a
    # tag_categories column treat every tag as "general", which reproduces the
    # historical flat behavior exactly (full shuffle, full dropout scope).
    tag_category_order: list[str] = field(
        default_factory=lambda: ["quality", "year", "artist", "copyright", "character", "general"],
        metadata={"help": "Prompt order of tag categories; categories not listed (e.g. meta) never enter the prompt."},
    )
    shuffled_tag_categories: list[str] = field(
        default_factory=lambda: ["general"],
        metadata={"help": "Categories whose tags are shuffled when shuffle_tags is on; the rest keep their dataset order."},
    )
    droppable_tag_categories: list[str] = field(
        default_factory=lambda: ["general"],
        metadata={"help": "Categories subject to single_tag_dropout; the rest (quality/artist/...) are never dropped individually."},
    )
    # Declarative train-time subsetting: one exported dataset can serve many
    # runs (e.g. a best-quality-only experiment) by filtering manifest rows;
    # works for both local parquet and hf:// streaming datasets.
    dataset_include_any_tags: list[str] = field(
        default_factory=list,
        metadata={"help": "Keep only samples containing at least one of these tags (empty = keep all). E.g. ['best quality']."},
    )
    dataset_exclude_tags: list[str] = field(
        default_factory=list,
        metadata={"help": "Drop samples containing any of these tags."},
    )
    use_enhanced_embeddings: bool = field(default=False, metadata={"help": "Whether to use enhanced prompt embeddings for training."})
    condition_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "Probability to drop conditional text (CFG-style dropout) to teach unconditional branch."},
    )

    seed: int = field(default_factory=lambda: random.randint(0, 1_000_000_000), metadata={"help": "Seed for reproducibility."})
    model_name: str = field(default="my_model", metadata={"help": "Model name."})
    save_dir: str = field(default="out", metadata={"help": "Directory to save the model."})
    save_dtype: str = field(default="fp32", metadata={"help": "Save dtype."})
    weight_dtype: str = field(default="fp32", metadata={"help": "Weight dtype."})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay for optimizers that support it."})
    mixed_precision: Literal["float16", "bfloat16", "fp16", "bf16"] = field(default="bf16", metadata={"help": "Mixed precision."})
    prediction_type: Literal["epsilon", "v_prediction", "sample"] | None = field(
        default=None,
        metadata={"help": "Prediction type."},
    )
    n_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    dataloader_num_workers: int = field(
        default=2,
        metadata={"help": "Number of DataLoader worker processes. 0 loads latents on the main process (GPU stalls on disk IO). >0 prefetches in parallel."},
    )

    mode: Literal["full-finetune", "lora", "lokr", "loha", "locon"] = field(default="lokr", metadata={"help": "Mode."})

    # Common LoRA parameters (used by all LoRA variants)
    lora_dim: int = field(default=16, metadata={"help": "Dimension for all LoRA variants (lora, loha, locon)."})
    lora_rank: int | None = field(default=None, metadata={"help": "Alias for lora_dim (backward compatibility)."})
    lora_alpha: float = field(default=1.0, metadata={"help": "Alpha for all LoRA variants (lora, loha, locon)."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for LoRA variants that support it (locon)."})
    # Convolutional layer parameters for LoRA variants
    conv_dim: int | None = field(default=None, metadata={"help": "Convolutional layer dimension for LoRA variants. If None, uses lora_dim."})
    conv_alpha: float | None = field(default=None, metadata={"help": "Convolutional layer alpha for LoRA variants. If None, uses lora_alpha."})

    # Specific parameters for certain variants
    lokr_factor: int = field(default=16, metadata={"help": "Factor for LoKr decomposition. Use -1 for adaptive factor."})

    # Advanced LoRA configuration
    lora_multiplier: float = field(default=1.0, metadata={"help": "LoRA multiplier for all modes."})
    lokr_linear_dim: int = field(default=10000, metadata={"help": "LoKr linear dimension (use large value for full dimension)."})
    lokr_feedforward_factor_ratio: float = field(default=0.5, metadata={"help": "Ratio for FeedForward factor relative to Attention factor in LoKr."})

    noise_offset: float = field(
        default=0.0,
        metadata={"help": "Noise offset. 0.02-0.1 typical for epsilon models; keep 0 with rescale_betas_zero_snr (ZTSNR already restores full dynamic range)."},
    )
    noise_offset_probability: float = field(
        default=1.0,
        metadata={"help": "Probability of applying noise offset. 0.25 means 25% of the time. 1.0 means always."},
    )
    input_perturbation: float = field(
        default=0.0,
        metadata={"help": "Input perturbation strength. Off by default — measurably softens fine detail. 0.01-0.1 if enabled."},
    )
    input_perturbation_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for input perturbation linear decay. 0 means no decay (constant perturbation)."},
    )

    # Multi-resolution noise settings
    use_multires_noise: bool = field(
        default=False,
        metadata={"help": "Enable multi-resolution (pyramid) noise. Spatially correlated noise mismatches the white noise fed at inference; opt-in only."},
    )
    multires_noise_iterations: int = field(default=6, metadata={"help": "Number of noise levels/iterations. Higher = more detail, more computation."})
    multires_noise_discount: float = field(default=0.8, metadata={"help": "Discount factor between levels. Lower = more variation. Range: 0.1-0.9."})
    multires_noise_scales: list[float] | None = field(default=None, metadata={"help": "Custom scales [1.0, 0.5, 0.25]. Overrides iterations if set."})
    multires_noise_weights: list[float] | None = field(default=None, metadata={"help": "Custom weights for scales. Must match scales length."})
    use_brownian_noise: bool = field(default=False, metadata={"help": "Use Brownian noise (random walk) instead of Gaussian for low-frequency richness."})
    brownian_noise_scale: float = field(default=1.0, metadata={"help": "Scale factor for Brownian noise amplitude."})

    # Advanced SNR options
    vpred_snr_floor: float = field(
        default=0.01,
        metadata={
            "help": "SNR floor for the snr-detail sampling density: the ZTSNR terminal regime (SNR=0) keeps this "
            "fraction of probability instead of dropping to epsilon's literal zero, keeping the terminal band "
            "trainable. Raise toward 0.05-0.1 if dark-scene capability stalls.",
        },
    )
    use_smooth_min_snr: bool = field(default=True, metadata={"help": "Use smooth Min-SNR weighting instead of hard clipping when SNR gamma is set."})
    smooth_min_snr_mode: Literal["clip", "sigmoid", "tanh"] = field(default="sigmoid", metadata={"help": "Smoothing mode for Min-SNR."})
    smooth_min_snr_factor: float = field(default=0.15, metadata={"help": "Smoothing factor for Min-SNR (higher = less smooth, more stable)."})

    # Flash Attention (xformers) support
    enable_flash_attention: bool = field(
        default=True,
        metadata={"help": "Enable Flash Attention (xformers) for memory efficiency. Reduces VRAM usage by 30-50%."},
    )
    flash_attention_unet: bool = field(default=True, metadata={"help": "Enable Flash Attention for UNet (recommended)."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing."})
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Compile models with torch.compile (inductor). Multi-resolution buckets can hit inductor dynamic-shape bugs on some torch versions."},
    )
    timestep_bias_strategy: Literal["uniform", "logit", "lognormal", "snr-detail"] = field(
        default="uniform",
        metadata={
            "help": "Timestep bias strategy. uniform keeps the high-noise tail trained (needed for ZTSNR); "
            "logit/lognormal focus on mid timesteps; snr-detail samples t with p ∝ max(SNR, vpred_snr_floor)/(SNR+1), "
            "concentrating compute in the detail regime with unit loss weights (no compute wasted on "
            "near-zero-weight samples).",
        },
    )
    timestep_bias_start_step: int = field(
        default=0,
        metadata={
            "help": "Curriculum for biased timestep sampling: sample uniformly until this step (lets the "
            "epsilon->v remap and the ZTSNR terminal regime train at full density), then switch to the "
            "configured bias strategy. 0 applies the bias from the start.",
        },
    )
    timestep_lognormal_mean: float = field(
        default=-1.2,
        metadata={"help": "Mean for lognormal sigma timestep sampling (EDM-style)."},
    )
    timestep_lognormal_std: float = field(
        default=1.2,
        metadata={"help": "Std for lognormal sigma timestep sampling (EDM-style)."},
    )
    timestep_bias_m: float = field(
        default=0.0,
        metadata={"help": "Mean (m) parameter for logit timestep bias. Controls the center of the log-normal distribution."},
    )
    timestep_bias_s: float = field(
        default=1.0,
        metadata={"help": "Scale (s) parameter for logit timestep bias. Controls the spread of the log-normal distribution."},
    )
    # Min-SNR loss rebalancing (https://arxiv.org/abs/2303.09556). Off by default:
    # under v-prediction the weight min(SNR, gamma)/(SNR+1) goes to 0 at the ZTSNR
    # terminal step, starving exactly the timesteps ZTSNR exists to train.
    snr_gamma: float | None = field(
        default=None,
        metadata={"help": "Min-SNR gamma (5.0 typical for epsilon models). 0 or unset disables. Avoid with v_prediction + rescale_betas_zero_snr."},
    )
    # Use debiased estimation technique to weight the loss by SNR, making the model focus more on high SNR (low noise) regions
    use_debiased_estimation: bool = field(
        default=False,
        metadata={"help": "Use debiased estimation technique to reweight loss. Focuses learning on high SNR (low noise) regions."},
    )
    rescale_betas_zero_snr: bool = field(
        default=False,
        metadata={
            "help": "Enable zero terminal SNR. Only sound with v-prediction: epsilon models cannot recover x0 at SNR=0 "
            "and intermittently render black previews (NaN at the terminal step).",
        },
    )

    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    use_ema: bool = field(default=False, metadata={"help": "Use EMA."})
    ema_start_step: int = field(default=0, metadata={"help": "Global step to start EMA updates."})
    use_dual_ema: bool = field(default=False, metadata={"help": "Maintain a secondary short EMA alongside long EMA."})
    ema_decay_long: float = field(default=0.9999, metadata={"help": "Decay for long EMA."})
    ema_decay_short: float = field(default=0.9, metadata={"help": "Decay for short EMA when use_dual_ema is enabled."})
    save_every_n_steps: int = field(default=0, metadata={"help": "Save every n steps."})
    save_every_n_epochs: int = field(default=1, metadata={"help": "Save every n epochs."})
    preview_every_n_steps: int = field(default=0, metadata={"help": "Preview every n steps."})
    preview_every_n_epochs: int = field(default=1, metadata={"help": "Preview every n epochs."})
    preview_before_training: bool = field(default=True, metadata={"help": "Generate preview before training starts."})

    log_with: Literal["pandm", "wandb", "tensorboard", "none"] = field(
        default="pandm",
        metadata={"help": "Logger (pandm is local-first: metrics land in ./.pandm, view with `pandm ui`)."},
    )

    optimizer: Literal["adamW8bit", "adafactor", "prodigy", "lion", "lion8bit"] = field(
        default="adamW8bit",
        metadata={"help": "Optimizer (adamW8bit/adafactor/prodigy/lion/lion8bit)."},
    )
    optimizer_warmup_steps: int = field(default=0, metadata={"help": "Optimizer warmup steps."})
    optimizer_num_cycles: int = field(default=1, metadata={"help": "Optimizer num cycles."})
    optimizer_restart_decay: float = field(
        default=1.0,
        metadata={
            "help": "SGDR peak decay: with num_cycles > 1, each cosine restart peaks at peak_lr * decay^cycle. "
            "1.0 keeps diffusers' full-amplitude restarts; ~0.7 shrinks each reheat so repeated cycles stop "
            "re-entering the detail-grinding LR regime while every valley consolidates at a lower floor.",
        },
    )

    zero_grad_set_to_none: bool = field(default=True, metadata={"help": "Zero grad set to none."})
    preview_sample_options: list[SampleOptions] = field(default_factory=list, metadata={"help": "Preview sample options."})

    checkpoint_every_n_steps: int = field(default=1000, metadata={"help": "Checkpoint steps."})

    def __post_init__(self) -> None:
        # convert preview_sample_options to SampleOptions
        self.preview_sample_options = [item if isinstance(item, SampleOptions) else SampleOptions(**item) for item in self.preview_sample_options]

        # automatically combine save_dir with model_name
        self.save_dir = str(Path(self.save_dir) / self.model_name)

        # handle lora_rank as alias for lora_dim (backward compatibility)
        if self.lora_rank is not None:
            self.lora_dim = self.lora_rank


@dataclass
class SDXLConfig(BaseConfig):
    text_encoder_1_lr: float = field(default=1e-6, metadata={"help": "Text encoder 1 learning rate."})
    text_encoder_2_lr: float = field(default=1e-6, metadata={"help": "Text encoder 2 learning rate."})


@dataclass
class SD15Config(BaseConfig):
    text_encoder_lr: float = field(default=1e-6, metadata={"help": "Text encoder learning rate."})
    clip_skip: int = field(default=2, metadata={"help": "CLIP skip, A1111/WebUI semantics: 1 = last layer, 2 = penultimate layer (NAI convention)."})


@dataclass
class FlowMatchSettings:
    """Rectified-flow training knobs, mixed into the config of every flow-matching family.

    Deliberately NOT on ``BaseConfig``: the DDPM lineage cannot act on any of
    these, and a knob that silently does nothing is worse than one that errors.
    Because dataclass inheritance keeps the TOML key space flat, a config file
    is unaffected by which class a field lives on — but writing
    ``flow_match_shift`` in an SD 1.5 config is now a construction-time
    TypeError instead of a value nothing reads.
    """

    flow_match_timestep_sampling: Literal["logit_normal", "uniform", "mode"] = field(
        default="logit_normal",
        metadata={
            "help": "Sigma sampling density for flow matching. logit_normal (SD3 paper default) concentrates on mid "
            "sigmas where the velocity field is hardest; uniform trains the whole range flat.",
        },
    )
    flow_match_logit_mean: float = field(default=0.0, metadata={"help": "Mean of the logit-normal sigma distribution. Negative biases toward clean latents."})
    flow_match_logit_std: float = field(default=1.0, metadata={"help": "Std of the logit-normal sigma distribution."})
    flow_match_mode_scale: float = field(default=1.29, metadata={"help": "Scale for the 'mode' sampling scheme (SD3 paper). Unused otherwise."})
    flow_match_shift: float | None = field(
        default=None,
        metadata={
            "help": "Resolution shift applied to sampled sigmas: sigma' = shift*sigma / (1 + (shift-1)*sigma). "
            "None follows the model's own sampler (Lumina 2 ships 6.0), which keeps training and inference on the "
            "same sigma band; 1.0 disables the shift.",
        },
    )
    flow_match_loss_weighting: Literal["uniform", "sigma_sqrt", "cosmap"] = field(
        default="uniform",
        metadata={"help": "Per-sample loss weighting for flow matching. uniform pairs with logit_normal sampling (SD3 recipe)."},
    )


@dataclass
class FlowMatchConfig(FlowMatchSettings, BaseConfig):
    """A config a rectified-flow objective can drive.

    ``FlowMatchObjective`` types against this so it reaches both the shared
    knobs (input perturbation, noise construction) and the flow-matching block,
    without knowing which model family it is serving.
    """


# Lumina 2 conditions on Gemma-2 hidden states behind a fixed instruction
# preamble; the pipeline joins them as f"{system_prompt} <Prompt Start> {prompt}".
# This is diffusers' default, pinned here rather than read off the pipeline so a
# diffusers upgrade cannot silently change what a trained model was conditioned on.
LUMINA2_DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)


@dataclass
class Lumina2Config(FlowMatchConfig):
    """Lumina 2: NextDiT + Gemma-2 + 16-channel VAE, trained with rectified flow.

    The DDPM-only knobs on ``BaseConfig`` (``prediction_type``, ``snr_gamma``,
    ``rescale_betas_zero_snr``, ``use_debiased_estimation``,
    ``timestep_bias_*``, ``smooth_min_snr_*``) are inherited but inert here;
    the tuner warns if any are set. Their replacements are the
    ``flow_match_*`` block above.
    """

    text_encoder_lr: float = field(
        default=0.0,
        metadata={"help": "Gemma-2 learning rate. 0 keeps it frozen — the recommended default; a 2B LLM is easy to wreck at image-model LRs."},
    )
    system_prompt: str = field(
        default=LUMINA2_DEFAULT_SYSTEM_PROMPT,
        metadata={"help": "Instruction preamble prepended to every prompt, joined with ' <Prompt Start> '. Must match inference to avoid a train/test gap."},
    )
    max_sequence_length: int = field(
        default=256,
        metadata={"help": "Gemma-2 token budget INCLUDING the system prompt (~35 tokens). diffusers' default is 256."},
    )
    gemma_skip_layers: int = field(
        default=2,
        metadata={
            "help": "Which Gemma-2 hidden state conditions the DiT, counted from the end (CLIP-skip semantics): "
            "2 = penultimate, which is what Lumina 2 was trained with and what the pipeline uses. Change only deliberately.",
        },
    )
