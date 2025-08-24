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
    use_enhanced_embeddings: bool = field(default=False, metadata={"help": "Whether to use enhanced prompt embeddings for training."})

    seed: int = field(default_factory=lambda: random.randint(0, 1_000_000_000), metadata={"help": "Seed for reproducibility."})
    model_name: str = field(default="my_model", metadata={"help": "Model name."})
    save_dir: str = field(default="out", metadata={"help": "Directory to save the model."})
    save_dtype: str = field(default="fp32", metadata={"help": "Save dtype."})
    weight_dtype: str = field(default="fp32", metadata={"help": "Weight dtype."})
    mixed_precision: Literal["float16", "bfloat16", "fp16", "bf16"] = field(default="bf16", metadata={"help": "Mixed precision."})
    prediction_type: Literal["epsilon", "v_prediction", "sample"] | None = field(
        default=None,
        metadata={"help": "Prediction type."},
    )
    n_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})

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

    noise_offset: float = field(default=0.02, metadata={"help": "Noise offset for improved training quality. 0.02-0.1 recommended."})
    noise_offset_probability: float = field(
        default=1.0,
        metadata={"help": "Probability of applying noise offset. 0.25 means 25% of the time. 1.0 means always."},
    )
    input_perturbation: float = field(
        default=0.01,
        metadata={"help": "Input perturbation strength for improved training stability. 0.01-0.1 recommended."},
    )
    input_perturbation_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for input perturbation linear decay. 0 means no decay (constant perturbation)."},
    )

    # Multi-resolution noise settings
    use_multires_noise: bool = field(default=True, metadata={"help": "Enable multi-resolution noise for improved training quality."})
    multires_noise_iterations: int = field(default=6, metadata={"help": "Number of noise levels/iterations. Higher = more detail, more computation."})
    multires_noise_discount: float = field(default=0.8, metadata={"help": "Discount factor between levels. Lower = more variation. Range: 0.1-0.9."})
    multires_noise_scales: list[float] | None = field(default=None, metadata={"help": "Custom scales [1.0, 0.5, 0.25]. Overrides iterations if set."})
    multires_noise_weights: list[float] | None = field(default=None, metadata={"help": "Custom weights for scales. Must match scales length."})

    # Advanced SNR options
    use_smooth_min_snr: bool = field(default=True, metadata={"help": "Use smooth Min-SNR weighting instead of hard clipping when SNR gamma is set."})
    smooth_min_snr_mode: Literal["clip", "sigmoid", "tanh"] = field(default="sigmoid", metadata={"help": "Smoothing mode for Min-SNR."})
    smooth_min_snr_factor: float = field(default=0.15, metadata={"help": "Smoothing factor for Min-SNR (higher = less smooth, more stable)."})

    # Adaptive noise scheduling
    use_adaptive_noise: bool = field(default=False, metadata={"help": "Use adaptive noise scheduling based on timesteps."})
    adaptive_noise_type: Literal["linear", "cosine", "exponential"] = field(default="cosine", metadata={"help": "Type of adaptive noise schedule."})
    adaptive_noise_strength: float = field(default=1.0, metadata={"help": "Strength factor for adaptive noise."})

    # Flash Attention (xformers) support
    enable_flash_attention: bool = field(
        default=True,
        metadata={"help": "Enable Flash Attention (xformers) for memory efficiency. Reduces VRAM usage by 30-50%."},
    )
    flash_attention_unet: bool = field(default=True, metadata={"help": "Enable Flash Attention for UNet (recommended)."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing."})
    timestep_bias_strategy: Literal["uniform", "logit"] = field(
        default="uniform",
        metadata={"help": "Timestep bias strategy."},
    )
    timestep_bias_m: float = field(
        default=0.0,
        metadata={"help": "Mean (m) parameter for logit timestep bias. Controls the center of the log-normal distribution."},
    )
    timestep_bias_s: float = field(
        default=1.0,
        metadata={"help": "Scale (s) parameter for logit timestep bias. Controls the spread of the log-normal distribution."},
    )
    # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
    # More details here: https://arxiv.org/abs/2303.09556.
    snr_gamma: float = field(default=5.0, metadata={"help": "SNR gamma. Recommended value is 5.0."})
    # Use debiased estimation technique to weight the loss by SNR, making the model focus more on high SNR (low noise) regions
    use_debiased_estimation: bool = field(
        default=False,
        metadata={"help": "Use debiased estimation technique to reweight loss. Focuses learning on high SNR (low noise) regions."},
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    use_ema: bool = field(default=False, metadata={"help": "Use EMA."})
    save_every_n_steps: int = field(default=0, metadata={"help": "Save every n steps."})
    save_every_n_epochs: int = field(default=1, metadata={"help": "Save every n epochs."})
    preview_every_n_steps: int = field(default=0, metadata={"help": "Preview every n steps."})
    preview_every_n_epochs: int = field(default=1, metadata={"help": "Preview every n epochs."})
    preview_before_training: bool = field(default=True, metadata={"help": "Generate preview before training starts."})

    log_with: Literal["wandb", "tensorboard", "none"] = field(default="wandb", metadata={"help": "Logger."})

    optimizer: Literal["adamW8bit", "adafactor"] = field(default="adamW8bit", metadata={"help": "Optimizer."})
    optimizer_warmup_steps: int = field(default=0, metadata={"help": "Optimizer warmup steps."})
    optimizer_num_cycles: int = field(default=1, metadata={"help": "Optimizer num cycles."})

    zero_grad_set_to_none: bool = field(default=True, metadata={"help": "Zero grad set to none."})
    preview_sample_options: list[SampleOptions] = field(default_factory=list, metadata={"help": "Preview sample options."})

    checkpoint_every_n_steps: int = field(default=1000, metadata={"help": "Checkpoint steps."})
    checkpoint_epochs: int = field(default=0, metadata={"help": "Checkpoint epochs."})

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
    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    text_encoder_1_lr: float = field(default=1e-6, metadata={"help": "Text encoder 1 learning rate."})
    text_encoder_2_lr: float = field(default=1e-6, metadata={"help": "Text encoder 2 learning rate."})


@dataclass
class SD15Config(BaseConfig):
    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    text_encoder_lr: float = field(default=1e-6, metadata={"help": "Text encoder learning rate."})
