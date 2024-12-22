import secrets
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SampleOptions:
    prompt: str
    negative_prompt: str
    steps: int
    seed: int


@dataclass
class SDXLConfig:
    model_path: str = field(metadata={"help": "Path to the model."})

    image_path: str | None = field(default=None, metadata={"help": "Path to the images."})
    skip_prepare_image: bool = field(default=False, metadata={"help": "Skip preparing the images."})

    meta_path: str | None = field(default=None, metadata={"help": "Path to the metadata."})
    vae_path: str | None = field(default=None, metadata={"help": "Path to the VAE."})
    vae_dtype: str = field(default="fp32", metadata={"help": "the VAE dtype."})
    ss_latent_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset latents."})
    ss_meta_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset metadata."})

    seed: int = field(default_factory=lambda: secrets.randbelow(1_000_000_000), metadata={"help": "Seed for reproducibility."})
    model_name: str = field(default="my_model", metadata={"help": "Model name."})
    save_dir: str = field(default="out", metadata={"help": "Directory to save the model."})
    save_dtype: str = field(default="fp16", metadata={"help": "Save dtype."})
    weight_dtype: str = field(default="bf16", metadata={"help": "Weight dtype."})
    mixed_precision: Literal["float16", "bfloat16", "fp16", "bf16"] = field(default="bf16", metadata={"help": "Mixed precision."})
    prediction_type: Literal["epsilon", "v_prediction", "sample"] | None = field(default=None, metadata={"help": "Prediction type."})
    n_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    text_encoder_1_lr: float = field(default=1e-6, metadata={"help": "Text encoder 1 learning rate."})
    text_encoder_2_lr: float = field(default=1e-6, metadata={"help": "Text encoder 2 learning rate."})
    mode: Literal["full-finetune", "lora", "lokr", "loha"] = field(default="lokr", metadata={"help": "Mode."})
    lora_rank: int = field(default=4, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=1, metadata={"help": "Lora alpha."})
    lokr_factor: int = field(default=16, metadata={"help": "LoKr factor."})
    noise_offset: float = field(default=0.0, metadata={"help": "Noise offset."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing."})
    timestep_bias_strategy: Literal["none", "earlier", "later", "range"] = field(
        default="none",
        metadata={"help": "Timestep bias strategy."},
    )
    timestep_bias_multiplier: float = field(default=1.0, metadata={"help": "Timestep bias multiplier."})
    timestep_bias_portion: float = field(default=0.25, metadata={"help": "Timestep bias portion."})
    timestep_bias_begin: int = field(default=0, metadata={"help": "Timestep bias begin."})
    timestep_bias_end: int = field(default=1000, metadata={"help": "Timestep bias end."})
    # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
    # More details here: https://arxiv.org/abs/2303.09556.
    snr_gamma: float | None = field(default=None, metadata={"help": "SNR gamma. Recommended value is 5.0."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    use_ema: bool = field(default=False, metadata={"help": "Use EMA."})
    save_every_n_steps: int = field(default=0, metadata={"help": "Save every n steps."})
    save_every_n_epochs: int = field(default=1, metadata={"help": "Save every n epochs."})
    preview_every_n_steps: int = field(default=0, metadata={"help": "Preview every n steps."})
    preview_every_n_epochs: int = field(default=1, metadata={"help": "Preview every n epochs."})
    log_with: Literal["wandb", "tensorboard", "none"] = field(default="none", metadata={"help": "Logger."})

    gradient_precision: Literal["fp32", "fp16", "bf16"] = field(default="fp32", metadata={"help": "Gradient precision."})

    optimizer: Literal["adamW8bit", "adafactor"] = field(default="adamW8bit", metadata={"help": "Optimizer."})
    optimizer_warmup_steps: int = field(default=0, metadata={"help": "Optimizer warmup steps."})
    optimizer_num_cycles: int = field(default=1, metadata={"help": "Optimizer num cycles."})

    zero_grad_set_to_none: bool = field(default=False, metadata={"help": "Zero grad set to none."})
    preview_sample_options: list[SampleOptions] = field(default_factory=list, metadata={"help": "Preview sample options."})

    def __post_init__(self) -> None:
        # convert preview_sample_options to SampleOptions
        self.preview_sample_options = [
            item if isinstance(item, SampleOptions) else SampleOptions(**item) for item in self.preview_sample_options
        ]