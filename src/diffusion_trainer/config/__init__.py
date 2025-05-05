import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SampleOptions:
    prompt: str
    negative_prompt: str
    steps: int
    seed: int
    width: int | None
    height: int | None


@dataclass
class BaseConfig:
    model_path: str = field(metadata={"help": "Path to the model."})
    meta_path: str = field(metadata={"help": "Path to the metadata."})
    image_path: str | None = field(default=None, metadata={"help": "Path to the images."})
    skip_prepare_image: bool = field(default=False, metadata={"help": "Skip preparing the images."})

    vae_path: str | None = field(default=None, metadata={"help": "Path to the VAE."})
    vae_dtype: str = field(default="fp32", metadata={"help": "the VAE dtype."})

    ss_latent_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset latents."})
    ss_meta_path: str | None = field(default=None, metadata={"help": "Path to the ss-script dataset metadata."})

    shuffle_tags: bool = field(default=False, metadata={"help": "Shuffle tags."})
    single_tag_dropout: float = field(default=0.0, metadata={"help": "Single tag dropout."})
    all_tags_dropout: float = field(default=0.0, metadata={"help": "All tags dropout."})
    caption_dropout: float = field(default=0.0, metadata={"help": "Caption dropout."})

    seed: int = field(default_factory=lambda: random.randint(0, 1_000_000_000), metadata={"help": "Seed for reproducibility."})
    model_name: str = field(default="my_model", metadata={"help": "Model name."})
    save_dir: str = field(default="out", metadata={"help": "Directory to save the model."})
    save_dtype: str = field(default="fp16", metadata={"help": "Save dtype."})
    weight_dtype: str = field(default="bf16", metadata={"help": "Weight dtype."})
    mixed_precision: Literal["float16", "bfloat16", "fp16", "bf16"] = field(default="bf16", metadata={"help": "Mixed precision."})
    prediction_type: Literal["epsilon", "v_prediction", "sample"] | None = field(
        default=None,
        metadata={"help": "Prediction type."},
    )
    n_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})

    mode: Literal["full-finetune", "lora", "lokr", "loha"] = field(default="lokr", metadata={"help": "Mode."})
    lora_rank: int = field(default=4, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=1, metadata={"help": "Lora alpha."})
    lokr_factor: int = field(default=16, metadata={"help": "LoKr factor."})

    noise_offset: float = field(default=0.0, metadata={"help": "Noise offset. 0.1 is recommended."})
    input_perturbation: float = field(default=0.0, metadata={"help": "Input perturbation strength for improved training stability. 0.1 is recommended."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing."})
    timestep_bias_strategy: Literal["uniform", "logit"] = field(
        default="uniform",
        metadata={"help": "Timestep bias strategy."},
    )
    # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
    # More details here: https://arxiv.org/abs/2303.09556.
    snr_gamma: float | None = field(default=None, metadata={"help": "SNR gamma. Recommended value is 5.0."})
    # 使用去偏估计技术，通过SNR权重加权损失，使得模型更关注高SNR（低噪声）区域
    use_debiased_estimation: bool = field(
        default=False, metadata={"help": "Use debiased estimation technique to reweight loss. Focuses learning on high SNR (low noise) regions."}
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    use_ema: bool = field(default=False, metadata={"help": "Use EMA."})
    save_every_n_steps: int = field(default=0, metadata={"help": "Save every n steps."})
    save_every_n_epochs: int = field(default=1, metadata={"help": "Save every n epochs."})
    preview_every_n_steps: int = field(default=0, metadata={"help": "Preview every n steps."})
    preview_every_n_epochs: int = field(default=1, metadata={"help": "Preview every n epochs."})

    log_with: Literal["wandb", "tensorboard", "none"] = field(default="wandb", metadata={"help": "Logger."})

    optimizer: Literal["adamW8bit", "adafactor"] = field(default="adamW8bit", metadata={"help": "Optimizer."})
    optimizer_warmup_steps: int = field(default=0, metadata={"help": "Optimizer warmup steps."})
    optimizer_num_cycles: int = field(default=1, metadata={"help": "Optimizer num cycles."})

    zero_grad_set_to_none: bool = field(default=False, metadata={"help": "Zero grad set to none."})
    preview_sample_options: list[SampleOptions] = field(default_factory=list, metadata={"help": "Preview sample options."})

    checkpoint_every_n_steps: int = field(default=1000, metadata={"help": "Checkpoint steps."})
    checkpoint_epochs: int = field(default=0, metadata={"help": "Checkpoint epochs."})

    def __post_init__(self) -> None:
        # convert preview_sample_options to SampleOptions
        self.preview_sample_options = [item if isinstance(item, SampleOptions) else SampleOptions(**item) for item in self.preview_sample_options]


@dataclass
class SDXLConfig(BaseConfig):
    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    text_encoder_1_lr: float = field(default=1e-6, metadata={"help": "Text encoder 1 learning rate."})
    text_encoder_2_lr: float = field(default=1e-6, metadata={"help": "Text encoder 2 learning rate."})


@dataclass
class SD15Config(BaseConfig):
    unet_lr: float = field(default=1e-5, metadata={"help": "UNet learning rate."})
    text_encoder_lr: float = field(default=1e-6, metadata={"help": "Text encoder learning rate."})
