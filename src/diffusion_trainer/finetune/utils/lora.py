from logging import getLogger
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
from lycoris import LycorisNetwork, create_lycoris
from torch import nn
from torch.nn.utils import parametrize

from diffusion_trainer.finetune.utils import format_size

if TYPE_CHECKING:
    from diffusion_trainer.config import BaseConfig

logger = getLogger("diffusion_trainer")


class LoraTargets(NamedTuple):
    """Module CLASS NAMES LyCORIS should wrap, by role.

    LyCORIS matches targets by class name, so every architecture spells these
    differently: UNets (SD 1.5 / SDXL) use diffusers' ``FeedForward``, Lumina
    2's DiT uses ``LuminaFeedForward`` (a SwiGLU with linear_1/2/3). A preset
    naming the wrong one silently wraps only the attention blocks — the run
    trains, converges worse, and nothing warns you.

    Roles are declared rather than inferred: lokr gives attention and
    feed-forward different factors, so guessing "whatever isn't called
    Attention is the FFN" would misfactor any architecture whose attention
    class has a family-specific name.
    """

    attention: tuple[str, ...]
    feedforward: tuple[str, ...]
    # The convolutional path, wrapped by locon only -- the 3x3 convs in
    # ResnetBlock2D (conv1/conv2/conv_shortcut) and the Down/Upsample2D convs,
    # where most of an image's texture and style lives. Those classes contain
    # no attention or feed-forward, so there is no double-wrapping. Empty for
    # DiTs, which have no conv path at all (locon then degenerates to lora).
    conv: tuple[str, ...] = ()

    @property
    def linear(self) -> list[str]:
        return [*self.attention, *self.feedforward]

    @property
    def all_targets(self) -> list[str]:
        return [*self.linear, *self.conv]


UNET_LORA_TARGETS = LoraTargets(
    attention=("Attention",),
    feedforward=("FeedForward",),
    conv=("ResnetBlock2D", "Downsample2D", "Upsample2D"),
)
LUMINA2_LORA_TARGETS = LoraTargets(attention=("Attention",), feedforward=("LuminaFeedForward",))
# MOSS-Image keeps attention and the SiTU FFN inside one class, ``_Block``,
# whose linears sit in a plain ModuleDict (proj.q/k/v/o, proj.ffn_up_gate,
# proj.ffn_down) — there is no FFN class to target separately. Wrapping the
# block reaches all six: the 42 joint blocks only, which is Kohaku's
# "attn-mlp" preset; the text refiner (``_TextBlock``) and the modulation
# heads stay frozen. The cost is that lokr applies its attention factor to the
# FFN linears too.
MOSS_IMAGE_LORA_TARGETS = LoraTargets(attention=("_Block",), feedforward=())


def apply_lora_config(
    mode: Literal["lora", "loha", "lokr", "locon"],
    model: torch.nn.Module,
    config: "BaseConfig | None" = None,
    targets: LoraTargets = UNET_LORA_TARGETS,
) -> LycorisNetwork:
    """Wrap ``model`` with a LyCORIS network targeting ``targets``."""
    # Use config values if provided, otherwise use defaults
    lora_dim = config.lora_dim if config else 16
    lora_alpha = config.lora_alpha if config else 1.0
    lora_dropout = config.lora_dropout if config else 0.0
    lokr_factor = config.lokr_factor if config else 16

    # Convolutional layer parameters (fallback to linear parameters if not specified)
    conv_dim = config.conv_dim if config and config.conv_dim is not None else lora_dim
    conv_alpha = config.conv_alpha if config and config.conv_alpha is not None else lora_alpha

    # Advanced configuration parameters
    multiplier = config.lora_multiplier if config else 1.0
    lokr_linear_dim = config.lokr_linear_dim if config else 10000
    lokr_ff_factor_ratio = config.lokr_feedforward_factor_ratio if config else 0.5

    if mode == "lora":
        lycoris_config = {
            "algo": "lora",
            "multiplier": multiplier,
            "linear_dim": lora_dim,
            "linear_alpha": lora_alpha,
            "conv_dim": conv_dim,
            "conv_alpha": conv_alpha,
        }
        LycorisNetwork.apply_preset({"target_module": targets.linear})
    elif mode == "loha":
        lycoris_config = {
            "algo": "loha",
            "multiplier": multiplier,
            "linear_dim": lora_dim,
            "linear_alpha": lora_alpha,
            "conv_dim": conv_dim,
            "conv_alpha": conv_alpha,
        }
        LycorisNetwork.apply_preset({"target_module": targets.linear})
    elif mode == "lokr":
        lycoris_config = {
            "algo": "lokr",
            "multiplier": multiplier,
            "linear_dim": lokr_linear_dim,
            "linear_alpha": 1,  # Ignored when using full dimension
            "factor": lokr_factor,
        }
        ff_factor = max(1, int(lokr_factor * lokr_ff_factor_ratio))
        LycorisNetwork.apply_preset(
            {
                "target_module": targets.linear,
                "module_algo_map": {
                    **{name: {"factor": lokr_factor} for name in targets.attention},
                    **{name: {"factor": ff_factor} for name in targets.feedforward},
                },
            },
        )
    elif mode == "locon":
        lycoris_config = {
            "algo": "locon",
            "multiplier": multiplier,
            "linear_dim": lora_dim,
            "linear_alpha": lora_alpha,
            "linear_dropout": lora_dropout,
            "conv_dim": conv_dim,
            "conv_alpha": conv_alpha,
            "conv_dropout": lora_dropout,
        }
        # LoCon = LoRA-for-Convolution: the linear path plus the architecture's
        # conv classes, to which conv_dim/conv_alpha apply. Architectures with
        # no conv path (DiTs) declare an empty tuple, making locon == lora.
        LycorisNetwork.apply_preset({"target_module": targets.all_targets})
    else:  # type: ignore[misc]  # Defensive programming for runtime safety
        msg = f"Unsupported mode: {mode}"
        raise ValueError(msg)

    lycoris_model = create_lycoris(
        model,
        **lycoris_config,
    )
    lycoris_model.apply_to()
    lora_init = config.lora_init if config else "default"
    if lora_init != "default":
        if mode not in ("lora", "locon"):
            msg = f"lora_init={lora_init!r} needs a lora_down projection; mode {mode!r} has none."
            raise ValueError(msg)
        apply_lora_init(lycoris_model, lora_init)
        logger.info("LoRA down-projection initialized with %s", lora_init)
    lycoris_num_params = sum(p.numel() for p in lycoris_model.parameters())
    logger.info(
        "LyCORIS network (%s) has been initialized with %s parameters (%s)",
        mode.upper(),
        f"{lycoris_num_params:,}",
        format_size(lycoris_num_params),
    )
    return lycoris_model


_NORA_EPS = 1e-8


def normalize_columns(a: torch.Tensor) -> torch.Tensor:
    """Scale each input column of a down-projection to unit L2 norm over the rank dim.

    ``a`` is ``lora_down.weight``: (r, in) for linears, (r, in, kh, kw) for
    convs, where every (in, kh, kw) position counts as one column. The norm is
    taken in fp32 -- in bf16 a rank-32 sum of squares loses the last digits.
    """
    flat = a.reshape(a.shape[0], -1).float()
    flat = flat / flat.norm(dim=0, keepdim=True).clamp_min(_NORA_EPS)
    return flat.reshape(a.shape).to(a.dtype)


class _ColumnNorm(nn.Module):
    """Parametrization behind persistent NoRA: forward sees Norm(A), the optimizer the raw A."""

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return normalize_columns(a)

    def right_inverse(self, a: torch.Tensor) -> torch.Tensor:
        # Norm is idempotent, so an already-normalized A (what checkpoints
        # store) is its own preimage.
        return a


def _rename_to_parametrized(state_dict: dict, prefix: str, *_args: object) -> None:
    # LoConModule.custom_state_dict writes the *normalized* A as
    # ``lora_down.weight`` -- exactly what an exported LoRA must hold -- but a
    # parametrized Linear loads from ``parametrizations.weight.original``.
    # Resuming therefore restarts each raw column at norm 1; the forward pass
    # is unchanged, only Adam's per-column step size shifts for a few steps.
    key = f"{prefix}weight"
    if key in state_dict:
        state_dict[f"{prefix}parametrizations.weight.original"] = state_dict.pop(key)


def _block_identity_(a: torch.Tensor) -> None:
    """BIMI: A = [I_r, I_r, ..., E_q]; column j is the basis vector e_(j mod r)."""
    flat = torch.zeros(a.shape[0], a[0].numel(), dtype=a.dtype, device=a.device)
    cols = torch.arange(flat.shape[1], device=a.device)
    flat[cols % a.shape[0], cols] = 1
    a.copy_(flat.reshape(a.shape))


@torch.no_grad()
def apply_lora_init(network: LycorisNetwork, lora_init: Literal["nora", "nora-init", "bimi"]) -> None:
    """Re-initialize every ``lora_down`` per Normalized LoRA (arXiv 2608.31036).

    B stays zero, so the network still starts as the identity. What changes is
    A's per-input scale: LyCORIS's kaiming init gives columns of norm about
    sqrt(r / 3k), these give 1, i.e. A grows by ~sqrt(3k / r) -- and with it the
    effective lr of B. An lr tuned for default LoRA is not a fair baseline.
    """
    for module in network.loras:
        down = module.lora_down
        if lora_init == "bimi":
            _block_identity_(down.weight)
        else:
            down.weight.copy_(normalize_columns(down.weight))
        if lora_init == "nora":
            parametrize.register_parametrization(down, "weight", _ColumnNorm())
            down._register_load_state_dict_pre_hook(_rename_to_parametrized)  # noqa: SLF001
