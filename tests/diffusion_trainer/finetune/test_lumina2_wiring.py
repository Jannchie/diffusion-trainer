"""Wiring checks for the Lumina 2 trainer that don't need the 10 GB checkpoint.

These cover the places where a DiT silently diverges from the UNet assumptions
baked into the rest of the trainer: LyCORIS matches wrap targets by class name,
the DiT's forward signature is keyword-shaped and mask-carrying, and the
16-channel VAE adds a shift term to the latent normalization.
"""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel

from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.utils.lora import LUMINA2_LORA_TARGETS, UNET_LORA_TARGETS

# The fixture itself is injected by pytest; only its type needs naming here.
TinyTransformerFactory = Callable[..., Lumina2Transformer2DModel]


def test_lumina_feedforward_class_name_is_not_the_unet_one(tiny_transformer: TinyTransformerFactory) -> None:
    """The reason LUMINA2_LORA_TARGETS exists.

    LyCORIS matches by class name, so the UNet preset's "FeedForward" would miss
    every FFN in the DiT and quietly wrap only the attention blocks.
    """
    module_classes = {type(module).__name__ for _, module in tiny_transformer().named_modules()}

    assert "LuminaFeedForward" in module_classes
    assert "FeedForward" not in module_classes
    assert set(LUMINA2_LORA_TARGETS.linear) <= module_classes
    assert not set(UNET_LORA_TARGETS.linear) <= module_classes
    # A DiT has no conv path, so locon must degenerate to lora rather than
    # silently wrapping nothing extra.
    assert LUMINA2_LORA_TARGETS.conv == ()


def test_target_modules_cover_both_attention_and_ffn(tiny_transformer: TinyTransformerFactory) -> None:
    """Both wrap targets must actually resolve to parameterized submodules."""
    transformer = tiny_transformer()
    matched: dict[str, int] = dict.fromkeys(LUMINA2_LORA_TARGETS.linear, 0)
    for _, module in transformer.named_modules():
        name = type(module).__name__
        if name in matched:
            matched[name] += 1

    assert all(count > 0 for count in matched.values()), matched


def test_transformer_forward_accepts_the_trainer_call_shape(tiny_transformer: TinyTransformerFactory) -> None:
    """Pin the exact keyword call ``Lumina2Tuner.get_model_pred`` makes.

    A diffusers release that renames ``encoder_attention_mask`` (or drops the
    mask) would otherwise surface as a confusing runtime failure deep in a run.
    """
    torch.manual_seed(0)
    transformer = tiny_transformer().eval()

    batch, height, width = 2, 8, 8
    hidden_states = torch.randn(batch, transformer.config.in_channels, height, width)
    encoder_hidden_states = torch.randn(batch, 12, transformer.config.cap_feat_dim)
    encoder_attention_mask = torch.ones(batch, 12, dtype=torch.long)
    # 1 - sigma, the objective's model_timesteps output.
    timestep = torch.tensor([0.3, 0.7])

    with torch.no_grad():
        out = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


def test_attention_mask_changes_the_prediction(tiny_transformer: TinyTransformerFactory) -> None:
    """Padding must be masked out, not attended to.

    Gemma-2 is causal and its pad positions carry arbitrary activations; if the
    mask were dropped on the way to the DiT this test would pass trivially and
    prompt adherence would degrade for every short prompt.
    """
    torch.manual_seed(0)
    transformer = tiny_transformer().eval()

    hidden_states = torch.randn(1, transformer.config.in_channels, 8, 8)
    encoder_hidden_states = torch.randn(1, 12, transformer.config.cap_feat_dim)
    timestep = torch.tensor([0.5])

    full_mask = torch.ones(1, 12, dtype=torch.long)
    short_mask = torch.zeros(1, 12, dtype=torch.long)
    short_mask[:, :4] = 1

    with torch.no_grad():
        kwargs = {"hidden_states": hidden_states, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states, "return_dict": False}
        with_full = transformer(**kwargs, encoder_attention_mask=full_mask)[0]
        with_short = transformer(**kwargs, encoder_attention_mask=short_mask)[0]

    assert not torch.allclose(with_full, with_short)


class _StubVae(torch.nn.Module):
    """Minimal stand-in for the pipeline VAE: config lookups and an identity decode."""

    def __init__(self, scaling_factor: float, shift_factor: float | None) -> None:
        super().__init__()
        self.config = FrozenDict({"scaling_factor": scaling_factor, "shift_factor": shift_factor})
        self._weight = torch.nn.Parameter(torch.zeros(1))
        self.decoded: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor) -> SimpleNamespace:
        self.decoded = latents
        return SimpleNamespace(sample=latents)


class _StubImageProcessor:
    @staticmethod
    def postprocess(image: torch.Tensor, output_type: str) -> list[torch.Tensor]:
        del output_type
        return [image]


class _LatentSpaceTuner(BaseTuner):
    """Concrete BaseTuner whose only working parts are the latent-space methods."""

    def get_pipeline(self) -> object:
        raise NotImplementedError

    def process_batch(self, batch: object) -> object:
        raise NotImplementedError

    def train_each_batch(self, batch: object) -> None:
        raise NotImplementedError

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _configure_full_finetune(self) -> None:
        raise NotImplementedError


def make_scaling_tuner(scaling_factor: float, shift_factor: float | None) -> _LatentSpaceTuner:
    """A tuner wired only enough to exercise the latent-space methods.

    Built with ``__new__`` deliberately: ``__init__`` would spin up an
    accelerator and load a real pipeline, while the two methods under test read
    nothing but ``self.pipeline.vae``.
    """
    tuner = _LatentSpaceTuner.__new__(_LatentSpaceTuner)
    tuner.pipeline = SimpleNamespace(vae=_StubVae(scaling_factor, shift_factor), image_processor=_StubImageProcessor())  # type: ignore[attr-defined]
    return tuner


@pytest.mark.parametrize(
    ("scaling_factor", "shift_factor"),
    [(0.3611, 0.1159), (0.18215, None)],
    ids=["lumina2-16ch", "sd-4ch"],
)
def test_vae_scaling_round_trips_through_the_real_methods(scaling_factor: float, shift_factor: float | None) -> None:
    """Encode and decode must be exact inverses — driving the production code.

    Deliberately calls ``_apply_vae_scaling`` and ``_decode_preview_latents``
    rather than restating their arithmetic: a test that reimplements both sides
    asserts an algebraic identity that holds no matter what the trainer does,
    and would stay green if a sign flipped in either method.
    """
    tuner = make_scaling_tuner(scaling_factor, shift_factor)
    raw = torch.randn(1, 16, 4, 4)

    encoded = tuner._apply_vae_scaling(raw)  # noqa: SLF001
    # The stub decodes to identity, so the returned image IS the un-normalized latent.
    decoded = tuner._decode_preview_latents(encoded)  # noqa: SLF001

    torch.testing.assert_close(decoded, raw, rtol=1e-4, atol=1e-4)


def test_shift_factor_actually_moves_the_latents() -> None:
    """Guards against a trainer that silently ignores shift_factor.

    Without this, the round-trip above would still pass if both methods
    dropped the shift together.
    """
    zeros = torch.zeros(1, 16, 2, 2)

    with_shift = make_scaling_tuner(0.3611, 0.1159)._apply_vae_scaling(zeros)  # noqa: SLF001
    without_shift = make_scaling_tuner(0.3611, None)._apply_vae_scaling(zeros)  # noqa: SLF001

    assert not torch.allclose(with_shift, without_shift)
    torch.testing.assert_close(with_shift, torch.full_like(zeros, -0.1159 * 0.3611))
