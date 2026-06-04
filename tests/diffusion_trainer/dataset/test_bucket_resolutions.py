"""Tests for the per-base-resolution bucket tables used by latents encoding."""

import pytest

from diffusion_trainer.dataset.processors.latents_generate_processor import PREDEFINED_RESOS, SimpleLatentsProcessor


@pytest.mark.parametrize("base", sorted(PREDEFINED_RESOS))
def test_bucket_tables_are_unet_safe(base: int) -> None:
    resos = PREDEFINED_RESOS[base]
    for width, height in resos:
        assert width % 64 == 0 and height % 64 == 0, f"{width}x{height} not divisible by 64"  # noqa: PT018
        assert width * height <= base * base, f"{width}x{height} exceeds {base}^2 area budget"
    # Mirrored pairs keep portrait/landscape coverage symmetric.
    assert {(h, w) for w, h in resos} == set(resos)
    # Sorted by aspect ratio so argmin lookup semantics stay obvious.
    ars = [w / h for w, h in resos]
    assert ars == sorted(ars)


def test_unknown_base_resolution_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SimpleLatentsProcessor, "load_vae_model", lambda _self: None)
    with pytest.raises(ValueError, match="base_resolution"):
        SimpleLatentsProcessor("dummy-vae", base_resolution=640)


def test_select_reso_uses_requested_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SimpleLatentsProcessor, "load_vae_model", lambda _self: None)
    processor = SimpleLatentsProcessor("dummy-vae", base_resolution=768)

    reso, _resized = processor.select_reso(1000, 1000)
    assert tuple(reso) == (768, 768)
    reso, _resized = processor.select_reso(896, 1992)
    assert tuple(reso) == (512, 1152)
