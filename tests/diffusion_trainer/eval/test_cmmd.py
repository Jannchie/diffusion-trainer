import torch
import torch.nn.functional as F

from diffusion_trainer.eval.cmmd import compute_cmmd


def _random_unit(n: int, d: int, seed: int, shift: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x[:, 0] += shift
    return F.normalize(x, dim=-1)


def test_identical_sets_give_zero() -> None:
    x = _random_unit(200, 64, seed=0)
    assert abs(compute_cmmd(x, x)) < 1e-6


def test_symmetric() -> None:
    x = _random_unit(150, 64, seed=1)
    y = _random_unit(180, 64, seed=2, shift=1.0)
    assert compute_cmmd(x, y) == compute_cmmd(y, x)


def test_larger_shift_gives_larger_cmmd() -> None:
    x = _random_unit(300, 64, seed=3)
    near = compute_cmmd(x, _random_unit(300, 64, seed=4, shift=0.5))
    far = compute_cmmd(x, _random_unit(300, 64, seed=4, shift=2.0))
    assert 0 < near < far


def test_blocked_matches_unblocked() -> None:
    x = _random_unit(130, 32, seed=5)
    y = _random_unit(170, 32, seed=6, shift=1.0)
    assert abs(compute_cmmd(x, y, block_size=7) - compute_cmmd(x, y, block_size=4096)) < 1e-9
