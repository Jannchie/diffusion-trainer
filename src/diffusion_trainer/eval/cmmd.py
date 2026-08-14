"""CMMD: Maximum Mean Discrepancy between two sets of image embeddings.

Follows the reference implementation from google-research/cmmd (the
"Rethinking FID" paper): RBF kernel with bandwidth sigma=10 over
l2-normalized embeddings, biased estimator, reported at 1000x scale.
The paper uses CLIP ViT-L/14@336 embeddings, but the estimator itself is
backbone-agnostic — any l2-normalized dual-encoder embedding works, which
is what lets us swap in SigLIP2.
"""

import torch

# Reference constants from google-research/cmmd. Keeping them fixed makes our
# numbers comparable with published CMMD values (for the same backbone).
_SIGMA = 10.0
_SCALE = 1000.0


def _mean_rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float, block_size: int) -> float:
    """Mean of exp(-gamma * ||xi - yj||^2) over all pairs, block-wise to bound memory at large N."""
    x_sq = (x * x).sum(dim=1)
    y_sq = (y * y).sum(dim=1)
    total = 0.0
    for i in range(0, x.shape[0], block_size):
        xb, xb_sq = x[i : i + block_size], x_sq[i : i + block_size]
        for j in range(0, y.shape[0], block_size):
            yb, yb_sq = y[j : j + block_size], y_sq[j : j + block_size]
            # ||a-b||^2 expanded as a matmul — cdist would take a sqrt only for
            # us to square it right back. Clamp guards tiny negative rounding.
            sq_dists = (xb_sq[:, None] + yb_sq[None, :] - 2.0 * (xb @ yb.T)).clamp_(min=0)
            total += torch.exp(-gamma * sq_dists).sum().item()
    return total / (x.shape[0] * y.shape[0])


def compute_cmmd(ref_embs: torch.Tensor, gen_embs: torch.Tensor, block_size: int = 1024) -> float:
    """CMMD between reference and generated embedding sets, each (N, D) l2-normalized float32."""
    if ref_embs.ndim != 2 or gen_embs.ndim != 2 or ref_embs.shape[1] != gen_embs.shape[1]:
        msg = f"expected (N, D) embedding matrices with matching D, got {tuple(ref_embs.shape)} and {tuple(gen_embs.shape)}"
        raise ValueError(msg)
    x = ref_embs.to(torch.float64)
    y = gen_embs.to(torch.float64)
    gamma = 1.0 / (2.0 * _SIGMA**2)
    k_xx = _mean_rbf_kernel(x, x, gamma, block_size)
    k_yy = _mean_rbf_kernel(y, y, gamma, block_size)
    k_xy = _mean_rbf_kernel(x, y, gamma, block_size)
    return _SCALE * (k_xx + k_yy - 2.0 * k_xy)
