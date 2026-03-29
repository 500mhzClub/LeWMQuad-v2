"""Sketched-Isotropic-Gaussian Regularizer (SIGReg).

Enforces latent embeddings to match an isotropic Gaussian N(0, I) by:
1. Projecting embeddings onto M random unit-norm directions (Cramér-Wold)
2. Computing the Epps-Pulley normality test statistic on each 1-D projection
3. Averaging over all projections

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
    Learning without the Heuristics", 2025.  (arXiv:2511.08544)
"""
from __future__ import annotations

import torch



def sigreg(
    Z: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
) -> torch.Tensor:
    """Compute the SIGReg loss for a batch of embeddings.

    Matches the official le-wm repo exactly:
      - t in [0, 3] with Gaussian window w(t) = exp(-t²/2)
      - Result scaled by B (so lambda=0.1 has correct magnitude)
      - Computation runs in float32 for numerical stability under bfloat16 autocast

    Args:
        Z: (B, D) batch of latent embeddings.
        n_projections: number of random unit-norm directions M.
        n_knots: number of trapezoid quadrature nodes K.

    Returns:
        Scalar SIGReg loss scaled by B.
    """
    orig_dtype = Z.dtype
    # Run in float32 for numerical stability (safe under bfloat16 autocast)
    Z = Z.float()

    B, D = Z.shape

    # Random unit-norm directions on S^{D-1}
    u = torch.randn(D, n_projections, device=Z.device, dtype=torch.float32)
    u = u / u.norm(p=2, dim=0)

    # Project embeddings:  h^(m) = Z @ u^(m),  shape (B, M)
    h = Z @ u

    # Quadrature nodes t in [0, 3], trapezoid weights, Gaussian window w(t)
    t_nodes = torch.linspace(0.0, 3.0, n_knots, device=Z.device, dtype=torch.float32)
    dt = 3.0 / (n_knots - 1)
    trap_w = torch.full((n_knots,), 2 * dt, device=Z.device, dtype=torch.float32)
    trap_w[0] = dt
    trap_w[-1] = dt
    window = torch.exp(-0.5 * t_nodes * t_nodes)   # w(t) = exp(-t²/2)
    weights = trap_w * window                        # combined quadrature weights

    # (B, M, K)
    th = h.unsqueeze(-1) * t_nodes.view(1, 1, -1)
    ecf_real = th.cos().mean(dim=0)   # (M, K)  — mean over B
    ecf_imag = th.sin().mean(dim=0)   # (M, K)

    diff_sq = (ecf_real - window.unsqueeze(0)).square() + ecf_imag.square()
    statistic = (diff_sq * weights.unsqueeze(0)).sum(dim=-1)  # (M,)

    # Scale by B to keep lambda-scale consistent with official repo
    return (statistic.mean() * B).to(orig_dtype)


def sigreg_stepwise(
    Z_seq: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
    **_kwargs,
) -> torch.Tensor:
    """Step-wise SIGReg: average SIGReg independently at each time step.

    Args:
        Z_seq: (B, T, D) — embeddings over a temporal sequence.

    Returns:
        Scalar loss (mean over T time-steps).
    """
    B, T, D = Z_seq.shape
    total = torch.tensor(0.0, device=Z_seq.device, dtype=Z_seq.dtype)
    for t in range(T):
        total = total + sigreg(Z_seq[:, t], n_projections, n_knots)
    return total / T
