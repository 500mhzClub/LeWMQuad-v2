"""Learned scalar energy heads for latent-space planning."""
from __future__ import annotations

import torch
import torch.nn as nn


class GoalEnergyHead(nn.Module):
    """Scores compatibility between a predicted latent and a goal latent.

    Input: concatenation of [z_pred, z_goal, z_pred - z_goal, z_pred * z_goal].
    Output: scalar energy (lower = more compatible).
    """

    def __init__(self, latent_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)


class LatentEnergyHead(nn.Module):
    """Maps a single latent to a scalar energy for trajectory scoring.

    Trained as a probe on frozen encoder latents.  The target is a
    composite of three dataset label channels:

        energy = w_safety * (1 - clearance/clip)
               + w_mobility * (1 - traversability/horizon)
               + w_beacon * (beacon_range/clip)        [when visible]

    Lower energy → safer, more traversable, closer to a beacon.

    This gives useful gradient *everywhere* in the maze (safety alone
    keeps the agent off walls and out of dead ends), while beacon
    attraction takes over once a breadcrumb is within sensor range.

    At planning time, score a candidate action sequence by summing
    per-step energies over the ``plan_rollout`` horizon.
    """

    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) -> (...,) non-negative scalar energy."""
        return self.net(z).squeeze(-1)

    def score_trajectory(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Score a full rollout. Lower is better.

        Args:
            z_seq: (B, H, D) predicted latents from plan_rollout.

        Returns:
            (B,) total trajectory energy (sum over horizon).
        """
        per_step = self.forward(z_seq)      # (B, H)
        return per_step.sum(dim=-1)         # (B,)


def composite_energy_target(
    clearance: torch.Tensor,
    traversability: torch.Tensor,
    beacon_range: torch.Tensor,
    clearance_clip: float = 1.0,
    traversability_horizon: int = 10,
    beacon_clip: float = 5.0,
    w_safety: float = 0.5,
    w_mobility: float = 0.3,
    w_beacon: float = 0.2,
) -> torch.Tensor:
    """Build a composite energy target from dataset labels.

    All three terms are normalised to [0, 1] before weighting.
    Returns a non-negative target where 0 = ideal state.

    Args:
        clearance: min distance to any obstacle (metres). Higher = safer.
        traversability: forward free steps (0..horizon). Higher = more open.
        beacon_range: distance to closest visible beacon (metres).
            999 = no beacon visible (treated as neutral, not penalised).
        clearance_clip: saturate clearance contribution beyond this distance.
        traversability_horizon: max traversability value from label pipeline.
        beacon_clip: saturate beacon_range contribution beyond this distance.
        w_safety: weight for wall-proximity penalty.
        w_mobility: weight for dead-end / blocked penalty.
        w_beacon: weight for beacon attraction.
    """
    # Safety: 0 when far from walls, 1 when touching a wall
    safety = 1.0 - (clearance / clearance_clip).clamp(0, 1)

    # Mobility: 0 when open corridor, 1 when completely stuck
    mobility = 1.0 - (traversability.float() / traversability_horizon).clamp(0, 1)

    # Beacon: 0 when on top of beacon, 1 when far / not visible
    beacon_norm = (beacon_range / beacon_clip).clamp(0, 1)
    # Mask out the beacon term where no beacon is visible (range >= 999)
    # so the head doesn't learn to penalise beacon-free corridors
    beacon_visible = beacon_range < beacon_clip * 2
    beacon_term = torch.where(beacon_visible, beacon_norm, torch.zeros_like(beacon_norm))
    # Re-weight: when no beacon is visible, redistribute its weight to safety
    w_safety_eff = torch.where(beacon_visible, torch.tensor(w_safety), torch.tensor(w_safety + w_beacon))
    w_beacon_eff = torch.where(beacon_visible, torch.tensor(w_beacon), torch.tensor(0.0))

    return w_safety_eff * safety + w_mobility * mobility + w_beacon_eff * beacon_term
