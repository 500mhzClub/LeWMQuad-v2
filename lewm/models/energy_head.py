"""Learned scalar energy heads for latent-space planning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEnergyHead(nn.Module):
    """Scores compatibility between a predicted latent and a goal latent.

    Input: concatenation of [z_pred, z_goal, z_pred - z_goal, z_pred * z_goal].
    Output: non-negative scalar energy (lower = more compatible).

    At planning time, the goal latent is the "breadcrumb" — a latent captured
    when the robot first observes the target beacon.  The head learns to
    produce low energy when the current state is approaching that specific
    beacon, and high energy otherwise.
    """

    def __init__(self, latent_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Softplus(),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """z_pred, z_goal: (..., D) -> (...,) non-negative energy."""
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)

    def score_trajectory(
        self, z_seq: torch.Tensor, z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Score a trajectory against a breadcrumb latent.

        Args:
            z_seq:  (B, H, D) predicted latents from plan_rollout.
            z_goal: (B, D) breadcrumb latent from target beacon observation.

        Returns:
            (B,) total goal energy over horizon (lower = closer to beacon).
        """
        B, H, D = z_seq.shape
        z_goal_exp = z_goal.unsqueeze(1).expand(B, H, D)
        per_step = self.forward(z_seq, z_goal_exp)   # (B, H)
        return per_step.sum(dim=-1)                   # (B,)


class ProgressEnergyHead(nn.Module):
    """Optional auxiliary probe for short-horizon goal progress.

    Inputs are the current latent, a future/predicted latent, and a goal latent.
    The output is a bounded progress bonus in ``[0, 1]`` where larger is better.

    This head is useful for experiments, but the default minimal inference stack
    can rely on safety energy, goal energy, and perceptual exploration alone.
    """

    def __init__(self, latent_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_dim * 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_now: torch.Tensor,
        z_future: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return progress bonus in ``[0, 1]`` for each sample."""
        x = torch.cat([
            z_now,
            z_future,
            z_goal,
            z_future - z_now,
            z_goal - z_now,
            z_goal - z_future,
        ], dim=-1)
        return self.net(x).squeeze(-1)

    def score_trajectory(
        self,
        z_seq: torch.Tensor,
        z_now: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return terminal progress bonus for each trajectory."""
        if z_seq.dim() != 3:
            raise ValueError(f"Expected z_seq shape (B, H, D), got {tuple(z_seq.shape)}")
        z_terminal = z_seq[:, -1, :]
        return self.forward(z_now, z_terminal, z_goal)


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


# --------------------------------------------------------------------- #
# Decomposed targets for the new training pipeline
# --------------------------------------------------------------------- #

def composite_safety_target(
    clearance: torch.Tensor,
    traversability: torch.Tensor,
    clearance_clip: float = 1.0,
    traversability_horizon: int = 10,
    w_safety: float = 0.6,
    w_mobility: float = 0.4,
) -> torch.Tensor:
    """Safety + mobility target without any beacon term.

    Used by the unconditional :class:`LatentEnergyHead`.  Beacon attraction
    is handled separately by the :class:`GoalEnergyHead`.

    Returns a value in [0, 1] where 0 = safe, open corridor.
    """
    safety = 1.0 - (clearance / clearance_clip).clamp(0, 1)
    mobility = 1.0 - (traversability.float() / traversability_horizon).clamp(0, 1)
    return w_safety * safety + w_mobility * mobility


def consequence_safety_target(
    clearance: torch.Tensor,
    traversability: torch.Tensor,
    collisions: torch.Tensor,
    traversability_horizon: int = 10,
    contact_clearance: float = 0.08,
    w_contact: float = 0.4,
    w_mobility: float = 0.6,
) -> torch.Tensor:
    """Consequence-based safety target — penalises *outcomes*, not proximity.

    Unlike :func:`composite_safety_target` this does NOT create a repulsive
    gradient around walls.  Walking parallel to a wall at 15 cm is fine;
    only actual contact and dead-end situations are penalised.

    Terms:
        contact — actual collision flag OR clearance below physical-touch
                  threshold.  Binary, not graded.
        mobility — forward traversability, same as before.  Captures
                   dead-ends and facing-wall situations without penalising
                   corridor proximity.

    Returns a value in [0, 1] where 0 = making progress freely.
    """
    # Contact: actual collision OR physically touching
    contact = collisions.float().clamp(0, 1)
    touching = (clearance < contact_clearance).float()
    contact_term = torch.max(contact, touching)

    # Mobility: 0 when open corridor, 1 when completely stuck
    mobility = 1.0 - (traversability.float() / traversability_horizon).clamp(0, 1)

    return w_contact * contact_term + w_mobility * mobility


def beacon_goal_target(
    beacon_range: torch.Tensor,
    beacon_identity: torch.Tensor,
    target_identity: int,
    beacon_clip: float = 5.0,
) -> torch.Tensor:
    """Identity-aware beacon attraction target for :class:`GoalEnergyHead`.

    Returns a value in [0, 1]:
      - 0 = on top of the matching beacon.
      - Proportional to distance when approaching the correct beacon.
      - 1 = wrong beacon, no beacon visible, or out of range.

    Args:
        beacon_range:    (N,) distance to closest visible beacon (metres).
        beacon_identity: (N,) int index of closest visible beacon (-1 = none).
        target_identity: which beacon we are seeking (index into BEACON_FAMILIES).
        beacon_clip:     saturate beyond this range.
    """
    visible = beacon_range < beacon_clip * 2
    matches = beacon_identity == target_identity
    active = visible & matches
    range_norm = (beacon_range / beacon_clip).clamp(0, 1)
    return torch.where(active, range_norm, torch.ones_like(range_norm))


# --------------------------------------------------------------------- #
# RND exploration bonus
# --------------------------------------------------------------------- #

class ExplorationBonus(nn.Module):
    """Random Network Distillation exploration bonus.

    A fixed random network (target) and a trained predictor.  Prediction
    error is high for latent states the predictor has not been trained on
    (i.e. novel / unvisited regions).  At planning time, *subtracting*
    this bonus from trajectory cost encourages the robot to explore
    unseen corridors in a novel maze.

    Training: minimise ``loss(z)`` on the *training* latents so that
    familiar states produce low bonus.  At inference, unvisited states
    will naturally have high bonus.
    """

    def __init__(self, latent_dim: int = 192, feature_dim: int = 128):
        super().__init__()
        # Fixed random target — never trained
        self.target = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Learned predictor — trained to match target on seen states
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Per-sample exploration bonus (higher = more novel).

        Args:
            z: (..., D) latent embeddings.

        Returns:
            (...,) non-negative bonus.
        """
        with torch.no_grad():
            t = self.target(z)
        p = self.predictor(z)
        return (p - t).square().mean(dim=-1)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """Training loss — average prediction error over the batch."""
        return self.forward(z).mean()

    def online_update(
        self,
        z: torch.Tensor,
        lr: float = 1e-3,
        n_steps: int = 1,
    ) -> float:
        """Finetune the predictor on observed latents so revisited states
        lose their novelty bonus.  Call once per inference step with the
        current observation's latent.

        This is intentionally lightweight — one or a few SGD steps on a
        tiny batch so it doesn't slow down the control loop.

        Args:
            z: (1, D) or (D,) latent from the current observation.
            lr: learning rate for the online update.
            n_steps: gradient steps per call.

        Returns:
            The prediction error (bonus) *before* the update.
        """
        z = z.detach()
        if z.ndim == 1:
            z = z.unsqueeze(0)

        with torch.no_grad():
            bonus_before = self.forward(z).mean().item()

        # Enable gradients for the predictor, run manual SGD
        for p in self.predictor.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            for _ in range(n_steps):
                t = self.target(z).detach()
                p = self.predictor(z)
                loss = (p - t).square().mean()
                grads = torch.autograd.grad(loss, self.predictor.parameters())
                for param, grad in zip(self.predictor.parameters(), grads):
                    param.data.sub_(lr * grad)

        for p in self.predictor.parameters():
            p.requires_grad_(False)

        return bonus_before


# --------------------------------------------------------------------- #
# Place-aware rollout snippet embedding
# --------------------------------------------------------------------- #

class PlaceSnippetHead(nn.Module):
    """Embeds rollout snippets into a place-aware metric space.

    The input is a short sequence of rollout latents with fixed length
    ``snippet_len``. The output embedding is L2-normalized so Euclidean
    distance can be used directly for nearest-neighbor novelty.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        snippet_len: int = 3,
        hidden_dim: int = 512,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.snippet_len = int(snippet_len)
        self.embedding_dim = int(embedding_dim)
        in_dim = self.latent_dim * self.snippet_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        if z_seq.ndim == 2:
            flat = z_seq
            expected = self.latent_dim * self.snippet_len
            if int(flat.shape[-1]) != expected:
                raise ValueError(
                    f"Expected flattened snippet dim {expected}, got {tuple(flat.shape)}",
                )
        elif z_seq.ndim == 3:
            if int(z_seq.shape[1]) != self.snippet_len or int(z_seq.shape[2]) != self.latent_dim:
                raise ValueError(
                    "Expected snippet tensor shape "
                    f"(B, {self.snippet_len}, {self.latent_dim}), got {tuple(z_seq.shape)}",
                )
            flat = z_seq.reshape(z_seq.shape[0], -1)
        else:
            raise ValueError(f"Expected 2D or 3D snippet tensor, got {tuple(z_seq.shape)}")
        emb = self.net(flat)
        return F.normalize(emb, dim=-1)


# --------------------------------------------------------------------- #
# Combined trajectory scorer for CEM / MPC planning
# --------------------------------------------------------------------- #

class TrajectoryScorer(nn.Module):
    """Merges safety, goal-seeking, and exploration into a single cost.

    .. math::

        \\text{cost} = E_{\\text{safety}}
                     + \\alpha \\, E_{\\text{goal}}
                     - \\beta  \\, \\text{exploration bonus}

    Lower cost → better trajectory.

    Usage at planning time::

        scorer = TrajectoryScorer(safety_head, goal_head, exploration)
        z_rollouts = world_model.plan_rollout(z_start, candidate_actions)
        costs = scorer.score(z_rollouts, z_breadcrumb)
        best = costs.argmin()
    """

    def __init__(
        self,
        safety_head: LatentEnergyHead,
        goal_head: GoalEnergyHead | None = None,
        progress_head: ProgressEnergyHead | None = None,
        exploration: ExplorationBonus | None = None,
        goal_weight: float = 1.0,
        progress_weight: float = 0.0,
        exploration_weight: float = 0.1,
    ):
        super().__init__()
        self.safety_head = safety_head
        self.goal_head = goal_head
        self.progress_head = progress_head
        self.exploration = exploration
        self.goal_weight = goal_weight
        self.progress_weight = progress_weight
        self.exploration_weight = exploration_weight

    def score(
        self,
        z_seq: torch.Tensor,
        z_goal: torch.Tensor | None = None,
        z_now: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidate trajectories.

        Args:
            z_seq:  (B, H, D) predicted latent trajectories.
            z_goal: (B, D) breadcrumb latent (omit for pure exploration).

        Returns:
            (B,) cost per trajectory.
        """
        cost = self.safety_head.score_trajectory(z_seq)

        if self.goal_head is not None and z_goal is not None:
            cost = cost + self.goal_weight * self.goal_head.score_trajectory(z_seq, z_goal)

        if self.progress_head is not None and z_goal is not None and z_now is not None:
            cost = cost - self.progress_weight * self.progress_head.score_trajectory(
                z_seq, z_now, z_goal,
            )

        if self.exploration is not None:
            B, H, D = z_seq.shape
            bonus = self.exploration(z_seq.reshape(B * H, D)).reshape(B, H)
            cost = cost - self.exploration_weight * bonus.sum(dim=-1)

        return cost
