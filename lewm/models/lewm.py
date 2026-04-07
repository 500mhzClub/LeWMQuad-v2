"""LeWorldModel — stable end-to-end JEPA from pixels.

Replaces the EMA student-teacher CanonicalJEPA with the LeWM approach:
  - Single encoder (no target encoder, no EMA, no stop-gradient).
  - Transformer predictor with AdaLN action conditioning.
  - Two-term loss: MSE prediction + λ·SIGReg anti-collapse.
  - BatchNorm projectors map encoder/predictor outputs to the space where
    the loss is computed.

Reference:
    Maes, Le Lidec, Scieur, LeCun, Balestriero.
    "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
     from Pixels", arXiv:2603.19312, Mar 2026.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .encoders import JointEncoder, Projector
from .predictor import TransformerPredictor
from .sigreg import sigreg_stepwise


class LeWorldModel(nn.Module):
    """End-to-end latent world model with SIGReg regularisation.

    Defaults follow the official le-wm repo: ViT-Tiny encoder, 192-dim
    predictor with 16×64 attention heads, BatchNorm projectors.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        cmd_dim: int = 3,
        pred_layers: int = 6,
        pred_heads: int = 16,
        pred_dim_head: int = 64,
        pred_mlp_dim: int = 2048,
        pred_dropout: float = 0.1,
        pred_emb_dropout: float = 0.0,
        max_seq_len: int = 4,
        sigreg_lambda: float = 0.09,
        sigreg_projections: int = 1024,
        sigreg_knots: int = 17,
        image_size: int = 224,
        patch_size: int = 14,
        use_proprio: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_projections = sigreg_projections
        self.sigreg_knots = sigreg_knots

        self.encoder = JointEncoder(
            latent_dim=latent_dim,
            image_size=image_size,
            patch_size=patch_size,
            use_proprio=use_proprio,
        )
        self.enc_projector = Projector(latent_dim, latent_dim)

        self.predictor = TransformerPredictor(
            latent_dim=latent_dim,
            cmd_dim=cmd_dim,
            n_layers=pred_layers,
            n_heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
            emb_dropout=pred_emb_dropout,
            max_seq_len=max_seq_len,
        )
        self.pred_projector = Projector(latent_dim, latent_dim)

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #

    def encode(
        self, vis: torch.Tensor, prop: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single frame.

        Returns:
            z_raw:  (B, D) backbone output  (for predictor input / rollout).
            z_proj: (B, D) projected output (for loss / goal matching).
        """
        z_raw = self.encoder(vis, prop)
        z_proj = self.enc_projector(z_raw)
        return z_raw, z_proj

    def encode_seq(
        self,
        vis_seq: torch.Tensor,
        prop_seq: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a temporal sequence of frames.

        Args:
            vis_seq:  (B, T, C, H, W)
            prop_seq: (B, T, P)

        Returns:
            z_raw:  (B, T, D)
            z_proj: (B, T, D)
        """
        B, T = vis_seq.shape[:2]
        vis_flat = vis_seq.reshape(B * T, *vis_seq.shape[2:])
        prop_flat = None
        if prop_seq is not None:
            prop_flat = prop_seq.reshape(B * T, *prop_seq.shape[2:])

        z_raw_flat = self.encoder(vis_flat, prop_flat)
        z_proj_flat = self.enc_projector(z_raw_flat)

        z_raw = z_raw_flat.reshape(B, T, -1)
        z_proj = z_proj_flat.reshape(B, T, -1)
        return z_raw, z_proj

    # ------------------------------------------------------------------ #
    # Training forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        vis_seq: torch.Tensor,
        prop_seq: torch.Tensor | None,
        cmd_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Full training forward pass.

        Args:
            vis_seq:  (B, T, 3, H, W) — uint8→float already handled.
            prop_seq: (B, T, P) or None.
            cmd_seq:  (B, T, 3)
            mask:     (B, T-1) bool — True for valid transitions.

        Returns:
            dict with keys:
                loss       — total loss  (pred + λ·sigreg).
                pred_loss  — MSE prediction loss.
                sigreg_loss — SIGReg regularisation.
                z_proj_std — per-dim std of projected embeddings (collapse check).
        """
        z_raw, z_proj = self.encode_seq(vis_seq, prop_seq)
        z_pred_raw = self.predictor(z_raw, cmd_seq)
        z_pred_proj = self.pred_projector.forward_seq(z_pred_raw)

        pred = z_pred_proj[:, :-1]
        target = z_proj[:, 1:]

        per_sample_mse = (pred - target).square().mean(dim=-1)

        if mask is not None:
            n_valid = mask.float().sum().clamp(min=1.0)
            pred_loss = (per_sample_mse * mask.float()).sum() / n_valid
        else:
            pred_loss = per_sample_mse.mean()

        sig_loss = sigreg_stepwise(
            z_proj,
            n_projections=self.sigreg_projections,
            n_knots=self.sigreg_knots,
        )

        total_loss = pred_loss + self.sigreg_lambda * sig_loss

        z_proj_std = z_proj.detach().float().std(dim=(0, 1)).mean()

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sig_loss,
            "z_proj_std": z_proj_std,
        }

    # ------------------------------------------------------------------ #
    # Planning helpers  (used by CEM / MPC)
    # ------------------------------------------------------------------ #

    def plan_rollout(
        self,
        z_start_raw: torch.Tensor,
        action_seq: torch.Tensor,
        z_history_raw: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rollout predictor auto-regressively for latent planning.

        Args:
            z_start_raw: (B, D) — raw encoder embedding of the current frame.
            action_seq:  (B, H, cmd_dim) — candidate action sequence.
            z_history_raw: optional (B, C, D) latent history ending at
                ``z_start_raw``.
            action_history: optional (B, C-1, cmd_dim) action history aligned
                with ``z_history_raw``.

        Returns:
            z_pred_proj: (B, H, D) — projected predicted latents at each step.
        """
        z_pred_raw = self.plan_rollout_raw(
            z_start_raw,
            action_seq,
            z_history_raw=z_history_raw,
            action_history=action_history,
        )
        z_pred_proj = self.pred_projector.forward_seq(z_pred_raw)
        return z_pred_proj

    def plan_rollout_raw(
        self,
        z_start_raw: torch.Tensor,
        action_seq: torch.Tensor,
        z_history_raw: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rollout predictor and return raw predicted latents."""
        return self.predictor.rollout(
            z_start_raw,
            action_seq,
            z_history=z_history_raw,
            action_history=action_history,
        )

    def plan_cost(
        self,
        z_pred_proj: torch.Tensor,
        z_goal_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Terminal goal-matching cost in projected space.

        Args:
            z_pred_proj: (B, H, D) or (B, D) — predicted latents.
            z_goal_proj: (B, D) — goal embedding (projected).

        Returns:
            (B,) — L2² cost.
        """
        if z_pred_proj.dim() == 3:
            z_pred_proj = z_pred_proj[:, -1, :]
        return (z_pred_proj - z_goal_proj).square().sum(dim=-1)

    # ------------------------------------------------------------------ #
    # Backward-compatible convenience methods
    # ------------------------------------------------------------------ #

    def encode_observation(
        self, vis: torch.Tensor, prop: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return projected embedding (for goal matching / energy head).

        Drop-in replacement for ``model.encode_target(vis, prop)`` in the
        old CanonicalJEPA API.
        """
        _, z_proj = self.encode(vis, prop)
        return z_proj

    def encode_raw(
        self, vis: torch.Tensor, prop: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return raw backbone embedding (for predictor input)."""
        return self.encoder(vis, prop)
