"""Transformer predictor with AdaLN-zero action conditioning.

Matches the official le-wm architecture (module.py):
  - Custom attention: dim_head=64, 16 heads -> 1024 total attention dim
  - Double LayerNorm: AdaLN (elementwise_affine=False) + internal affine LN
  - F.scaled_dot_product_attention with is_causal=True
  - Conv1d + MLP action encoder (Embedder)
  - 192-dim residual stream throughout (no up-projection)

Reference:
    Official le-wm repo: github.com/facebookresearch/le-wm
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------- #
# Action encoder
# ---------------------------------------------------------------------- #

class ActionEmbedder(nn.Module):
    """Matches the official le-wm ``Embedder`` class.

    Conv1d smoothing over the time axis followed by an MLP projection.
    """

    def __init__(
        self,
        input_dim: int = 3,
        smoothed_dim: int = 10,
        emb_dim: int = 192,
        mlp_scale: int = 4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x.float()
        x = x.permute(0, 2, 1)       # (B, D, T) for Conv1d
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)       # (B, T, D')
        x = self.embed(x)
        return x


# ---------------------------------------------------------------------- #
# Attention & FeedForward (with internal LayerNorm — double-norm pattern)
# ---------------------------------------------------------------------- #

class Attention(nn.Module):
    """Scaled dot-product attention with internal LayerNorm.

    Uses dim_head=64 so total attention dim = heads * dim_head (e.g. 1024).
    The QKV projection goes from ``dim`` -> ``inner_dim * 3``, and the output
    projection goes from ``inner_dim`` -> ``dim``, allowing a wider attention
    bottleneck than the residual stream.
    """

    def __init__(self, dim: int, heads: int = 16, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, T, D)"""
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0

        B, T, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)
            for t in qkv
        )  # (B, heads, T, dim_head)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, inner_dim)
        return self.to_out(out)


class FeedForward(nn.Module):
    """FeedForward with internal LayerNorm (double-norm pattern)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------- #
# ConditionalBlock (DiT-style AdaLN-zero)
# ---------------------------------------------------------------------- #

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.

    Double LayerNorm: the outer ``norm1``/``norm2`` (elementwise_affine=False)
    are modulated by AdaLN, and Attention/FeedForward have their own internal
    affine LayerNorm.  Zero-init on the modulation projection ensures the
    action conditioning is a no-op at the start of training.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-init so conditioning opens up gradually
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------- #
# TransformerPredictor  (official ARPredictor + Transformer, merged)
# ---------------------------------------------------------------------- #

class TransformerPredictor(nn.Module):
    """Autoregressive predictor matching the official le-wm architecture.

    192-dim residual stream, 16 heads x 64 dim_head = 1024 attention dim,
    2048-dim FFN, AdaLN-zero conditioning, causal attention.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        cmd_dim: int = 3,
        n_layers: int = 6,
        n_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        max_seq_len: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # Action encoder (official Embedder)
        self.action_embed = ActionEmbedder(
            input_dim=cmd_dim, smoothed_dim=10, emb_dim=latent_dim,
        )

        # Positional embedding (randn-initialized, matching official)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        self.input_drop = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ConditionalBlock(latent_dim, n_heads, dim_head, mlp_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm (matches official Transformer.norm)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z_seq: torch.Tensor,
        cmd_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-step embeddings.

        Args:
            z_seq:   (B, T, D) — latent observation embeddings.
            cmd_seq: (B, T, cmd_dim) — raw action/command sequence.

        Returns:
            (B, T, D) — predicted embeddings at each step.
        """
        batch, steps, _ = z_seq.shape
        if steps > self.max_seq_len:
            raise ValueError(
                f"Sequence length {steps} exceeds max_seq_len={self.max_seq_len}"
            )

        x = z_seq + self.pos_embed[:, :steps]
        x = self.input_drop(x)
        cond = self.action_embed(cmd_seq)

        for block in self.blocks:
            x = block(x, cond)

        x = self.norm(x)
        return x

    # ------------------------------------------------------------------ #
    # Rollout / inference helpers
    # ------------------------------------------------------------------ #

    def rollout(
        self,
        z_start: torch.Tensor,
        action_seq: torch.Tensor,
        z_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Auto-regressive rollout for latent planning.

        Args:
            z_start:    (B, D) — initial latent embedding.
            action_seq: (B, H, cmd_dim) — candidate action sequence.
            z_history: optional (B, C, D) history ending at ``z_start``.
            action_history: optional (B, C-1, cmd_dim) action history aligned
                with ``z_history`` source latents.

        Returns:
            (B, H, D) — predicted latents at each step.
        """
        batch, horizon, _ = action_seq.shape
        if z_history is not None:
            if z_history.ndim != 3:
                raise ValueError(
                    f"Expected z_history shape (B, C, D), got {tuple(z_history.shape)}",
                )
            if z_history.shape[0] != batch or z_history.shape[2] != z_start.shape[1]:
                raise ValueError(
                    f"Incompatible z_history shape {tuple(z_history.shape)} for "
                    f"z_start {tuple(z_start.shape)}",
                )
            z_tokens = z_history
            z_start_hist = z_history[:, -1, :]
            if not torch.allclose(z_start_hist, z_start, atol=1e-5, rtol=1e-4):
                raise ValueError("z_history must end at z_start.")
            if action_history is None:
                action_tokens = action_seq[:, :0, :]
            else:
                if action_history.ndim != 3:
                    raise ValueError(
                        "Expected action_history shape (B, C-1, cmd_dim), got "
                        f"{tuple(action_history.shape)}",
                    )
                if action_history.shape[0] != batch or action_history.shape[2] != action_seq.shape[2]:
                    raise ValueError(
                        f"Incompatible action_history shape {tuple(action_history.shape)} for "
                        f"action_seq {tuple(action_seq.shape)}",
                    )
                expected_hist = max(0, int(z_history.shape[1]) - 1)
                if int(action_history.shape[1]) != expected_hist:
                    raise ValueError(
                        f"action_history length {action_history.shape[1]} does not match "
                        f"z_history length {z_history.shape[1]}",
                    )
                action_tokens = action_history
        else:
            z_tokens = z_start.unsqueeze(1)
            action_tokens = action_seq[:, :0, :]
        preds = []

        for step in range(horizon):
            z_ctx = z_tokens
            a_ctx = torch.cat([action_tokens, action_seq[:, step : step + 1, :]], dim=1)
            # Sliding window: keep only the last max_seq_len tokens
            if z_ctx.shape[1] > self.max_seq_len:
                z_ctx = z_ctx[:, -self.max_seq_len:]
                a_ctx = a_ctx[:, -self.max_seq_len:]
            pred = self.forward(z_ctx, a_ctx)
            z_next = pred[:, -1, :]
            preds.append(z_next)
            z_tokens = torch.cat([z_tokens, z_next.unsqueeze(1)], dim=1)
            action_tokens = torch.cat([action_tokens, action_seq[:, step : step + 1, :]], dim=1)

        return torch.stack(preds, dim=1)

    def predict_step(
        self,
        z_history: torch.Tensor,
        cmd_history: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step prediction from history context."""
        return self.forward(z_history, cmd_history)[:, -1, :]
