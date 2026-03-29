"""Vision encoder and BatchNorm projector for LeWorldModel."""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTBlock(nn.Module):
    """Standard pre-norm ViT block."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Paper-aligned ViT-Tiny observation encoder."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected vision tensor of shape (B, C, H, W), got {tuple(x.shape)}"
            )
        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} inputs, got {height}x{width}"
            )

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class ProprioEncoder(nn.Module):
    """Optional proprio extension for quadruped experiments."""

    def __init__(self, input_dim: int = 47, feature_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointEncoder(nn.Module):
    """Wrapper around the paper encoder.

    The paper uses only pixels. ``use_proprio`` remains available as a local
    extension, but is disabled by default so the backbone matches the paper.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        image_size: int = 224,
        patch_size: int = 14,
        use_proprio: bool = False,
        proprio_dim: int = 47,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.vis_enc = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
        )
        if use_proprio:
            self.prop_enc = ProprioEncoder(proprio_dim, latent_dim)
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.GELU(),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.prop_enc = None
            self.fusion = None

    def forward(
        self,
        vision: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_vis = self.vis_enc(vision)
        if not self.use_proprio:
            return z_vis
        if proprio is None:
            raise ValueError("proprio is required when use_proprio=True")
        z = torch.cat([z_vis, self.prop_enc(proprio)], dim=-1)
        return self.fusion(z)


class Projector(nn.Module):
    """2-layer MLP with BatchNorm for SIGReg compatibility.

    Matches the official le-wm repo: Linear→BN→GELU→Linear with 2048 hidden.
    """

    def __init__(self, in_dim: int = 192, out_dim: int = 192, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        return self.net(x.reshape(batch * steps, -1)).reshape(batch, steps, -1)
