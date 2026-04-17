#!/usr/bin/env python3
"""Direct test of whether the WM predictor propagates action differences into z_pred.

Loads the WM, pulls a single frame from an HDF5 chunk to get a real z_raw,
then runs a fixed library of extreme action sequences through plan_rollout
and reports pairwise L2 distances of the resulting terminal projected latents.

Interpretation:
    pairwise_mean / ||z||  < 0.005  → predictor ignores actions; retrain it
    pairwise_mean / ||z||  > 0.05   → predictor is action-sensitive; CEM is
                                      the issue, not the WM
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import numpy as np
import torch

from lewm.checkpoint_utils import clean_state_dict
from lewm.models import LeWorldModel


def _clean_load_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")


def load_wm(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = clean_state_dict(ckpt["model_state_dict"])
    pos_embed = state["encoder.vis_enc.pos_embed"]
    patch_w = state["encoder.vis_enc.patch_embed.weight"]
    pred_pos = state["predictor.pos_embed"]
    cmd_w = state["predictor.action_embed.patch_embed.weight"]
    latent_dim = int(pos_embed.shape[-1])
    patch_size = int(patch_w.shape[-1])
    n_tokens = int(pos_embed.shape[1] - 1)
    grid = int(round(math.sqrt(n_tokens)))
    image_size = grid * patch_size
    max_seq_len = int(pred_pos.shape[1])
    cmd_dim = int(cmd_w.shape[1])
    use_proprio = any(k.startswith("encoder.prop_enc.") for k in state)
    print(f"  image_size={image_size} patch={patch_size} max_seq={max_seq_len} "
          f"cmd_dim={cmd_dim} latent={latent_dim}")
    model = LeWorldModel(
        latent_dim=latent_dim,
        cmd_dim=cmd_dim,
        image_size=image_size,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        use_proprio=use_proprio,
    ).to(device)
    _clean_load_state(model, state)
    model.eval()
    return model, dict(
        latent_dim=latent_dim, image_size=image_size, patch_size=patch_size,
        cmd_dim=cmd_dim, use_proprio=use_proprio, max_seq_len=max_seq_len,
    )


def load_one_frame(h5_path: str, image_size: int, device: torch.device):
    with h5py.File(h5_path, "r") as f:
        # rgb is (N, 3, H, W) uint8 typically; probe a mid-episode frame
        rgb_key = "rgb" if "rgb" in f else ("images" if "images" in f else None)
        if rgb_key is None:
            # Try to find any likely RGB dataset
            for k in f.keys():
                if "rgb" in k.lower():
                    rgb_key = k
                    break
        if rgb_key is None:
            raise RuntimeError(f"No RGB dataset in {h5_path}; keys={sorted(f.keys())}")
        arr = f[rgb_key]
        idx = min(500, arr.shape[0] - 1)
        frame = np.asarray(arr[idx])   # (3, H, W) or (H, W, 3)
        print(f"  Using frame {idx} from {rgb_key}, shape={frame.shape}")
    if frame.ndim == 3 and frame.shape[0] != 3 and frame.shape[-1] == 3:
        frame = frame.transpose(2, 0, 1)
    vis = torch.from_numpy(frame).unsqueeze(0).to(device).float().div_(255.0)
    # center-crop / resize if needed (assume already image_size)
    if vis.shape[-1] != image_size or vis.shape[-2] != image_size:
        vis = torch.nn.functional.interpolate(
            vis, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return vis


def build_action_library(horizon: int, cmd_dim: int, device: torch.device):
    """Return (N, H, cmd_dim) of extreme, diverse action sequences."""
    block_size = cmd_dim // 3 if cmd_dim % 3 == 0 else 1
    def pattern(vx, vy, yaw):
        if block_size == 1:
            return [vx, vy, yaw]
        return [vx, vy, yaw] * block_size

    patterns = [
        pattern(+0.8, 0.0,  0.0),   # full forward
        pattern(-0.4, 0.0,  0.0),   # full backward
        pattern( 0.0, 0.0, +1.0),   # full yaw left
        pattern( 0.0, 0.0, -1.0),   # full yaw right
        pattern(+0.8, 0.0, +1.0),   # forward + left
        pattern(+0.8, 0.0, -1.0),   # forward + right
        pattern(-0.4, 0.0, +1.0),   # backward + left
        pattern(-0.4, 0.0, -1.0),   # backward + right
        pattern( 0.0, +0.3, 0.0),   # strafe left
        pattern( 0.0, -0.3, 0.0),   # strafe right
        pattern( 0.0, 0.0,  0.0),   # stop
        pattern(+0.4, +0.15, +0.5), # diagonal
    ]
    seqs = []
    for p in patterns:
        seqs.append([p] * horizon)  # hold same macro for whole horizon
    out = torch.tensor(seqs, dtype=torch.float32, device=device)
    return out  # (N, H, cmd_dim)


@torch.no_grad()
def probe(model, vis, action_lib, device):
    vis_enc = vis if vis.ndim == 4 else vis.unsqueeze(0)
    # proprio placeholder (zeros)
    prop = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "prop_enc"):
        prop_dim = None
        for p in model.encoder.prop_enc.parameters():
            if p.ndim >= 2:
                prop_dim = p.shape[-1]
                break
        if prop_dim is not None:
            prop = torch.zeros(1, prop_dim, device=device)

    z_raw, z_proj = model.encode(vis_enc, prop)
    print(f"  z_raw: shape={tuple(z_raw.shape)}  ||z_raw||={z_raw.norm(dim=-1).item():.3f}")
    print(f"  z_proj:  shape={tuple(z_proj.shape)}   ||z_proj||={z_proj.norm(dim=-1).item():.3f}")

    N = action_lib.shape[0]
    z_start_batch = z_raw.expand(N, -1).contiguous()
    z_pred_proj = model.plan_rollout(z_start_batch, action_lib)  # (N, H, D)
    z_terminal = z_pred_proj[:, -1, :]  # (N, D)

    norms = z_terminal.norm(dim=-1)
    print(f"\nTerminal latent norms per candidate: "
          f"mean={norms.mean().item():.3f} std={norms.std().item():.4f}")

    # pairwise L2
    diff = z_terminal.unsqueeze(0) - z_terminal.unsqueeze(1)  # (N, N, D)
    dist = diff.norm(dim=-1)  # (N, N)
    mask = ~torch.eye(N, dtype=torch.bool, device=dist.device)
    pairwise = dist[mask]

    mean_norm = norms.mean().item()
    pair_mean = pairwise.mean().item()
    pair_max = pairwise.max().item()
    pair_min = pairwise.min().item()
    ratio_mean = pair_mean / max(mean_norm, 1e-6)
    ratio_max = pair_max / max(mean_norm, 1e-6)

    print(f"\nPairwise terminal L2 distance across {N} candidates:")
    print(f"  mean = {pair_mean:.4f}")
    print(f"  min  = {pair_min:.4f}")
    print(f"  max  = {pair_max:.4f}")
    print(f"  mean / ||z|| = {ratio_mean:.5f}")
    print(f"  max  / ||z|| = {ratio_max:.5f}")

    # Also show distance to the "stop" candidate (index 10) — baseline for "did anything happen"
    stop_idx = 10  # the all-zero pattern, per build_action_library ordering
    dists_to_stop = dist[stop_idx]
    names = ["fwd", "back", "yawL", "yawR", "fwdL", "fwdR", "backL", "backR",
             "strafeL", "strafeR", "STOP", "diag"]
    print("\nDistance from STOP terminal to each candidate terminal:")
    for i, n in enumerate(names):
        marker = "  " if i != stop_idx else "  (self)"
        print(f"  {n:>8s}: {dists_to_stop[i].item():.4f}{marker}")

    if ratio_mean < 0.005:
        print("\n→ Predictor is action-insensitive. All candidates collapse to "
              "near-identical terminal latents. Retrain WM predictor with "
              "action-reconstruction aux loss.")
    elif ratio_mean < 0.02:
        print("\n→ Predictor weakly action-sensitive. Heads will struggle but "
              "might work with aggressive CEM diversity.")
    else:
        print("\n→ Predictor is action-sensitive. Problem is elsewhere (heads, "
              "BN, CEM convergence).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", type=str, required=True)
    p.add_argument("--h5", type=str, required=True,
                   help="Path to one HDF5 chunk (any chunk from the training set works)")
    p.add_argument("--horizon", type=int, default=5,
                   help="Plan horizon (macro steps) — match inference")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"Loading WM from {args.wm_ckpt}")
    model, meta = load_wm(args.wm_ckpt, device)

    print(f"Loading one frame from {args.h5}")
    vis = load_one_frame(args.h5, meta["image_size"], device)

    action_lib = build_action_library(args.horizon, meta["cmd_dim"], device)
    print(f"\nAction library: {action_lib.shape[0]} sequences × "
          f"{action_lib.shape[1]} steps × {action_lib.shape[2]} dims")

    probe(model, vis, action_lib, device)


if __name__ == "__main__":
    main()
