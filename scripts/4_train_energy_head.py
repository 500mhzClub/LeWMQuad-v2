#!/usr/bin/env python3
"""Train the LatentEnergyHead probe on frozen LeWM encoder latents.

Two-phase approach:
  Phase 1 — Extract: run the frozen encoder once over the full dataset,
            compute composite energy targets, cache (z_proj, target) to disk.
  Phase 2 — Train:   load cached latents, train the MLP head on pure tensors.
            No ViT in the loop → minutes per epoch instead of hours.

Usage:
    python scripts/4_train_energy_head.py \
        --data_dir jepa_final_dataset_224 \
        --checkpoint lewm_checkpoints/epoch_20.pt \
        --epochs 10 --batch_size 256 --lr 3e-4
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from lewm.models import LeWorldModel, LatentEnergyHead, composite_energy_target
from lewm.data import StreamingJEPADataset
from lewm.checkpoint_utils import clean_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LatentEnergyHead probe")
    p.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained LeWM checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for latent extraction (must match dataset).")
    p.add_argument("--train_batch_size", type=int, default=8192,
                   help="Batch size for MLP training on cached latents.")
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="energy_head_checkpoints")
    p.add_argument("--log_dir", type=str, default="energy_head_logs")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory for cached latents. Defaults to <out_dir>/latent_cache.")
    # Model dims (must match the checkpoint)
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--use_proprio", action="store_true")
    # Composite target weights
    p.add_argument("--w_safety", type=float, default=0.5)
    p.add_argument("--w_mobility", type=float, default=0.3)
    p.add_argument("--w_beacon", type=float, default=0.2)
    p.add_argument("--clearance_clip", type=float, default=1.0,
                   help="Saturate clearance beyond this distance (metres).")
    p.add_argument("--beacon_clip", type=float, default=5.0,
                   help="Saturate beacon_range beyond this distance (metres).")
    # Extraction only
    p.add_argument("--extract_only", action="store_true",
                   help="Only extract latents, skip training.")
    return p.parse_args()


def load_frozen_encoder(args, device):
    """Load the LeWM encoder from a checkpoint and freeze it."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = clean_state_dict(ckpt["model_state_dict"])

    image_size = args.image_size or 224
    patch_size = args.patch_size or (14 if image_size == 224 else 4)

    model = LeWorldModel(
        latent_dim=args.latent_dim,
        image_size=image_size,
        patch_size=patch_size,
        use_proprio=args.use_proprio,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = torch.compile(model)
    return model


# --------------------------------------------------------------------- #
# Phase 1: Extract latents + targets to disk
# --------------------------------------------------------------------- #

SHARD_SAMPLES = 1_000_000  # ~750 MB per shard at dim=192


def extract_latents(args, device) -> str:
    """Encode the full dataset once and cache (z_proj, target) shards to disk."""
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "latent_cache")
    manifest_path = os.path.join(cache_dir, "manifest.pt")

    if os.path.exists(manifest_path):
        info = torch.load(manifest_path, map_location="cpu")
        print(f"Latent cache found: {info['n_samples']:,} samples in {info['n_shards']} shards")
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)

    encoder = load_frozen_encoder(args, device)
    print(f"Loaded frozen encoder from {args.checkpoint}")

    num_workers = 12
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=num_workers,
        load_labels=True,
    )
    channels, height, width = dataset.vision_shape
    if args.image_size is None:
        args.image_size = height
    if args.patch_size is None:
        args.patch_size = 14 if height == 224 else 4

    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=True, prefetch_factor=2,
    )

    z_buf: list[torch.Tensor] = []
    t_buf: list[torch.Tensor] = []
    buf_samples = 0
    shard_idx = 0
    total_samples = 0

    def flush_shard():
        nonlocal z_buf, t_buf, buf_samples, shard_idx, total_samples
        if not z_buf:
            return
        shard_path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save({
            "z": torch.cat(z_buf),
            "target": torch.cat(t_buf),
        }, shard_path)
        total_samples += buf_samples
        print(f"  Shard {shard_idx}: {buf_samples:,} samples -> {shard_path}")
        z_buf, t_buf = [], []
        buf_samples = 0
        shard_idx += 1

    t0 = time.time()
    for batch in tqdm(dataloader, desc="Extracting latents"):
        vision, proprio, cmds, dones, collisions, labels = batch

        clearance = labels.get("clearance")
        traversability = labels.get("traversability")
        beacon_range = labels.get("beacon_range")
        if clearance is None or traversability is None:
            continue

        vision = vision.to(device, non_blocking=True).float().div_(255.0)
        proprio = proprio.to(device, non_blocking=True)

        B, T = vision.shape[:2]

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            _, z_proj = encoder.encode_seq(vision, proprio)

        z_flat = z_proj.reshape(B * T, -1).float().cpu()

        # Compute composite target on CPU
        beacon_range = beacon_range.float() if beacon_range is not None else torch.full_like(clearance, 999.0)
        target = composite_energy_target(
            clearance.float(), traversability, beacon_range,
            clearance_clip=args.clearance_clip,
            traversability_horizon=10,
            beacon_clip=args.beacon_clip,
            w_safety=args.w_safety,
            w_mobility=args.w_mobility,
            w_beacon=args.w_beacon,
        ).reshape(B * T)

        z_buf.append(z_flat)
        t_buf.append(target)
        buf_samples += B * T

        if buf_samples >= SHARD_SAMPLES:
            flush_shard()

    flush_shard()  # remaining

    elapsed = time.time() - t0
    torch.save({
        "n_samples": total_samples,
        "n_shards": shard_idx,
        "latent_dim": args.latent_dim,
    }, manifest_path)
    print(f"Extraction complete: {total_samples:,} samples, {shard_idx} shards, {elapsed:.0f}s")

    # Free encoder VRAM
    del encoder
    torch.cuda.empty_cache()

    return cache_dir


# --------------------------------------------------------------------- #
# Phase 2: Train MLP on cached latents
# --------------------------------------------------------------------- #

def load_cached_latents(cache_dir: str):
    """Load all shards into a single TensorDataset."""
    manifest = torch.load(os.path.join(cache_dir, "manifest.pt"), map_location="cpu")
    z_all, t_all = [], []
    for i in range(manifest["n_shards"]):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i:04d}.pt"), map_location="cpu")
        z_all.append(shard["z"])
        t_all.append(shard["target"])
        print(f"  Loaded shard {i}: {shard['z'].shape[0]:,} samples")
    z_cat = torch.cat(z_all)
    t_cat = torch.cat(t_all)
    print(f"Total: {z_cat.shape[0]:,} samples, z={z_cat.shape}, target={t_cat.shape}")
    return TensorDataset(z_cat, t_cat)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training LatentEnergyHead on {device}")
    print(f"Target weights: safety={args.w_safety}, mobility={args.w_mobility}, beacon={args.w_beacon}")

    # Phase 1: extract (or reuse cache)
    cache_dir = extract_latents(args, device)

    if args.extract_only:
        return

    # Phase 2: load cache and train
    print("\nLoading cached latents...")
    cached_dataset = load_cached_latents(cache_dir)

    dataloader = DataLoader(
        cached_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    n_batches = len(dataloader)
    print(f"Training dataloader: {n_batches} batches of {args.train_batch_size}")

    # Energy head
    head = LatentEnergyHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Energy head parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "energy_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "mean_energy", "lr"])

    global_step = 0

    for epoch in range(args.epochs):
        head.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for z_batch, target_batch in pbar:
            z_batch = z_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            energy = head(z_batch)
            loss = nn.functional.mse_loss(energy, target_batch)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=args.grad_clip,
            ).item()

            if not torch.isfinite(loss) or not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            global_step += 1
            loss_val = loss.item()
            mean_energy = energy.detach().mean().item()
            epoch_loss_sum += loss_val
            epoch_batches += 1

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    global_step, epoch + 1,
                    f"{loss_val:.6f}",
                    f"{mean_energy:.4f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                ])

            if global_step % 5 == 0:
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    energy=f"{mean_energy:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"energy_step_{global_step}.pt")
                torch.save({"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch}, ckpt_path)
                print(f"\n  Saved: {ckpt_path}")

        scheduler.step()
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1} complete | avg_loss={avg_loss:.4f} | time={elapsed:.0f}s")

        epoch_path = os.path.join(args.out_dir, f"energy_epoch_{epoch + 1}.pt")
        torch.save({"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch}, epoch_path)

    print("Energy head training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
