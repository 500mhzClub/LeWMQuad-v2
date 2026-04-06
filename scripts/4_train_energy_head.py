#!/usr/bin/env python3
"""Train planning heads on frozen LeWM encoder latents.

The canonical runtime stack is:
  - unconditional safety energy
  - identity-conditioned goal energy
  - optional RND exploration bonus for pure-perception search

An additional progress head can still be trained as an auxiliary probe, but it
is not required by the default inference script.

The learned components are trained on the same cached latent bank:

  Phase 1 — Extract:  run frozen encoder once, cache (z_proj, labels) to disk.
  Phase 2 — Safety:   LatentEnergyHead on composite safety+mobility target.
  Phase 3 — Goal:     GoalEnergyHead on identity-conditioned beacon pairs.
  Phase 4 — Progress: optional ProgressEnergyHead auxiliary probe.
  Phase 5 — Explore:  ExplorationBonus (RND) on the full latent set.

The default inference-time scorer is:

    cost = safety + α·goal − β·exploration

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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from lewm.models import (
    LeWorldModel,
    LatentEnergyHead,
    GoalEnergyHead,
    ProgressEnergyHead,
    ExplorationBonus,
    TrajectoryScorer,
    composite_safety_target,
    consequence_safety_target,
)
from lewm.data import StreamingJEPADataset
from lewm.checkpoint_utils import clean_state_dict


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train planning heads on frozen LeWM latents")
    # Data / encoder
    p.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained LeWM checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for latent extraction.")
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=12,
                   help="DataLoader worker count for streamed HDF5 batches.")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Number of batches prefetched per worker.")
    p.add_argument("--temporal_stride", type=int, default=1,
                   help="Raw-step spacing between model observations.")
    p.add_argument("--action_block_size", type=int, default=None,
                   help="Raw-step action-block size per model step. Defaults to --temporal_stride.")
    p.add_argument("--window_stride", type=int, default=None,
                   help="Raw-step spacing between extraction sequence starts. Defaults to seq_len * temporal_stride.")
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--use_proprio", action="store_true")
    # Shared
    p.add_argument("--out_dir", type=str, default="energy_head_checkpoints_keyframe_exec")
    p.add_argument("--log_dir", type=str, default="energy_head_logs_keyframe_exec")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory for cached latents. Defaults to a stride/block-specific cache under <out_dir>.")
    p.add_argument("--extract_only", action="store_true",
                   help="Only extract latents, skip training.")
    # Model dims (must match encoder checkpoint)
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # Safety head
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--safety_mode", type=str, default="consequence",
                   choices=["proximity", "consequence"],
                   help="'proximity' = legacy clearance-gradient target, "
                        "'consequence' = contact + mobility target.")
    p.add_argument("--w_safety", type=float, default=0.6)
    p.add_argument("--w_mobility", type=float, default=0.4)
    p.add_argument("--clearance_clip", type=float, default=1.0)
    p.add_argument("--w_contact", type=float, default=0.4,
                   help="Weight for contact term (consequence mode).")
    p.add_argument("--contact_clearance", type=float, default=0.08,
                   help="Clearance below which counts as physical contact (m).")
    # Goal head
    p.add_argument("--skip_goal", action="store_true",
                   help="Skip GoalEnergyHead training.")
    p.add_argument("--goal_epochs", type=int, default=10)
    p.add_argument("--goal_lr", type=float, default=3e-4)
    p.add_argument("--goal_batch_size", type=int, default=4096)
    p.add_argument("--beacon_clip", type=float, default=5.0)
    # Progress head
    p.add_argument("--skip_progress", action="store_true",
                   help="Skip ProgressEnergyHead training.")
    p.add_argument("--progress_epochs", type=int, default=5)
    p.add_argument("--progress_lr", type=float, default=3e-4)
    p.add_argument("--progress_batch_size", type=int, default=256,
                   help="Sequence micro-batch size for progress training.")
    p.add_argument("--progress_seq_len", type=int, default=4,
                   help="Sequence length for progress-head training.")
    p.add_argument("--progress_visible_bonus", type=float, default=0.35,
                   help="Target progress bonus when the target beacon becomes newly visible.")
    # Exploration bonus
    p.add_argument("--skip_exploration", action="store_true",
                   help="Skip ExplorationBonus (RND) training.")
    p.add_argument("--exploration_epochs", type=int, default=5)
    p.add_argument("--exploration_lr", type=float, default=1e-3)
    p.add_argument("--exploration_feature_dim", type=int, default=128)
    # TrajectoryScorer weights
    p.add_argument("--goal_weight", type=float, default=1.0)
    p.add_argument("--progress_weight", type=float, default=1.0)
    p.add_argument("--exploration_weight", type=float, default=0.1)
    p.add_argument("--no_progress", action="store_true",
                   help="Disable animated tqdm progress bars and emit plain logs only.")
    return p.parse_args()


# --------------------------------------------------------------------- #
# Encoder loading
# --------------------------------------------------------------------- #

def cli_flag_provided(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def infer_encoder_meta_from_checkpoint(args, device):
    """Infer the frozen encoder configuration directly from the LeWM checkpoint."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = clean_state_dict(ckpt["model_state_dict"])

    pos_embed = sd["encoder.vis_enc.pos_embed"]
    patch_w = sd["encoder.vis_enc.patch_embed.weight"]
    pred_pos = sd["predictor.pos_embed"]
    cmd_w = sd["predictor.action_embed.patch_embed.weight"]
    latent_dim = int(pos_embed.shape[-1])
    patch_size = int(patch_w.shape[-1])
    n_tokens = int(pos_embed.shape[1] - 1)
    grid = int(round(math.sqrt(n_tokens)))
    image_size = grid * patch_size
    max_seq_len = int(pred_pos.shape[1])
    cmd_dim = int(cmd_w.shape[1])
    use_proprio = any(k.startswith("encoder.prop_enc.") for k in sd)
    command_representation = ckpt.get(
        "command_representation",
        "mean_scaled" if cmd_dim == 3 else "active_block",
    )
    command_latency = int(ckpt.get("command_latency", 2))

    inferred = {
        "latent_dim": latent_dim,
        "image_size": image_size,
        "patch_size": patch_size,
        "max_seq_len": max_seq_len,
        "cmd_dim": cmd_dim,
        "command_representation": command_representation,
        "command_latency": command_latency,
        "use_proprio": use_proprio,
    }
    return sd, inferred


def resolve_encoder_config(args, device):
    """Resolve encoder config from checkpoint and fail on explicit mismatches."""
    sd, inferred = infer_encoder_meta_from_checkpoint(args, device)

    if cli_flag_provided("--image_size") and args.image_size is not None and args.image_size != inferred["image_size"]:
        raise ValueError(
            f"--image_size={args.image_size} does not match checkpoint image_size={inferred['image_size']}"
        )
    if cli_flag_provided("--patch_size") and args.patch_size is not None and args.patch_size != inferred["patch_size"]:
        raise ValueError(
            f"--patch_size={args.patch_size} does not match checkpoint patch_size={inferred['patch_size']}"
        )
    if cli_flag_provided("--latent_dim") and args.latent_dim != inferred["latent_dim"]:
        raise ValueError(
            f"--latent_dim={args.latent_dim} does not match checkpoint latent_dim={inferred['latent_dim']}"
        )
    if cli_flag_provided("--use_proprio") and bool(args.use_proprio) != bool(inferred["use_proprio"]):
        raise ValueError(
            f"--use_proprio={args.use_proprio} does not match checkpoint use_proprio={inferred['use_proprio']}"
        )
    if cli_flag_provided("--seq_len") and int(args.seq_len) != int(inferred["max_seq_len"]):
        raise ValueError(
            f"--seq_len={args.seq_len} does not match checkpoint max_seq_len={inferred['max_seq_len']}"
        )

    args.image_size = inferred["image_size"]
    args.patch_size = inferred["patch_size"]
    args.latent_dim = inferred["latent_dim"]
    args.use_proprio = inferred["use_proprio"]
    args.encoder_max_seq_len = inferred["max_seq_len"]
    return sd, inferred


def load_frozen_encoder(args, device):
    """Load the LeWM encoder from a checkpoint and freeze it."""
    sd, inferred = resolve_encoder_config(args, device)

    model = LeWorldModel(
        latent_dim=inferred["latent_dim"],
        cmd_dim=inferred["cmd_dim"],
        image_size=inferred["image_size"],
        patch_size=inferred["patch_size"],
        max_seq_len=inferred["max_seq_len"],
        use_proprio=inferred["use_proprio"],
    )
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = torch.compile(model)
    return model


# --------------------------------------------------------------------- #
# Phase 1: Extract latents + labels to disk
# --------------------------------------------------------------------- #

SHARD_SAMPLES = 1_000_000  # ~750 MB per shard at dim=192
CACHE_VERSION = 5          # v5: temporal abstraction + encoder metadata validation


def progress_enabled(args: argparse.Namespace) -> bool:
    """Use animated tqdm bars only when stderr looks like a real terminal."""
    term = os.environ.get("TERM", "")
    return not args.no_progress and sys.stderr.isatty() and term.lower() != "dumb"


def make_progress(args: argparse.Namespace, iterable, *, desc: str):
    return tqdm(
        iterable,
        desc=desc,
        disable=not progress_enabled(args),
        dynamic_ncols=True,
        mininterval=1.0,
    )


def progress_write(message: str, pbar=None) -> None:
    if pbar is None:
        print(message)
        return
    pbar.write(message)


def extract_latents(args, device) -> str:
    """Encode the full dataset once and cache (z_proj, safety_target,
    beacon_identity, beacon_range) shards to disk."""
    _, encoder_meta = resolve_encoder_config(args, device)
    action_block_size = (
        args.action_block_size if args.action_block_size is not None else args.temporal_stride
    )
    window_stride = (
        args.window_stride
        if args.window_stride is not None
        else args.seq_len * args.temporal_stride
    )
    cache_name = (
        f"latent_cache_seq{args.seq_len}_stride{args.temporal_stride}"
        f"_block{action_block_size}_window{window_stride}"
    )
    cache_dir = args.cache_dir or os.path.join(args.out_dir, cache_name)
    manifest_path = os.path.join(cache_dir, "manifest.pt")

    if os.path.exists(manifest_path):
        info = torch.load(manifest_path, map_location="cpu")
        expected_cfg = {
            "seq_len": int(args.seq_len),
            "temporal_stride": int(args.temporal_stride),
            "action_block_size": int(action_block_size),
            "window_stride": int(window_stride),
        }
        expected_encoder_cfg = {
            "latent_dim": int(encoder_meta["latent_dim"]),
            "image_size": int(encoder_meta["image_size"]),
            "patch_size": int(encoder_meta["patch_size"]),
            "max_seq_len": int(encoder_meta["max_seq_len"]),
            "cmd_dim": int(encoder_meta["cmd_dim"]),
            "command_representation": str(encoder_meta["command_representation"]),
            "command_latency": int(encoder_meta["command_latency"]),
            "use_proprio": bool(encoder_meta["use_proprio"]),
        }
        if (
            info.get("version", 1) >= CACHE_VERSION
            and info.get("temporal_cfg") == expected_cfg
            and info.get("encoder_cfg") == expected_encoder_cfg
        ):
            print(f"Latent cache found: {info['n_samples']:,} samples in "
                  f"{info['n_shards']} shards (v{info.get('version', 1)})")
            return cache_dir
        print("Cache version/config mismatch — re-extracting.")

    os.makedirs(cache_dir, exist_ok=True)

    encoder = load_frozen_encoder(args, device)
    print(f"Loaded frozen encoder from {args.checkpoint}")

    num_workers = max(1, int(args.num_workers))
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        temporal_stride=args.temporal_stride,
        action_block_size=args.action_block_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=num_workers,
        load_labels=True,
    )
    channels, height, width = dataset.vision_shape
    if int(height) != int(args.image_size):
        raise ValueError(
            f"Dataset image_size={height} does not match checkpoint image_size={args.image_size}"
        )

    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=True, prefetch_factor=max(1, int(args.prefetch_factor)),
    )

    z_buf: list[torch.Tensor] = []
    st_buf: list[torch.Tensor] = []
    bid_buf: list[torch.Tensor] = []
    br_buf: list[torch.Tensor] = []
    coll_buf: list[torch.Tensor] = []
    buf_samples = 0
    shard_idx = 0
    total_samples = 0
    pbar = None

    def flush_shard():
        nonlocal z_buf, st_buf, bid_buf, br_buf, coll_buf, buf_samples, shard_idx, total_samples
        if not z_buf:
            return
        shard_path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save({
            "z": torch.cat(z_buf),
            "safety_target": torch.cat(st_buf),
            "beacon_identity": torch.cat(bid_buf),
            "beacon_range": torch.cat(br_buf),
            "collisions": torch.cat(coll_buf),
        }, shard_path)
        total_samples += buf_samples
        progress_write(f"  Shard {shard_idx}: {buf_samples:,} samples -> {shard_path}", pbar)
        z_buf, st_buf, bid_buf, br_buf, coll_buf = [], [], [], [], []
        buf_samples = 0
        shard_idx += 1

    t0 = time.time()
    with make_progress(args, dataloader, desc="Extracting latents") as pbar:
        for batch in pbar:
            vision, proprio, cmds, dones, collisions, labels = batch

            clearance = labels.get("clearance")
            traversability = labels.get("traversability")
            if clearance is None or traversability is None:
                continue

            beacon_range = labels.get("beacon_range")
            beacon_identity = labels.get("beacon_identity")

            vision = vision.to(device, non_blocking=True).float().div_(255.0)
            proprio = proprio.to(device, non_blocking=True)

            B, T = vision.shape[:2]

            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                _, z_proj = encoder.encode_seq(vision, proprio)

            z_flat = z_proj.reshape(B * T, -1).float().cpu()

            # Flatten collisions for this batch
            coll_flat = collisions.float().reshape(B * T)

            # Safety target — mode selects proximity vs consequence
            if args.safety_mode == "consequence":
                safety_t = consequence_safety_target(
                    clearance.float().reshape(B * T),
                    traversability.reshape(B * T),
                    coll_flat,
                    traversability_horizon=10,
                    contact_clearance=args.contact_clearance,
                    w_contact=args.w_contact,
                    w_mobility=args.w_mobility,
                )
            else:
                safety_t = composite_safety_target(
                    clearance.float(), traversability,
                    clearance_clip=args.clearance_clip,
                    traversability_horizon=10,
                    w_safety=args.w_safety,
                    w_mobility=args.w_mobility,
                ).reshape(B * T)

            # Beacon labels (flattened)
            if beacon_range is not None:
                br_flat = beacon_range.float().reshape(B * T)
            else:
                br_flat = torch.full((B * T,), 999.0)
            if beacon_identity is not None:
                bid_flat = beacon_identity.long().reshape(B * T)
            else:
                bid_flat = torch.full((B * T,), -1, dtype=torch.long)

            z_buf.append(z_flat)
            st_buf.append(safety_t)
            bid_buf.append(bid_flat)
            br_buf.append(br_flat)
            coll_buf.append(coll_flat)
            buf_samples += B * T

            if buf_samples >= SHARD_SAMPLES:
                flush_shard()

    pbar = None
    flush_shard()

    elapsed = time.time() - t0
    torch.save({
        "n_samples": total_samples,
        "n_shards": shard_idx,
        "latent_dim": args.latent_dim,
        "version": CACHE_VERSION,
        "encoder_cfg": {
            "latent_dim": int(encoder_meta["latent_dim"]),
            "image_size": int(encoder_meta["image_size"]),
            "patch_size": int(encoder_meta["patch_size"]),
            "max_seq_len": int(encoder_meta["max_seq_len"]),
            "cmd_dim": int(encoder_meta["cmd_dim"]),
            "command_representation": str(encoder_meta["command_representation"]),
            "command_latency": int(encoder_meta["command_latency"]),
            "use_proprio": bool(encoder_meta["use_proprio"]),
        },
        "temporal_cfg": {
            "seq_len": int(args.seq_len),
            "temporal_stride": int(args.temporal_stride),
            "action_block_size": int(action_block_size),
            "window_stride": int(window_stride),
        },
    }, manifest_path)
    print(f"Extraction complete: {total_samples:,} samples, "
          f"{shard_idx} shards, {elapsed:.0f}s")

    del encoder
    torch.cuda.empty_cache()
    return cache_dir


def load_cached_latents(cache_dir: str):
    """Load all shards and return (z, safety_target, beacon_identity, beacon_range)."""
    manifest = torch.load(os.path.join(cache_dir, "manifest.pt"), map_location="cpu")
    z_all, st_all, bid_all, br_all = [], [], [], []
    for i in range(manifest["n_shards"]):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i:04d}.pt"), map_location="cpu")
        z_all.append(shard["z"])
        st_all.append(shard["safety_target"])
        bid_all.append(shard["beacon_identity"])
        br_all.append(shard["beacon_range"])
        print(f"  Loaded shard {i}: {shard['z'].shape[0]:,} samples")
    z = torch.cat(z_all)
    st = torch.cat(st_all)
    bid = torch.cat(bid_all)
    br = torch.cat(br_all)
    print(f"Total: {z.shape[0]:,} samples, z={z.shape}")
    # Report collision stats if available in v3 shards
    n_contact = (st > 0.9).sum().item()
    print(f"  High-energy samples (>0.9): {n_contact:,} ({100*n_contact/len(st):.1f}%)")
    return z, st, bid, br


def build_goal_latent_pools(
    z_all: torch.Tensor,
    beacon_identity: torch.Tensor,
    beacon_range: torch.Tensor,
    beacon_clip: float,
) -> dict[int, torch.Tensor]:
    """Collect close-range goal examples for each beacon identity."""
    pools: dict[int, list[int]] = {}
    close_thresh = max(0.75, 0.35 * float(beacon_clip))
    for idx in range(len(z_all)):
        bid = int(beacon_identity[idx].item())
        br = float(beacon_range[idx].item())
        if bid < 0 or br >= close_thresh:
            continue
        pools.setdefault(bid, []).append(idx)

    if not pools:
        for idx in range(len(z_all)):
            bid = int(beacon_identity[idx].item())
            br = float(beacon_range[idx].item())
            if bid < 0 or br >= float(beacon_clip) * 2.0:
                continue
            pools.setdefault(bid, []).append(idx)

    goal_pools = {
        bid: z_all[torch.tensor(indices, dtype=torch.long)]
        for bid, indices in pools.items()
        if indices
    }
    return goal_pools


def sample_progress_batch(
    z_now: torch.Tensor,
    z_future: torch.Tensor,
    bid_now: torch.Tensor,
    br_now: torch.Tensor,
    bid_future: torch.Tensor,
    br_future: torch.Tensor,
    goal_pools: dict[int, torch.Tensor],
    beacon_clip: float,
    visible_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct goal latents and scalar progress targets for one minibatch."""
    valid_ids = list(goal_pools.keys())
    if not valid_ids:
        raise ValueError("No goal pools available for progress training.")

    n_pairs = z_now.shape[0]
    device = z_now.device
    dtype = z_now.dtype
    z_goal = torch.empty_like(z_now)
    targets = torch.zeros(n_pairs, device=device, dtype=dtype)

    for idx in range(n_pairs):
        cur_bid = int(bid_now[idx].item())
        nxt_bid = int(bid_future[idx].item())
        cur_visible = cur_bid >= 0 and float(br_now[idx].item()) < beacon_clip * 2.0
        nxt_visible = nxt_bid >= 0 and float(br_future[idx].item()) < beacon_clip * 2.0

        positive_ids: list[int] = []
        if cur_visible:
            positive_ids.append(cur_bid)
        if nxt_visible and nxt_bid not in positive_ids:
            positive_ids.append(nxt_bid)

        choose_positive = positive_ids and torch.rand((), device=device).item() < 0.6
        if choose_positive:
            goal_id = positive_ids[int(torch.randint(len(positive_ids), (1,), device=device).item())]
        else:
            goal_id = valid_ids[int(torch.randint(len(valid_ids), (1,), device=device).item())]

        pool = goal_pools[goal_id]
        pool_idx = int(torch.randint(len(pool), (1,), device=device).item())
        z_goal[idx] = pool[pool_idx].to(device=device, dtype=dtype)

        progress = 0.0
        cur_match = cur_visible and cur_bid == goal_id
        nxt_match = nxt_visible and nxt_bid == goal_id
        if cur_match and nxt_match:
            delta = (float(br_now[idx].item()) - float(br_future[idx].item())) / max(1e-6, beacon_clip)
            progress = max(0.0, min(1.0, delta))
        elif (not cur_match) and nxt_match:
            progress = float(visible_bonus)
        targets[idx] = progress

    return z_goal, targets


# --------------------------------------------------------------------- #
# Phase 2: Train LatentEnergyHead (safety + mobility)
# --------------------------------------------------------------------- #

def train_safety_head(args, z_all, safety_target, device):
    print("\n" + "=" * 60)
    print("Phase 2: Training LatentEnergyHead (safety + mobility)")
    print("=" * 60)

    dataset = TensorDataset(z_all, safety_target)
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"  {len(dataloader)} batches of {args.train_batch_size}")

    head = LatentEnergyHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "safety_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "mean_energy", "lr"])

    global_step = 0
    for epoch in range(args.epochs):
        head.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()

        with make_progress(args, dataloader, desc=f"  Safety {epoch + 1}/{args.epochs}") as pbar:
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
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1, f"{loss_val:.6f}",
                        f"{energy.detach().mean().item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.out_dir, f"safety_step_{global_step}.pt")
                    torch.save({"head_state_dict": head.state_dict(), "step": global_step}, ckpt_path)
                    progress_write(f"    Saved: {ckpt_path}", pbar)

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.4f} | time={time.time() - t0:.0f}s")

        torch.save(
            {"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch},
            os.path.join(args.out_dir, f"safety_epoch_{epoch + 1}.pt"),
        )

    print("  Safety head training complete.")
    return head


# --------------------------------------------------------------------- #
# Phase 3: Train GoalEnergyHead (identity-conditioned beacon pairs)
# --------------------------------------------------------------------- #

class GoalPairedDataset(Dataset):
    """Builds (z_anchor, z_goal, target) triples on-the-fly.

    For each sample:
      1. Pick a random target beacon identity from those present in the data.
      2. Sample a z_goal from the pool of latents where that identity is visible.
      3. Compute target:
           - If anchor sees the same identity: target ∝ range (low = close).
           - Otherwise: target = 1 (high energy).

    This teaches the GoalEnergyHead to discriminate which beacon
    the breadcrumb refers to and to produce low energy only when
    approaching that specific beacon.
    """

    def __init__(self, z_all, beacon_identity, beacon_range, beacon_clip=5.0):
        self.z = z_all
        self.identity = beacon_identity.long()
        self.range = beacon_range
        self.beacon_clip = beacon_clip

        # Build index: identity_id -> sample indices where that beacon is visible
        self.pools: dict[int, torch.Tensor] = {}
        pool_lists: dict[int, list[int]] = {}
        for i in range(len(beacon_identity)):
            bid = beacon_identity[i].item()
            if bid >= 0 and beacon_range[i].item() < beacon_clip * 2:
                pool_lists.setdefault(bid, []).append(i)
        for k, v in pool_lists.items():
            self.pools[k] = torch.tensor(v, dtype=torch.long)

        self.valid_ids = list(self.pools.keys())
        if not self.valid_ids:
            raise ValueError(
                "No beacon-visible samples found in the dataset. "
                "Cannot train GoalEnergyHead — use --skip_goal."
            )
        print(f"    GoalPairedDataset: {len(self.valid_ids)} beacon identities, "
              f"pools: {', '.join(f'{k}:{len(v)}' for k, v in pool_lists.items())}")

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        z_anchor = self.z[idx]

        # Random target identity
        tid = self.valid_ids[torch.randint(len(self.valid_ids), (1,)).item()]

        # Sample z_goal from that identity's pool
        pool = self.pools[tid]
        goal_idx = pool[torch.randint(len(pool), (1,)).item()].item()
        z_goal = self.z[goal_idx]

        # Target: low when anchor sees matching beacon, high otherwise
        anchor_bid = self.identity[idx].item()
        anchor_range = self.range[idx].item()

        if anchor_bid == tid and anchor_range < self.beacon_clip * 2:
            target = min(anchor_range / self.beacon_clip, 1.0)
        else:
            target = 1.0

        return z_anchor, z_goal, torch.tensor(target, dtype=z_anchor.dtype)


def train_goal_head(args, z_all, beacon_identity, beacon_range, device):
    print("\n" + "=" * 60)
    print("Phase 3: Training GoalEnergyHead (beacon identity matching)")
    print("=" * 60)

    try:
        dataset = GoalPairedDataset(z_all, beacon_identity, beacon_range, args.beacon_clip)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return None

    dataloader = DataLoader(
        dataset, batch_size=args.goal_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"  {len(dataloader)} batches of {args.goal_batch_size}")

    head = GoalEnergyHead(
        latent_dim=args.latent_dim, dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.goal_lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.goal_epochs)

    csv_path = os.path.join(args.log_dir, "goal_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "mean_energy", "lr"])

    global_step = 0
    for epoch in range(args.goal_epochs):
        head.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()

        with make_progress(args, dataloader, desc=f"  Goal {epoch + 1}/{args.goal_epochs}") as pbar:
            for z_a, z_g, target in pbar:
                z_a = z_a.to(device, non_blocking=True)
                z_g = z_g.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                energy = head(z_a, z_g)
                loss = nn.functional.mse_loss(energy, target)
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
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1, f"{loss_val:.6f}",
                        f"{energy.detach().mean().item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.out_dir, f"goal_step_{global_step}.pt")
                    torch.save({"head_state_dict": head.state_dict(), "step": global_step}, ckpt_path)
                    progress_write(f"    Saved: {ckpt_path}", pbar)

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.4f} | time={time.time() - t0:.0f}s")

        torch.save(
            {"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch},
            os.path.join(args.out_dir, f"goal_epoch_{epoch + 1}.pt"),
        )

    print("  Goal head training complete.")
    return head


# --------------------------------------------------------------------- #
# Phase 4: Train ProgressEnergyHead
# --------------------------------------------------------------------- #

def train_progress_head(args, goal_pools, device):
    print("\n" + "=" * 60)
    print("Phase 4: Training ProgressEnergyHead (short-horizon goal progress)")
    print("=" * 60)

    if not goal_pools:
        print("  Skipping: no goal pools with visible beacons were found.")
        return None

    encoder = load_frozen_encoder(args, device)
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=max(2, args.progress_seq_len),
        temporal_stride=args.temporal_stride,
        action_block_size=args.action_block_size,
        window_stride=args.window_stride,
        batch_size=args.progress_batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=max(1, int(args.num_workers)),
        load_labels=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=max(1, int(args.num_workers)),
        pin_memory=True, prefetch_factor=max(1, int(args.prefetch_factor)),
    )

    head = ProgressEnergyHead(
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.progress_lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.progress_epochs)

    csv_path = os.path.join(args.log_dir, "progress_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "mean_bonus", "mean_target", "lr"])

    global_step = 0
    for epoch in range(args.progress_epochs):
        head.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()

        with make_progress(args, dataloader, desc=f"  Progress {epoch + 1}/{args.progress_epochs}") as pbar:
            for batch in pbar:
                vision, proprio, _cmds, _dones, _collisions, labels = batch
                if "beacon_identity" not in labels or "beacon_range" not in labels:
                    continue

                vision = vision.to(device, non_blocking=True).float().div_(255.0)
                proprio = proprio.to(device, non_blocking=True)
                beacon_identity = labels["beacon_identity"].to(device, non_blocking=True).long()
                beacon_range = labels["beacon_range"].to(device, non_blocking=True).float()

                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    _, z_proj = encoder.encode_seq(vision, proprio)
                z_proj = z_proj.float()

                z_now = z_proj[:, :-1, :].reshape(-1, z_proj.shape[-1])
                z_future = z_proj[:, 1:, :].reshape(-1, z_proj.shape[-1])
                bid_now = beacon_identity[:, :-1].reshape(-1)
                bid_future = beacon_identity[:, 1:].reshape(-1)
                br_now = beacon_range[:, :-1].reshape(-1)
                br_future = beacon_range[:, 1:].reshape(-1)

                try:
                    z_goal, target = sample_progress_batch(
                        z_now,
                        z_future,
                        bid_now,
                        br_now,
                        bid_future,
                        br_future,
                        goal_pools,
                        beacon_clip=args.beacon_clip,
                        visible_bonus=args.progress_visible_bonus,
                    )
                except ValueError:
                    continue

                optimizer.zero_grad(set_to_none=True)
                pred = head(z_now, z_future, z_goal)
                loss = nn.functional.mse_loss(pred, target)
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
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1, f"{loss_val:.6f}",
                        f"{pred.detach().mean().item():.4f}",
                        f"{target.detach().mean().item():.4f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        bonus=f"{pred.detach().mean().item():.3f}",
                        target=f"{target.detach().mean().item():.3f}",
                    )

                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.out_dir, f"progress_step_{global_step}.pt")
                    torch.save({"head_state_dict": head.state_dict(), "step": global_step}, ckpt_path)
                    progress_write(f"    Saved: {ckpt_path}", pbar)

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.4f} | time={time.time() - t0:.0f}s")

        torch.save(
            {"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch},
            os.path.join(args.out_dir, f"progress_epoch_{epoch + 1}.pt"),
        )

    print("  Progress head training complete.")
    del encoder
    torch.cuda.empty_cache()
    return head


# --------------------------------------------------------------------- #
# Phase 5: Train ExplorationBonus (RND)
# --------------------------------------------------------------------- #

def train_exploration(args, z_all, device):
    print("\n" + "=" * 60)
    print("Phase 5: Training ExplorationBonus (RND)")
    print("=" * 60)

    dataset = TensorDataset(z_all)
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"  {len(dataloader)} batches of {args.train_batch_size}")

    bonus = ExplorationBonus(
        latent_dim=args.latent_dim,
        feature_dim=args.exploration_feature_dim,
    ).to(device)
    bonus = torch.compile(bonus)

    n_trainable = sum(p.numel() for p in bonus.predictor.parameters())
    print(f"  Predictor parameters: {n_trainable:,} (target is frozen)")

    optimizer = torch.optim.AdamW(
        bonus.predictor.parameters(), lr=args.exploration_lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.exploration_epochs)

    csv_path = os.path.join(args.log_dir, "exploration_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr"])

    global_step = 0
    for epoch in range(args.exploration_epochs):
        bonus.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()

        with make_progress(args, dataloader, desc=f"  RND {epoch + 1}/{args.exploration_epochs}") as pbar:
            for (z_batch,) in pbar:
                z_batch = z_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = bonus.loss(z_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(bonus.predictor.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1
                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1, f"{loss_val:.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(loss=f"{loss_val:.6f}")

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.6f} | time={time.time() - t0:.0f}s")

    torch.save(
        {"bonus_state_dict": bonus.state_dict(), "epoch": args.exploration_epochs},
        os.path.join(args.out_dir, "exploration_bonus.pt"),
    )
    print("  RND training complete.")
    return bonus


# --------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------- #

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _, encoder_meta = resolve_encoder_config(args, device)

    print(f"Training planning heads on {device}")
    print(f"Safety mode: {args.safety_mode}")
    print(
        "Frozen encoder checkpoint: "
        f"latent_dim={encoder_meta['latent_dim']} "
        f"image_size={encoder_meta['image_size']} "
        f"patch_size={encoder_meta['patch_size']} "
        f"seq_len={encoder_meta['max_seq_len']} "
        f"cmd_dim={encoder_meta['cmd_dim']} "
        f"cmd_repr={encoder_meta['command_representation']} "
        f"use_proprio={encoder_meta['use_proprio']}"
    )
    action_block_size = (
        args.action_block_size if args.action_block_size is not None else args.temporal_stride
    )
    window_stride = (
        args.window_stride
        if args.window_stride is not None
        else args.seq_len * args.temporal_stride
    )
    print(
        "Temporal abstraction: "
        f"seq_len={args.seq_len}, stride={args.temporal_stride}, "
        f"action_block={action_block_size}, window_stride={window_stride}"
    )
    if args.safety_mode == "consequence":
        print(f"  w_contact={args.w_contact}, w_mobility={args.w_mobility}, "
              f"contact_clearance={args.contact_clearance}m")
    else:
        print(f"  w_safety={args.w_safety}, w_mobility={args.w_mobility}")
    print(
        f"Goal weight: {args.goal_weight}, Progress weight: {args.progress_weight}, "
        f"Exploration weight: {args.exploration_weight}"
    )

    # Phase 1: extract (or reuse cache)
    cache_dir = extract_latents(args, device)
    if args.extract_only:
        return

    # Load all cached latents
    print("\nLoading cached latents...")
    z_all, safety_target, beacon_id, beacon_range = load_cached_latents(cache_dir)

    # Phase 2: safety head
    safety_head = train_safety_head(args, z_all, safety_target, device)

    # Phase 3: goal head
    goal_head = None
    if not args.skip_goal:
        goal_head = train_goal_head(args, z_all, beacon_id, beacon_range, device)

    # Phase 4: progress head
    progress_head = None
    if not args.skip_progress:
        goal_pools = build_goal_latent_pools(z_all, beacon_id, beacon_range, args.beacon_clip)
        progress_head = train_progress_head(args, goal_pools, device)

    # Phase 5: exploration bonus
    exploration = None
    if not args.skip_exploration:
        exploration = train_exploration(args, z_all, device)

    # Save combined TrajectoryScorer checkpoint
    scorer_path = os.path.join(args.out_dir, "trajectory_scorer.pt")
    scorer_data = {
        "safety_head": safety_head.state_dict(),
        "goal_head": goal_head.state_dict() if goal_head is not None else None,
        "progress_head": progress_head.state_dict() if progress_head is not None else None,
        "exploration": exploration.state_dict() if exploration is not None else None,
        "goal_weight": args.goal_weight,
        "progress_weight": args.progress_weight,
        "exploration_weight": args.exploration_weight,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "exploration_feature_dim": args.exploration_feature_dim,
        "safety_mode": args.safety_mode,
        "seq_len": args.seq_len,
        "temporal_stride": args.temporal_stride,
        "action_block_size": action_block_size,
        "window_stride": window_stride,
        "image_size": int(encoder_meta["image_size"]),
        "patch_size": int(encoder_meta["patch_size"]),
        "cmd_dim": int(encoder_meta["cmd_dim"]),
        "command_representation": str(encoder_meta["command_representation"]),
        "command_latency": int(encoder_meta["command_latency"]),
        "use_proprio": bool(encoder_meta["use_proprio"]),
        "max_seq_len": int(encoder_meta["max_seq_len"]),
    }
    torch.save(scorer_data, scorer_path)
    print(f"\nTrajectoryScorer checkpoint saved: {scorer_path}")
    print("All training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
