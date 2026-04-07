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
    PlaceSnippetHead,
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
    p.add_argument("--raw_data_dir", type=str, default=None,
                   help="Optional directory with raw chunk_*.npz rollouts for pose-supervised place training.")
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
    # Place head
    p.add_argument("--skip_place", action="store_true",
                   help="Skip rollout-snippet place embedding training.")
    p.add_argument("--place_epochs", type=int, default=5)
    p.add_argument("--place_lr", type=float, default=3e-4)
    p.add_argument("--place_batch_size", type=int, default=1024)
    p.add_argument("--place_snippet_len", type=int, default=None,
                   help="Rollout snippet length for place head. Defaults to seq_len-1.")
    p.add_argument("--place_embedding_dim", type=int, default=64)
    p.add_argument("--place_triplet_margin", type=float, default=0.2)
    p.add_argument("--place_positive_radius", type=int, default=1,
                   help="Positive snippets come from the same episode within this many snippet starts.")
    p.add_argument("--place_negative_gap", type=int, default=6,
                   help="Same-episode negatives must be at least this many snippet starts away.")
    p.add_argument("--place_positive_radius_m", type=float, default=0.25,
                   help="If rollout pose is available, positives come from the same episode within this XY radius (m).")
    p.add_argument("--place_negative_gap_m", type=float, default=0.75,
                   help="If rollout pose is available, same-episode negatives must be at least this XY distance away (m).")
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

SHARD_SAMPLES = 500_000    # v8 stores rollout pose metadata for place-head training
CACHE_VERSION = 8          # v8: add rollout XY/yaw metadata


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
    """Encode the full dataset once and cache two latent views.

    Encoder view:
      ``enc_projector(z_raw_t)`` for current-frame matching tasks such as
      breadcrumb / goal supervision.

    Rollout view:
      ``pred_projector(predictor(z_raw_t, cmd_t))`` teacher-forced and aligned
      to the next frame's labels. This matches the distribution consumed by the
      planner's safety and exploration heads at inference.
    """
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
    require_rollout_pose = bool(args.raw_data_dir)

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
            and (not require_rollout_pose or bool(info.get("rollout_pose_supervision", False)))
        ):
            print(
                "Latent cache found: "
                f"enc={info.get('n_encoder_samples', info.get('n_samples', 0)):,} "
                f"rollout={info.get('n_rollout_samples', 0):,} "
                f"samples in {info['n_shards']} shards (v{info.get('version', 1)})"
            )
            return cache_dir
        print("Cache version/config mismatch — re-extracting.")

    os.makedirs(cache_dir, exist_ok=True)

    encoder = load_frozen_encoder(args, device)
    print(f"Loaded frozen encoder from {args.checkpoint}")

    num_workers = max(1, int(args.num_workers))
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        raw_data_dir=args.raw_data_dir,
        seq_len=args.seq_len,
        temporal_stride=args.temporal_stride,
        action_block_size=args.action_block_size,
        command_representation=encoder_meta["command_representation"],
        command_latency=encoder_meta["command_latency"],
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=num_workers,
        load_labels=True,
        load_pose=bool(args.raw_data_dir),
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

    z_enc_buf: list[torch.Tensor] = []
    bid_enc_buf: list[torch.Tensor] = []
    br_enc_buf: list[torch.Tensor] = []
    z_rollout_buf: list[torch.Tensor] = []
    st_rollout_buf: list[torch.Tensor] = []
    bid_rollout_buf: list[torch.Tensor] = []
    br_rollout_buf: list[torch.Tensor] = []
    coll_rollout_buf: list[torch.Tensor] = []
    epid_rollout_buf: list[torch.Tensor] = []
    step_rollout_buf: list[torch.Tensor] = []
    xy_rollout_buf: list[torch.Tensor] = []
    yaw_rollout_buf: list[torch.Tensor] = []
    buf_encoder_samples = 0
    buf_rollout_samples = 0
    shard_idx = 0
    total_encoder_samples = 0
    total_rollout_samples = 0
    pbar = None

    def flush_shard():
        nonlocal z_enc_buf, bid_enc_buf, br_enc_buf
        nonlocal z_rollout_buf, st_rollout_buf, bid_rollout_buf, br_rollout_buf, coll_rollout_buf
        nonlocal epid_rollout_buf, step_rollout_buf, xy_rollout_buf, yaw_rollout_buf
        nonlocal buf_encoder_samples, buf_rollout_samples
        nonlocal shard_idx, total_encoder_samples, total_rollout_samples
        if not z_enc_buf:
            return
        shard_path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save({
            "z_enc": torch.cat(z_enc_buf),
            "beacon_identity_enc": torch.cat(bid_enc_buf),
            "beacon_range_enc": torch.cat(br_enc_buf),
            "z_rollout": torch.cat(z_rollout_buf) if z_rollout_buf else torch.empty((0, args.latent_dim)),
            "safety_target_rollout": (
                torch.cat(st_rollout_buf) if st_rollout_buf else torch.empty((0,), dtype=torch.float32)
            ),
            "beacon_identity_rollout": (
                torch.cat(bid_rollout_buf) if bid_rollout_buf else torch.empty((0,), dtype=torch.long)
            ),
            "beacon_range_rollout": (
                torch.cat(br_rollout_buf) if br_rollout_buf else torch.empty((0,), dtype=torch.float32)
            ),
            "collisions_rollout": (
                torch.cat(coll_rollout_buf) if coll_rollout_buf else torch.empty((0,), dtype=torch.float32)
            ),
            "episode_id_rollout": (
                torch.cat(epid_rollout_buf) if epid_rollout_buf else torch.empty((0,), dtype=torch.long)
            ),
            "obs_step_rollout": (
                torch.cat(step_rollout_buf) if step_rollout_buf else torch.empty((0,), dtype=torch.long)
            ),
            "robot_xy_rollout": (
                torch.cat(xy_rollout_buf) if xy_rollout_buf else torch.empty((0, 2), dtype=torch.float32)
            ),
            "robot_yaw_rollout": (
                torch.cat(yaw_rollout_buf) if yaw_rollout_buf else torch.empty((0,), dtype=torch.float32)
            ),
        }, shard_path)
        total_encoder_samples += buf_encoder_samples
        total_rollout_samples += buf_rollout_samples
        progress_write(
            f"  Shard {shard_idx}: enc={buf_encoder_samples:,} rollout={buf_rollout_samples:,} "
            f"samples -> {shard_path}",
            pbar,
        )
        z_enc_buf, bid_enc_buf, br_enc_buf = [], [], []
        z_rollout_buf, st_rollout_buf = [], []
        bid_rollout_buf, br_rollout_buf, coll_rollout_buf = [], [], []
        epid_rollout_buf, step_rollout_buf = [], []
        xy_rollout_buf, yaw_rollout_buf = [], []
        buf_encoder_samples = 0
        buf_rollout_samples = 0
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
            episode_id = labels.get("episode_id")
            obs_step = labels.get("obs_step")
            robot_xy = labels.get("robot_xy")
            robot_yaw = labels.get("robot_yaw")

            vision = vision.to(device, non_blocking=True).float().div_(255.0)
            proprio = proprio.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True).float()

            B, T = vision.shape[:2]

            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                z_raw, z_proj = encoder.encode_seq(vision, proprio)
                z_pred_raw = encoder.predictor(z_raw, cmds)
                z_pred_proj = encoder.pred_projector.forward_seq(z_pred_raw)

            z_enc_flat = z_proj.reshape(B * T, -1).float().cpu()

            # Encoder-view beacon labels (flattened)
            if beacon_range is not None:
                br_enc_flat = beacon_range.float().reshape(B * T)
            else:
                br_enc_flat = torch.full((B * T,), 999.0)
            if beacon_identity is not None:
                bid_enc_flat = beacon_identity.long().reshape(B * T)
            else:
                bid_enc_flat = torch.full((B * T,), -1, dtype=torch.long)

            z_enc_buf.append(z_enc_flat)
            bid_enc_buf.append(bid_enc_flat)
            br_enc_buf.append(br_enc_flat)
            buf_encoder_samples += B * T

            if T > 1:
                z_rollout_flat = z_pred_proj[:, :-1, :].reshape(B * (T - 1), -1).float().cpu()
                coll_rollout_flat = collisions.float()[:, 1:].reshape(B * (T - 1))

                if args.safety_mode == "consequence":
                    safety_rollout = consequence_safety_target(
                        clearance.float()[:, 1:].reshape(B * (T - 1)),
                        traversability[:, 1:].reshape(B * (T - 1)),
                        coll_rollout_flat,
                        traversability_horizon=10,
                        contact_clearance=args.contact_clearance,
                        w_contact=args.w_contact,
                        w_mobility=args.w_mobility,
                    )
                else:
                    safety_rollout = composite_safety_target(
                        clearance.float()[:, 1:],
                        traversability[:, 1:],
                        clearance_clip=args.clearance_clip,
                        traversability_horizon=10,
                        w_safety=args.w_safety,
                        w_mobility=args.w_mobility,
                    ).reshape(B * (T - 1))

                if beacon_range is not None:
                    br_rollout_flat = beacon_range.float()[:, 1:].reshape(B * (T - 1))
                else:
                    br_rollout_flat = torch.full((B * (T - 1),), 999.0)
                if beacon_identity is not None:
                    bid_rollout_flat = beacon_identity.long()[:, 1:].reshape(B * (T - 1))
                else:
                    bid_rollout_flat = torch.full((B * (T - 1),), -1, dtype=torch.long)
                if episode_id is not None:
                    epid_rollout_flat = episode_id.long()[:, 1:].reshape(B * (T - 1))
                else:
                    epid_rollout_flat = torch.full((B * (T - 1),), -1, dtype=torch.long)
                if obs_step is not None:
                    step_rollout_flat = obs_step.long()[:, 1:].reshape(B * (T - 1))
                else:
                    step_rollout_flat = torch.full((B * (T - 1),), -1, dtype=torch.long)
                if robot_xy is not None:
                    xy_rollout_flat = robot_xy.float()[:, 1:, :].reshape(B * (T - 1), 2)
                else:
                    xy_rollout_flat = torch.full((B * (T - 1), 2), float("nan"), dtype=torch.float32)
                if robot_yaw is not None:
                    yaw_rollout_flat = robot_yaw.float()[:, 1:].reshape(B * (T - 1))
                else:
                    yaw_rollout_flat = torch.full((B * (T - 1),), float("nan"), dtype=torch.float32)

                z_rollout_buf.append(z_rollout_flat)
                st_rollout_buf.append(safety_rollout)
                bid_rollout_buf.append(bid_rollout_flat)
                br_rollout_buf.append(br_rollout_flat)
                coll_rollout_buf.append(coll_rollout_flat)
                epid_rollout_buf.append(epid_rollout_flat)
                step_rollout_buf.append(step_rollout_flat)
                xy_rollout_buf.append(xy_rollout_flat)
                yaw_rollout_buf.append(yaw_rollout_flat)
                buf_rollout_samples += B * (T - 1)

            if buf_encoder_samples >= SHARD_SAMPLES:
                flush_shard()

    pbar = None
    flush_shard()

    elapsed = time.time() - t0
    torch.save({
        "n_samples": total_encoder_samples,
        "n_encoder_samples": total_encoder_samples,
        "n_rollout_samples": total_rollout_samples,
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
        "latent_views": {
            "encoder": "enc_projector(z_obs_t)",
            "rollout": "pred_projector(predictor(z_obs_t, cmd_t)) aligned to labels[t+1]",
        },
        "rollout_pose_supervision": bool(args.raw_data_dir),
    }, manifest_path)
    print(
        f"Extraction complete: enc={total_encoder_samples:,} rollout={total_rollout_samples:,} "
        f"samples, {shard_idx} shards, {elapsed:.0f}s"
    )

    del encoder
    torch.cuda.empty_cache()
    return cache_dir


def load_cached_latents(cache_dir: str):
    """Load all shards and return encoder-view and rollout-view latent banks."""
    manifest = torch.load(os.path.join(cache_dir, "manifest.pt"), map_location="cpu")
    z_enc_all, bid_enc_all, br_enc_all = [], [], []
    z_rollout_all, st_rollout_all = [], []
    bid_rollout_all, br_rollout_all, coll_rollout_all = [], [], []
    epid_rollout_all, step_rollout_all = [], []
    xy_rollout_all, yaw_rollout_all = [], []
    for i in range(manifest["n_shards"]):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i:04d}.pt"), map_location="cpu")
        z_enc_all.append(shard["z_enc"])
        bid_enc_all.append(shard["beacon_identity_enc"])
        br_enc_all.append(shard["beacon_range_enc"])
        z_rollout_all.append(shard["z_rollout"])
        st_rollout_all.append(shard["safety_target_rollout"])
        bid_rollout_all.append(shard["beacon_identity_rollout"])
        br_rollout_all.append(shard["beacon_range_rollout"])
        coll_rollout_all.append(shard["collisions_rollout"])
        epid_rollout_all.append(shard.get("episode_id_rollout", torch.empty((0,), dtype=torch.long)))
        step_rollout_all.append(shard.get("obs_step_rollout", torch.empty((0,), dtype=torch.long)))
        xy_rollout_all.append(shard.get("robot_xy_rollout", torch.empty((0, 2), dtype=torch.float32)))
        yaw_rollout_all.append(shard.get("robot_yaw_rollout", torch.empty((0,), dtype=torch.float32)))
        print(
            f"  Loaded shard {i}: enc={shard['z_enc'].shape[0]:,} "
            f"rollout={shard['z_rollout'].shape[0]:,} samples"
        )

    enc_z = torch.cat(z_enc_all)
    enc_bid = torch.cat(bid_enc_all)
    enc_br = torch.cat(br_enc_all)
    rollout_z = torch.cat(z_rollout_all)
    rollout_st = torch.cat(st_rollout_all)
    rollout_bid = torch.cat(bid_rollout_all)
    rollout_br = torch.cat(br_rollout_all)
    rollout_coll = torch.cat(coll_rollout_all)
    rollout_epid = torch.cat(epid_rollout_all)
    rollout_step = torch.cat(step_rollout_all)
    rollout_xy = torch.cat(xy_rollout_all)
    rollout_yaw = torch.cat(yaw_rollout_all)

    print(f"Encoder view: {enc_z.shape[0]:,} samples, z={enc_z.shape}")
    print(f"Rollout view: {rollout_z.shape[0]:,} samples, z={rollout_z.shape}")
    n_contact = (rollout_st > 0.9).sum().item()
    print(
        f"  High-energy rollout samples (>0.9): {n_contact:,} "
        f"({100*n_contact/max(1, len(rollout_st)):.1f}%)"
    )
    return {
        "encoder": {
            "z": enc_z,
            "beacon_identity": enc_bid,
            "beacon_range": enc_br,
        },
        "rollout": {
            "z": rollout_z,
            "safety_target": rollout_st,
            "beacon_identity": rollout_bid,
            "beacon_range": rollout_br,
            "collisions": rollout_coll,
            "episode_id": rollout_epid,
            "obs_step": rollout_step,
            "robot_xy": rollout_xy,
            "robot_yaw": rollout_yaw,
        },
    }


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
# Phase 6: Train PlaceSnippetHead (rollout snippet metric)
# --------------------------------------------------------------------- #

def build_rollout_snippet_bank(
    z_rollout: torch.Tensor,
    episode_id: torch.Tensor,
    obs_step: torch.Tensor,
    snippet_len: int,
    temporal_stride: int,
    robot_xy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate rollout states and assemble contiguous snippet windows."""
    if snippet_len <= 0:
        raise ValueError(f"snippet_len must be positive, got {snippet_len}")

    per_episode: dict[int, dict[int, tuple[torch.Tensor, int]]] = {}
    per_episode_xy: dict[int, dict[int, tuple[torch.Tensor, int]]] | None = (
        {} if robot_xy is not None else None
    )
    for idx in range(int(z_rollout.shape[0])):
        ep = int(episode_id[idx].item())
        step = int(obs_step[idx].item())
        if ep < 0 or step < 0:
            continue
        episode_map = per_episode.setdefault(ep, {})
        if step not in episode_map:
            episode_map[step] = (z_rollout[idx].clone(), 1)
        else:
            accum, count = episode_map[step]
            episode_map[step] = (accum + z_rollout[idx], count + 1)
        if per_episode_xy is not None:
            xy = robot_xy[idx]
            if torch.isfinite(xy).all():
                xy_map = per_episode_xy.setdefault(ep, {})
                if step not in xy_map:
                    xy_map[step] = (xy.clone(), 1)
                else:
                    xy_accum, xy_count = xy_map[step]
                    xy_map[step] = (xy_accum + xy, xy_count + 1)

    snippets: list[torch.Tensor] = []
    snippet_epids: list[int] = []
    snippet_starts: list[int] = []
    snippet_xy: list[torch.Tensor] = []
    stride = int(temporal_stride)
    for ep, step_map in per_episode.items():
        ordered_steps = sorted(step_map.keys())
        if len(ordered_steps) < snippet_len:
            continue
        dedup_latents = {
            step: accum / float(count)
            for step, (accum, count) in step_map.items()
        }
        dedup_xy = None
        if per_episode_xy is not None and ep in per_episode_xy:
            dedup_xy = {
                step: accum / float(count)
                for step, (accum, count) in per_episode_xy[ep].items()
            }
        for start_idx in range(len(ordered_steps) - snippet_len + 1):
            window_steps = ordered_steps[start_idx:start_idx + snippet_len]
            if any((b - a) != stride for a, b in zip(window_steps[:-1], window_steps[1:])):
                continue
            snippets.append(torch.stack([dedup_latents[step] for step in window_steps], dim=0))
            snippet_epids.append(ep)
            snippet_starts.append(window_steps[0])
            if dedup_xy is not None:
                center_step = window_steps[snippet_len // 2]
                center_xy = dedup_xy.get(center_step)
                if center_xy is None:
                    snippet_xy.append(torch.full((2,), float("nan"), dtype=torch.float32))
                else:
                    snippet_xy.append(center_xy.clone().to(dtype=torch.float32))
            else:
                snippet_xy.append(torch.full((2,), float("nan"), dtype=torch.float32))

    if not snippets:
        raise ValueError("No contiguous rollout snippets could be built from the cached rollout bank.")

    return (
        torch.stack(snippets, dim=0),
        torch.tensor(snippet_epids, dtype=torch.long),
        torch.tensor(snippet_starts, dtype=torch.long),
        torch.stack(snippet_xy, dim=0),
    )


def train_place_head(
    args,
    rollout_snippets: torch.Tensor,
    snippet_episode_id: torch.Tensor,
    snippet_start: torch.Tensor,
    snippet_xy: torch.Tensor | None,
    device,
):
    print("\n" + "=" * 60)
    print("Phase 6: Training PlaceSnippetHead (rollout snippet metric)")
    print("=" * 60)

    n_snippets = int(rollout_snippets.shape[0])
    if n_snippets < 2:
        print("  Skipping: not enough rollout snippets for place training.")
        return None

    snippet_len = int(rollout_snippets.shape[1])
    print(
        f"  Snippets: {n_snippets:,} | len={snippet_len} | "
        f"positive_radius={args.place_positive_radius} | negative_gap={args.place_negative_gap}"
    )
    start_stride = max(1, int(args.temporal_stride))
    positive_radius_raw = int(args.place_positive_radius) * start_stride
    negative_gap_raw = int(args.place_negative_gap) * start_stride
    print(
        f"  Snippet-start stride={start_stride} raw steps "
        f"(positive<= {positive_radius_raw}, negative>= {negative_gap_raw})"
    )
    use_pose_supervision = bool(
        snippet_xy is not None
        and snippet_xy.numel() > 0
        and torch.isfinite(snippet_xy).all()
        and float(args.place_positive_radius_m) > 0.0
        and float(args.place_negative_gap_m) > float(args.place_positive_radius_m)
    )
    if use_pose_supervision:
        print(
            "  Place supervision: pose_xy_same_episode "
            f"(positive<= {float(args.place_positive_radius_m):.3f}m, "
            f"negative>= {float(args.place_negative_gap_m):.3f}m)"
        )
    else:
        print("  Place supervision: temporal proxy fallback")

    head = PlaceSnippetHead(
        latent_dim=args.latent_dim,
        snippet_len=snippet_len,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.place_embedding_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.place_lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.place_epochs)

    snippets_cpu = rollout_snippets
    ep_cpu = snippet_episode_id.long()
    start_cpu = snippet_start.long()
    xy_cpu = snippet_xy.float() if snippet_xy is not None else None
    all_indices = torch.arange(n_snippets, dtype=torch.long)

    by_episode: dict[int, torch.Tensor] = {}
    for ep in ep_cpu.unique(sorted=True).tolist():
        by_episode[int(ep)] = torch.nonzero(ep_cpu == int(ep), as_tuple=False).squeeze(1)

    csv_path = os.path.join(args.log_dir, "place_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "d_ap", "d_an", "lr"])

    global_step = 0
    n_batches = max(1, math.ceil(n_snippets / max(1, args.place_batch_size)))
    for epoch in range(args.place_epochs):
        epoch_loss = 0.0
        epoch_n = 0
        epoch_triplets = 0
        t0 = time.time()
        head.train()

        with make_progress(args, range(n_batches), desc=f"  Place {epoch + 1}/{args.place_epochs}") as pbar:
            for _ in pbar:
                anchor_idx = torch.randint(0, n_snippets, (args.place_batch_size,), dtype=torch.long)
                pos_idx: list[int] = []
                neg_idx: list[int] = []
                valid_anchor: list[int] = []

                for idx in anchor_idx.tolist():
                    ep = int(ep_cpu[idx].item())
                    same_ep = by_episode[ep]
                    if use_pose_supervision:
                        anchor_xy = xy_cpu[idx]
                        same_xy = xy_cpu[same_ep]
                        delta_xy = (same_xy - anchor_xy.unsqueeze(0)).square().sum(dim=-1).sqrt()
                        pos_candidates = same_ep[(delta_xy > 0.0) & (delta_xy <= float(args.place_positive_radius_m))]
                        if pos_candidates.numel() == 0:
                            continue
                        far_same = same_ep[delta_xy >= float(args.place_negative_gap_m)]
                    else:
                        step = int(start_cpu[idx].item())
                        same_steps = start_cpu[same_ep]
                        delta_raw = (same_steps - step).abs()
                        pos_candidates = same_ep[(delta_raw > 0) & (delta_raw <= positive_radius_raw)]
                        if pos_candidates.numel() == 0:
                            continue
                        far_same = same_ep[delta_raw >= negative_gap_raw]

                    if far_same.numel() > 0 and torch.rand(()).item() < 0.5:
                        neg_candidate_pool = far_same
                    else:
                        neg_candidate_pool = all_indices[ep_cpu != ep]
                        if neg_candidate_pool.numel() == 0:
                            continue

                    pos_choice = int(pos_candidates[torch.randint(0, pos_candidates.numel(), (1,))].item())
                    neg_choice = int(neg_candidate_pool[torch.randint(0, neg_candidate_pool.numel(), (1,))].item())
                    valid_anchor.append(idx)
                    pos_idx.append(pos_choice)
                    neg_idx.append(neg_choice)

                if not valid_anchor:
                    continue

                anchor = snippets_cpu[torch.tensor(valid_anchor, dtype=torch.long)].to(device, non_blocking=True)
                positive = snippets_cpu[torch.tensor(pos_idx, dtype=torch.long)].to(device, non_blocking=True)
                negative = snippets_cpu[torch.tensor(neg_idx, dtype=torch.long)].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                z_anchor = head(anchor)
                z_positive = head(positive)
                z_negative = head(negative)

                d_ap = (z_anchor - z_positive).square().sum(dim=-1)
                d_an = (z_anchor - z_negative).square().sum(dim=-1)
                loss = torch.relu(d_ap - d_an + float(args.place_triplet_margin)).mean()
                loss = loss + 0.1 * d_ap.mean()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip).item()
                if not torch.isfinite(loss) or not math.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                global_step += 1
                loss_val = float(loss.item())
                epoch_loss += loss_val
                epoch_n += 1
                epoch_triplets += len(valid_anchor)

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step,
                        epoch + 1,
                        f"{loss_val:.6f}",
                        f"{d_ap.detach().mean().item():.6f}",
                        f"{d_an.detach().mean().item():.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        d_ap=f"{d_ap.detach().mean().item():.3f}",
                        d_an=f"{d_an.detach().mean().item():.3f}",
                    )

        if epoch_n == 0:
            raise RuntimeError(
                "PlaceSnippetHead training found zero valid triplets. "
                "This usually means the positive/negative radius is inconsistent "
                "with the rollout snippet spacing."
            )
        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        avg_triplets = epoch_triplets / max(1, epoch_n)
        print(
            f"  Epoch {epoch + 1} | avg_loss={avg:.6f} | "
            f"avg_triplets={avg_triplets:.1f} | time={time.time() - t0:.0f}s"
        )

    torch.save(
        {"place_head_state_dict": head.state_dict(), "epoch": args.place_epochs},
        os.path.join(args.out_dir, "place_head.pt"),
    )
    print("  Place head training complete.")
    return head


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

    # Load cached encoder-view and rollout-view latent banks
    print("\nLoading cached latents...")
    latent_views = load_cached_latents(cache_dir)
    z_enc = latent_views["encoder"]["z"]
    beacon_id_enc = latent_views["encoder"]["beacon_identity"]
    beacon_range_enc = latent_views["encoder"]["beacon_range"]
    z_rollout = latent_views["rollout"]["z"]
    safety_target_rollout = latent_views["rollout"]["safety_target"]
    rollout_episode_id = latent_views["rollout"]["episode_id"]
    rollout_obs_step = latent_views["rollout"]["obs_step"]
    rollout_robot_xy = latent_views["rollout"]["robot_xy"]
    print(
        f"Using rollout-view latents for safety/exploration: {tuple(z_rollout.shape)} | "
        f"encoder-view latents for goal/progress: {tuple(z_enc.shape)}"
    )

    # Phase 2: safety head
    safety_head = train_safety_head(args, z_rollout, safety_target_rollout, device)

    # Phase 3: goal head
    goal_head = None
    if not args.skip_goal:
        goal_head = train_goal_head(args, z_enc, beacon_id_enc, beacon_range_enc, device)

    # Phase 4: progress head
    progress_head = None
    if not args.skip_progress:
        goal_pools = build_goal_latent_pools(z_enc, beacon_id_enc, beacon_range_enc, args.beacon_clip)
        progress_head = train_progress_head(args, goal_pools, device)

    # Phase 5: exploration bonus
    exploration = None
    if not args.skip_exploration:
        exploration = train_exploration(args, z_rollout, device)

    # Phase 6: place head
    place_head = None
    place_snippet_len = int(
        args.place_snippet_len if args.place_snippet_len is not None else max(1, args.seq_len - 1)
    )
    if not args.skip_place:
        rollout_snippets, snippet_episode_id, snippet_start, snippet_xy = build_rollout_snippet_bank(
            z_rollout,
            rollout_episode_id,
            rollout_obs_step,
            snippet_len=place_snippet_len,
            temporal_stride=args.temporal_stride,
            robot_xy=rollout_robot_xy,
        )
        place_head = train_place_head(
            args,
            rollout_snippets,
            snippet_episode_id,
            snippet_start,
            snippet_xy,
            device,
        )

    # Save combined TrajectoryScorer checkpoint
    scorer_path = os.path.join(args.out_dir, "trajectory_scorer.pt")
    scorer_data = {
        "safety_head": safety_head.state_dict(),
        "goal_head": goal_head.state_dict() if goal_head is not None else None,
        "progress_head": progress_head.state_dict() if progress_head is not None else None,
        "exploration": exploration.state_dict() if exploration is not None else None,
        "place_head": place_head.state_dict() if place_head is not None else None,
        "goal_weight": args.goal_weight,
        "progress_weight": args.progress_weight,
        "exploration_weight": args.exploration_weight,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "exploration_feature_dim": args.exploration_feature_dim,
        "place_embedding_dim": args.place_embedding_dim,
        "place_snippet_len": place_snippet_len,
        "safety_mode": args.safety_mode,
        "safety_latent_source": "pred_projector_teacher_forced",
        "exploration_latent_source": "pred_projector_teacher_forced",
        "place_latent_source": "pred_projector_teacher_forced_snippet",
        "place_supervision": (
            "pose_xy_same_episode"
            if args.raw_data_dir is not None
            else "temporal_proxy"
        ),
        "goal_latent_source": "enc_projector",
        "progress_latent_source": "enc_projector",
        "cache_version": CACHE_VERSION,
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
