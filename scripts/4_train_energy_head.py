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
import torch.nn.functional as F
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
    DisplacementHead,
    CoverageGainHead,
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
                   help="Optional directory with raw chunk_*.npz rollouts for pose-supervised place/displacement training.")
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
    # Displacement head
    p.add_argument("--skip_displacement", action="store_true",
                   help="Skip pose-supervised displacement head training.")
    p.add_argument("--displacement_epochs", type=int, default=5)
    p.add_argument("--displacement_lr", type=float, default=3e-4)
    p.add_argument("--displacement_batch_size", type=int, default=4096)
    p.add_argument("--displacement_hops", type=int, default=5,
                   help="Train displacement targets between obs_t and obs_{t+hops}. Should usually match the planner horizon in model steps.")
    p.add_argument("--displacement_weight", type=float, default=0.0,
                   help="Optional planner bonus weight for the displacement head. Kept at 0 by default for audit-first evaluation.")
    # Coverage-gain head
    p.add_argument("--skip_coverage_gain", action="store_true",
                   help="Skip sequence-level coverage-gain head training.")
    p.add_argument("--coverage_gain_epochs", type=int, default=5)
    p.add_argument("--coverage_gain_lr", type=float, default=3e-4)
    p.add_argument("--coverage_gain_batch_size", type=int, default=4096)
    p.add_argument("--coverage_gain_hops", type=int, default=5,
                   help="Number of future model steps in the rollout snippet scored by the coverage-gain head.")
    p.add_argument("--coverage_gain_target_hops", type=int, default=20,
                   help="Number of future model steps used to build the realized coverage-gain target.")
    p.add_argument("--coverage_gain_context_hops", type=int, default=10,
                   help="Number of recent model steps used to define already-visited local context for coverage-gain targets.")
    p.add_argument("--coverage_gain_radius_m", type=float, default=0.18,
                   help="Radius used when converting novel path length into a local coverage-area gain proxy.")
    p.add_argument("--coverage_gain_densify_step_m", type=float, default=0.04,
                   help="Interpolation step for pose-based local coverage-gain targets.")
    p.add_argument("--coverage_gain_weight", type=float, default=0.0,
                   help="Optional planner bonus weight for the coverage-gain head. Kept at 0 by default for audit-first evaluation.")
    # Held-out evaluation
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip held-out episode evaluation for safety/place heads.")
    p.add_argument("--eval_holdout_fraction", type=float, default=0.10,
                   help="Fraction of rollout episodes held out from training for validation.")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--eval_place_max_queries", type=int, default=4096,
                   help="Cap the number of held-out snippet queries used for place retrieval evaluation.")
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

SHARD_SAMPLES = 500_000    # v9 stores encoder + rollout pose metadata for displacement-head training
CACHE_VERSION = 9          # v9: add encoder episode/step/XY/yaw metadata


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
    epid_enc_buf: list[torch.Tensor] = []
    step_enc_buf: list[torch.Tensor] = []
    xy_enc_buf: list[torch.Tensor] = []
    yaw_enc_buf: list[torch.Tensor] = []
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
        nonlocal z_enc_buf, bid_enc_buf, br_enc_buf, epid_enc_buf, step_enc_buf, xy_enc_buf, yaw_enc_buf
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
            "episode_id_enc": (
                torch.cat(epid_enc_buf) if epid_enc_buf else torch.empty((0,), dtype=torch.long)
            ),
            "obs_step_enc": (
                torch.cat(step_enc_buf) if step_enc_buf else torch.empty((0,), dtype=torch.long)
            ),
            "robot_xy_enc": (
                torch.cat(xy_enc_buf) if xy_enc_buf else torch.empty((0, 2), dtype=torch.float32)
            ),
            "robot_yaw_enc": (
                torch.cat(yaw_enc_buf) if yaw_enc_buf else torch.empty((0,), dtype=torch.float32)
            ),
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
        epid_enc_buf, step_enc_buf = [], []
        xy_enc_buf, yaw_enc_buf = [], []
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
            if episode_id is not None:
                epid_enc_flat = episode_id.long().reshape(B * T)
            else:
                epid_enc_flat = torch.full((B * T,), -1, dtype=torch.long)
            if obs_step is not None:
                step_enc_flat = obs_step.long().reshape(B * T)
            else:
                step_enc_flat = torch.full((B * T,), -1, dtype=torch.long)
            if robot_xy is not None:
                xy_enc_flat = robot_xy.float().reshape(B * T, 2)
            else:
                xy_enc_flat = torch.full((B * T, 2), float("nan"), dtype=torch.float32)
            if robot_yaw is not None:
                yaw_enc_flat = robot_yaw.float().reshape(B * T)
            else:
                yaw_enc_flat = torch.full((B * T,), float("nan"), dtype=torch.float32)

            z_enc_buf.append(z_enc_flat)
            bid_enc_buf.append(bid_enc_flat)
            br_enc_buf.append(br_enc_flat)
            epid_enc_buf.append(epid_enc_flat)
            step_enc_buf.append(step_enc_flat)
            xy_enc_buf.append(xy_enc_flat)
            yaw_enc_buf.append(yaw_enc_flat)
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
    epid_enc_all, step_enc_all = [], []
    xy_enc_all, yaw_enc_all = [], []
    z_rollout_all, st_rollout_all = [], []
    bid_rollout_all, br_rollout_all, coll_rollout_all = [], [], []
    epid_rollout_all, step_rollout_all = [], []
    xy_rollout_all, yaw_rollout_all = [], []
    for i in range(manifest["n_shards"]):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i:04d}.pt"), map_location="cpu")
        z_enc_all.append(shard["z_enc"])
        bid_enc_all.append(shard["beacon_identity_enc"])
        br_enc_all.append(shard["beacon_range_enc"])
        epid_enc_all.append(shard.get("episode_id_enc", torch.empty((0,), dtype=torch.long)))
        step_enc_all.append(shard.get("obs_step_enc", torch.empty((0,), dtype=torch.long)))
        xy_enc_all.append(shard.get("robot_xy_enc", torch.empty((0, 2), dtype=torch.float32)))
        yaw_enc_all.append(shard.get("robot_yaw_enc", torch.empty((0,), dtype=torch.float32)))
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
    enc_epid = torch.cat(epid_enc_all)
    enc_step = torch.cat(step_enc_all)
    enc_xy = torch.cat(xy_enc_all)
    enc_yaw = torch.cat(yaw_enc_all)
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
            "episode_id": enc_epid,
            "obs_step": enc_step,
            "robot_xy": enc_xy,
            "robot_yaw": enc_yaw,
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


def build_rollout_holdout_masks(
    episode_id: torch.Tensor,
    holdout_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]] | None:
    """Split rollout samples by episode id so evaluation uses unseen episodes."""
    if holdout_fraction <= 0.0:
        return None
    valid = episode_id >= 0
    if not torch.any(valid):
        return None

    unique_eps = torch.unique(episode_id[valid], sorted=True)
    n_eps = int(unique_eps.numel())
    if n_eps < 2:
        return None

    n_eval_eps = max(1, int(round(float(holdout_fraction) * n_eps)))
    n_eval_eps = min(n_eval_eps, n_eps - 1)
    if n_eval_eps <= 0:
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_eps, generator=generator)
    eval_eps = unique_eps[perm[:n_eval_eps]]
    eval_mask = valid & torch.isin(episode_id, eval_eps)
    train_mask = valid & (~eval_mask)

    if not torch.any(train_mask) or not torch.any(eval_mask):
        return None

    return train_mask, eval_mask, {
        "n_episodes": n_eps,
        "n_train_episodes": int(torch.unique(episode_id[train_mask]).numel()),
        "n_eval_episodes": int(torch.unique(episode_id[eval_mask]).numel()),
        "n_train_samples": int(train_mask.sum().item()),
        "n_eval_samples": int(eval_mask.sum().item()),
    }


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x64 = x.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    x_center = x64 - x64.mean()
    y_center = y64 - y64.mean()
    denom = torch.sqrt((x_center.square().mean()) * (y_center.square().mean()))
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float((x_center * y_center).mean().div(denom).item())


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.to(dtype=torch.bool)
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores64 = scores.to(dtype=torch.float64)
    order = torch.argsort(scores64, stable=True)
    ranks = torch.empty_like(scores64, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores64) + 1, dtype=torch.float64)
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc.item())


def evaluate_safety_head(
    args,
    head: nn.Module,
    z_eval: torch.Tensor,
    target_eval: torch.Tensor,
    collisions_eval: torch.Tensor,
    device,
) -> dict[str, float] | None:
    if z_eval.numel() == 0:
        return None

    dataset = TensorDataset(z_eval, target_eval, collisions_eval)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    preds_all: list[torch.Tensor] = []
    target_all: list[torch.Tensor] = []
    coll_all: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for z_batch, target_batch, coll_batch in dataloader:
            z_batch = z_batch.to(device, non_blocking=True)
            pred = head(z_batch).float().cpu()
            preds_all.append(pred)
            target_all.append(target_batch.float().cpu())
            coll_all.append(coll_batch.float().cpu())

    pred = torch.cat(preds_all)
    target = torch.cat(target_all)
    coll = torch.cat(coll_all)
    mse = float((pred - target).square().mean().item())
    metrics = {
        "mse": mse,
        "rmse": math.sqrt(max(0.0, mse)),
        "mae": float((pred - target).abs().mean().item()),
        "pearson": pearson_corrcoef(pred, target),
        "collision_rate": float(coll.mean().item()),
        "collision_auc": binary_auc(pred, coll > 0.5),
        "pred_mean_collision": float(pred[coll > 0.5].mean().item()) if torch.any(coll > 0.5) else float("nan"),
        "pred_mean_no_collision": float(pred[coll <= 0.5].mean().item()) if torch.any(coll <= 0.5) else float("nan"),
        "target_mean": float(target.mean().item()),
        "pred_mean": float(pred.mean().item()),
        "n_eval": int(pred.numel()),
    }
    print(
        "  Held-out safety | "
        f"n={metrics['n_eval']:,} "
        f"rmse={metrics['rmse']:.4f} "
        f"mae={metrics['mae']:.4f} "
        f"pearson={metrics['pearson']:.3f} "
        f"collision_auc={metrics['collision_auc']:.3f} "
        f"pred(coll)={metrics['pred_mean_collision']:.3f} "
        f"pred(no-coll)={metrics['pred_mean_no_collision']:.3f}"
    )
    return metrics


def evaluate_place_head(
    args,
    head: nn.Module,
    rollout_snippets: torch.Tensor,
    snippet_episode_id: torch.Tensor,
    snippet_xy: torch.Tensor,
    device,
) -> dict[str, float] | None:
    if rollout_snippets.numel() == 0 or snippet_episode_id.numel() == 0:
        return None
    if not torch.isfinite(snippet_xy).all():
        return None

    dataset = TensorDataset(rollout_snippets)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    emb_all: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for (snippet_batch,) in dataloader:
            snippet_batch = snippet_batch.to(device, non_blocking=True)
            emb_all.append(head(snippet_batch).float().cpu())
    emb = torch.cat(emb_all)

    by_episode: dict[int, torch.Tensor] = {}
    for ep in snippet_episode_id.unique(sorted=True).tolist():
        ep_int = int(ep)
        if ep_int < 0:
            continue
        by_episode[ep_int] = torch.nonzero(snippet_episode_id == ep_int, as_tuple=False).squeeze(1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.eval_seed) + 17)
    query_budget = max(1, int(args.eval_place_max_queries))
    queried = 0
    recall1 = 0
    recall5 = 0
    d_pos_all: list[float] = []
    d_neg_all: list[float] = []

    for ep in sorted(by_episode.keys()):
        ep_idx = by_episode[ep]
        if int(ep_idx.numel()) < 2:
            continue
        ep_emb = emb[ep_idx]
        ep_xy = snippet_xy[ep_idx]
        xy_d = torch.cdist(ep_xy, ep_xy)
        emb_d = torch.cdist(ep_emb, ep_emb)
        emb_d.fill_diagonal_(float("inf"))
        positive_mask = (xy_d > 0.0) & (xy_d <= float(args.place_positive_radius_m))
        negative_mask = xy_d >= float(args.place_negative_gap_m)
        valid_queries = torch.nonzero(positive_mask.any(dim=1), as_tuple=False).squeeze(1)
        if valid_queries.numel() == 0:
            continue
        if queried >= query_budget:
            break
        remaining = query_budget - queried
        if valid_queries.numel() > remaining:
            take = torch.randperm(valid_queries.numel(), generator=generator)[:remaining]
            valid_queries = valid_queries[take]

        for q in valid_queries.tolist():
            ranking = torch.argsort(emb_d[q])
            top1 = ranking[:1]
            top5 = ranking[: min(5, ranking.numel())]
            recall1 += int(bool(positive_mask[q, top1].any().item()))
            recall5 += int(bool(positive_mask[q, top5].any().item()))
            d_pos_all.append(float(emb_d[q][positive_mask[q]].min().item()))
            if bool(negative_mask[q].any().item()):
                d_neg_all.append(float(emb_d[q][negative_mask[q]].min().item()))
        queried += int(valid_queries.numel())

    if queried == 0:
        return None

    d_pos_mean = float(sum(d_pos_all) / max(1, len(d_pos_all)))
    d_neg_mean = float(sum(d_neg_all) / max(1, len(d_neg_all))) if d_neg_all else float("nan")
    metrics = {
        "n_queries": queried,
        "recall_at_1": recall1 / float(queried),
        "recall_at_5": recall5 / float(queried),
        "nearest_positive_dist": d_pos_mean,
        "nearest_negative_dist": d_neg_mean,
        "distance_gap": d_neg_mean - d_pos_mean if math.isfinite(d_neg_mean) else float("nan"),
    }
    print(
        "  Held-out place | "
        f"queries={metrics['n_queries']:,} "
        f"R@1={metrics['recall_at_1']:.3f} "
        f"R@5={metrics['recall_at_5']:.3f} "
        f"d_pos={metrics['nearest_positive_dist']:.3f} "
        f"d_neg={metrics['nearest_negative_dist']:.3f}"
    )
    return metrics


def dedup_pose_latent_bank(
    z_bank: torch.Tensor,
    episode_id: torch.Tensor,
    obs_step: torch.Tensor,
    robot_xy: torch.Tensor,
) -> dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Average duplicate entries keyed by (episode_id, obs_step)."""
    per_episode: dict[int, dict[int, tuple[torch.Tensor, int, torch.Tensor, int]]] = {}
    for idx in range(int(z_bank.shape[0])):
        ep = int(episode_id[idx].item())
        step = int(obs_step[idx].item())
        if ep < 0 or step < 0:
            continue
        xy = robot_xy[idx]
        if xy.numel() != 2 or not torch.isfinite(xy).all():
            continue
        ep_map = per_episode.setdefault(ep, {})
        if step not in ep_map:
            ep_map[step] = (z_bank[idx].clone(), 1, xy.clone(), 1)
        else:
            z_accum, z_count, xy_accum, xy_count = ep_map[step]
            ep_map[step] = (
                z_accum + z_bank[idx],
                z_count + 1,
                xy_accum + xy,
                xy_count + 1,
            )

    dedup: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for ep, step_map in per_episode.items():
        dedup[ep] = {
            step: (
                z_accum / float(z_count),
                xy_accum / float(xy_count),
            )
            for step, (z_accum, z_count, xy_accum, xy_count) in step_map.items()
        }
    return dedup


def densify_xy_path(
    xy_seq: torch.Tensor,
    step_m: float,
) -> torch.Tensor:
    """Return a densely sampled polyline for XY path inputs."""
    if xy_seq.ndim != 2 or xy_seq.shape[-1] != 2:
        raise ValueError(f"Expected xy_seq shape (N, 2), got {tuple(xy_seq.shape)}")
    if xy_seq.shape[0] == 0:
        return xy_seq
    if xy_seq.shape[0] == 1:
        return xy_seq.clone()

    pts: list[torch.Tensor] = [xy_seq[0].to(dtype=torch.float32)]
    step_m = max(1e-3, float(step_m))
    for start, end in zip(xy_seq[:-1], xy_seq[1:]):
        delta = (end - start).to(dtype=torch.float32)
        dist = float(torch.linalg.vector_norm(delta, ord=2).item())
        if dist < 1e-6:
            continue
        n_steps = max(1, int(math.ceil(dist / step_m)))
        for idx in range(1, n_steps + 1):
            alpha = float(idx) / float(n_steps)
            pts.append(start.to(dtype=torch.float32) + alpha * delta)
    return torch.stack(pts, dim=0)


def novel_path_area_gain_proxy(
    context_xy_seq: torch.Tensor,
    future_xy_seq: torch.Tensor,
    radius_m: float,
    densify_step_m: float,
) -> float:
    """Approximate local coverage gain from a future path beyond recent context.

    The target is a pose-supervised local area proxy:

    - densify the recent context path and the future path
    - walk forward along the future path
    - count only arc-length that leaves a radius-``radius_m`` neighborhood of
      all previously covered points
    - convert that novel path length into a swept-area proxy by multiplying by
      ``2 * radius_m``

    This remains a coverage-gain target rather than a raw endpoint-displacement
    target, while staying cheap enough to build over the cached latent bank.
    """
    if context_xy_seq.shape[0] == 0 or future_xy_seq.shape[0] == 0:
        return 0.0
    context_dense = densify_xy_path(context_xy_seq, densify_step_m)
    chained_future = torch.cat(
        [context_xy_seq[-1:].to(dtype=torch.float32), future_xy_seq.to(dtype=torch.float32)],
        dim=0,
    )
    future_dense = densify_xy_path(chained_future, densify_step_m)
    if future_dense.shape[0] <= 1:
        return 0.0
    future_dense = future_dense[1:]

    support = context_dense
    novel_length_m = 0.0
    radius_m = float(radius_m)
    densify_step_m = float(densify_step_m)
    for point in future_dense:
        d_min = float(torch.cdist(point.view(1, 2), support).amin().item())
        if d_min > radius_m:
            novel_length_m += densify_step_m
        support = torch.cat([support, point.view(1, 2)], dim=0)
    return novel_length_m * (2.0 * radius_m)


def build_displacement_pairs(
    z_enc: torch.Tensor,
    enc_episode_id: torch.Tensor,
    enc_obs_step: torch.Tensor,
    enc_robot_xy: torch.Tensor,
    z_rollout: torch.Tensor,
    rollout_episode_id: torch.Tensor,
    rollout_obs_step: torch.Tensor,
    rollout_robot_xy: torch.Tensor,
    displacement_hops: int,
    temporal_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair current encoder latents with future rollout latents plus XY targets.

    The future rollout latent is aligned to the real observation at
    ``obs_step + displacement_hops * temporal_stride``, which makes the target
    directly comparable to planner terminal states scored in projected latent
    space.
    """
    if displacement_hops <= 0:
        raise ValueError(f"displacement_hops must be positive, got {displacement_hops}")

    current_bank = dedup_pose_latent_bank(z_enc, enc_episode_id, enc_obs_step, enc_robot_xy)
    future_bank = dedup_pose_latent_bank(z_rollout, rollout_episode_id, rollout_obs_step, rollout_robot_xy)
    raw_step_delta = int(displacement_hops) * int(temporal_stride)

    z_now_pairs: list[torch.Tensor] = []
    z_future_pairs: list[torch.Tensor] = []
    target_disp_pairs: list[torch.Tensor] = []
    for ep, cur_steps in current_bank.items():
        fut_steps = future_bank.get(ep)
        if not fut_steps:
            continue
        for cur_step, (z_now, xy_now) in cur_steps.items():
            target_step = cur_step + raw_step_delta
            if target_step not in fut_steps:
                continue
            z_future, xy_future = fut_steps[target_step]
            disp_m = torch.linalg.vector_norm((xy_future - xy_now).to(dtype=torch.float32), ord=2)
            z_now_pairs.append(z_now.to(dtype=torch.float32))
            z_future_pairs.append(z_future.to(dtype=torch.float32))
            target_disp_pairs.append(disp_m.reshape(1))

    if not z_now_pairs:
        raise ValueError(
            "No valid encoder->future rollout displacement pairs were found. "
            "Check that pose metadata exists and displacement_hops is compatible "
            "with the cached temporal stride."
        )

    return (
        torch.stack(z_now_pairs, dim=0),
        torch.stack(z_future_pairs, dim=0),
        torch.cat(target_disp_pairs, dim=0),
    )


def build_coverage_gain_pairs(
    z_enc: torch.Tensor,
    enc_episode_id: torch.Tensor,
    enc_obs_step: torch.Tensor,
    enc_robot_xy: torch.Tensor,
    z_rollout: torch.Tensor,
    rollout_episode_id: torch.Tensor,
    rollout_obs_step: torch.Tensor,
    rollout_robot_xy: torch.Tensor,
    coverage_gain_hops: int,
    coverage_gain_target_hops: int,
    coverage_gain_context_hops: int,
    temporal_stride: int,
    coverage_gain_radius_m: float,
    coverage_gain_densify_step_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build current-latent / rollout-snippet pairs with future coverage-gain targets."""
    if coverage_gain_hops <= 0:
        raise ValueError(f"coverage_gain_hops must be positive, got {coverage_gain_hops}")
    if coverage_gain_target_hops < coverage_gain_hops:
        raise ValueError(
            "coverage_gain_target_hops must be >= coverage_gain_hops, got "
            f"{coverage_gain_target_hops} < {coverage_gain_hops}",
        )
    if coverage_gain_context_hops < 0:
        raise ValueError(
            f"coverage_gain_context_hops must be non-negative, got {coverage_gain_context_hops}",
        )

    current_bank = dedup_pose_latent_bank(z_enc, enc_episode_id, enc_obs_step, enc_robot_xy)
    future_bank = dedup_pose_latent_bank(z_rollout, rollout_episode_id, rollout_obs_step, rollout_robot_xy)
    raw_stride = int(temporal_stride)

    z_now_pairs: list[torch.Tensor] = []
    z_future_seq_pairs: list[torch.Tensor] = []
    target_gain_pairs: list[torch.Tensor] = []

    for ep, cur_steps in current_bank.items():
        fut_steps = future_bank.get(ep)
        if not fut_steps:
            continue
        for cur_step, (z_now, xy_now) in cur_steps.items():
            snippet_steps = [cur_step + raw_stride * i for i in range(1, int(coverage_gain_hops) + 1)]
            target_steps = [cur_step + raw_stride * i for i in range(1, int(coverage_gain_target_hops) + 1)]
            if any(step not in fut_steps for step in snippet_steps):
                continue
            if any(step not in fut_steps for step in target_steps):
                continue

            context_steps = [cur_step]
            for hop in range(int(coverage_gain_context_hops), 0, -1):
                prev_step = cur_step - raw_stride * hop
                if prev_step in cur_steps:
                    context_steps.insert(-1 if context_steps else 0, prev_step)
            context_steps = sorted(set(context_steps))

            context_xy_seq = torch.stack(
                [cur_steps[step][1].to(dtype=torch.float32) for step in context_steps],
                dim=0,
            )
            if context_xy_seq.shape[0] == 0:
                context_xy_seq = xy_now.view(1, 2).to(dtype=torch.float32)

            snippet_pairs = [fut_steps[step] for step in snippet_steps]
            target_pairs = [fut_steps[step] for step in target_steps]
            future_latents = torch.stack(
                [z_future.to(dtype=torch.float32) for z_future, _xy in snippet_pairs],
                dim=0,
            )
            future_xy_seq = torch.stack(
                [xy_future.to(dtype=torch.float32) for _z_future, xy_future in target_pairs],
                dim=0,
            )
            gain_m2 = novel_path_area_gain_proxy(
                context_xy_seq,
                future_xy_seq,
                radius_m=float(coverage_gain_radius_m),
                densify_step_m=float(coverage_gain_densify_step_m),
            )

            z_now_pairs.append(z_now.to(dtype=torch.float32))
            z_future_seq_pairs.append(future_latents)
            target_gain_pairs.append(torch.tensor([gain_m2], dtype=torch.float32))

    if not z_now_pairs:
        raise ValueError(
            "No valid current->future rollout coverage-gain pairs were found. "
            "Check that pose metadata exists and coverage_gain_hops is compatible "
            "with the cached temporal stride."
        )

    return (
        torch.stack(z_now_pairs, dim=0),
        torch.stack(z_future_seq_pairs, dim=0),
        torch.cat(target_gain_pairs, dim=0),
    )


def train_displacement_head(
    args,
    z_now_pairs: torch.Tensor,
    z_future_pairs: torch.Tensor,
    target_disp_pairs: torch.Tensor,
    device,
):
    print("\n" + "=" * 60)
    print("Phase 7: Training DisplacementHead (pose-supervised mobility)")
    print("=" * 60)
    print(
        f"  Pairs: {int(target_disp_pairs.numel()):,} | "
        f"hops={int(args.displacement_hops)} | "
        f"target_mean={float(target_disp_pairs.mean().item()):.3f}m"
    )

    dataset = TensorDataset(
        z_now_pairs.to(dtype=torch.float32),
        z_future_pairs.to(dtype=torch.float32),
        target_disp_pairs.to(dtype=torch.float32),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.displacement_batch_size)),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  {len(dataloader)} batches of {max(1, int(args.displacement_batch_size))}")

    head = DisplacementHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.displacement_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.displacement_epochs)

    csv_path = os.path.join(args.log_dir, "displacement_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "pred_mean_m", "target_mean_m", "lr"])

    global_step = 0
    for epoch in range(args.displacement_epochs):
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()
        head.train()

        with make_progress(args, dataloader, desc=f"  Disp {epoch + 1}/{args.displacement_epochs}") as pbar:
            for z_now_batch, z_future_batch, target_batch in pbar:
                z_now_batch = z_now_batch.to(device, non_blocking=True)
                z_future_batch = z_future_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = head(z_now_batch, z_future_batch)
                loss = F.mse_loss(pred, target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                global_step += 1
                loss_val = float(loss.item())
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step,
                        epoch + 1,
                        f"{loss_val:.6f}",
                        f"{pred.detach().mean().item():.6f}",
                        f"{target_batch.detach().mean().item():.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        pred=f"{pred.detach().mean().item():.3f}",
                        target=f"{target_batch.detach().mean().item():.3f}",
                    )

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.6f} | time={time.time() - t0:.0f}s")

    torch.save(
        {"displacement_head_state_dict": head.state_dict(), "epoch": args.displacement_epochs},
        os.path.join(args.out_dir, "displacement_head.pt"),
    )
    print("  Displacement head training complete.")
    return head


def evaluate_displacement_head(
    args,
    head: nn.Module,
    z_now_pairs: torch.Tensor,
    z_future_pairs: torch.Tensor,
    target_disp_pairs: torch.Tensor,
    device,
) -> dict[str, float] | None:
    if target_disp_pairs.numel() == 0:
        return None

    dataset = TensorDataset(z_now_pairs, z_future_pairs, target_disp_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    preds_all: list[torch.Tensor] = []
    target_all: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for z_now_batch, z_future_batch, target_batch in dataloader:
            z_now_batch = z_now_batch.to(device, non_blocking=True)
            z_future_batch = z_future_batch.to(device, non_blocking=True)
            pred = head(z_now_batch, z_future_batch).float().cpu()
            preds_all.append(pred)
            target_all.append(target_batch.float().cpu())

    pred = torch.cat(preds_all)
    target = torch.cat(target_all)
    mse = float((pred - target).square().mean().item())
    metrics = {
        "mse": mse,
        "rmse": math.sqrt(max(0.0, mse)),
        "mae": float((pred - target).abs().mean().item()),
        "pearson": pearson_corrcoef(pred, target),
        "pred_mean_m": float(pred.mean().item()),
        "target_mean_m": float(target.mean().item()),
        "n_eval": int(target.numel()),
    }
    print(
        "  Held-out displacement | "
        f"n={metrics['n_eval']:,} "
        f"rmse={metrics['rmse']:.4f}m "
        f"mae={metrics['mae']:.4f}m "
        f"pearson={metrics['pearson']:.3f} "
        f"pred={metrics['pred_mean_m']:.3f}m "
        f"target={metrics['target_mean_m']:.3f}m"
    )
    return metrics


def train_coverage_gain_head(
    args,
    z_now_pairs: torch.Tensor,
    z_future_seq_pairs: torch.Tensor,
    target_gain_pairs: torch.Tensor,
    device,
):
    print("\n" + "=" * 60)
    print("Phase 8: Training CoverageGainHead (sequence-level coverage value)")
    print("=" * 60)
    print(
        f"  Pairs: {int(target_gain_pairs.numel()):,} | "
        f"hops={int(args.coverage_gain_hops)} | "
        f"target_hops={int(args.coverage_gain_target_hops)} | "
        f"context={int(args.coverage_gain_context_hops)} | "
        f"target_mean={float(target_gain_pairs.mean().item()):.4f}m^2"
    )

    dataset = TensorDataset(
        z_now_pairs.to(dtype=torch.float32),
        z_future_seq_pairs.to(dtype=torch.float32),
        target_gain_pairs.to(dtype=torch.float32),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.coverage_gain_batch_size)),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  {len(dataloader)} batches of {max(1, int(args.coverage_gain_batch_size))}")

    head = CoverageGainHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.coverage_gain_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.coverage_gain_epochs)

    csv_path = os.path.join(args.log_dir, "coverage_gain_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "step", "epoch", "loss", "pred_mean_m2", "target_mean_m2", "lr",
            ])

    global_step = 0
    for epoch in range(args.coverage_gain_epochs):
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()
        head.train()

        with make_progress(args, dataloader, desc=f"  CovGain {epoch + 1}/{args.coverage_gain_epochs}") as pbar:
            for z_now_batch, z_future_seq_batch, target_batch in pbar:
                z_now_batch = z_now_batch.to(device, non_blocking=True)
                z_future_seq_batch = z_future_seq_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = head(z_now_batch, z_future_seq_batch)
                loss = F.mse_loss(pred, target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                global_step += 1
                loss_val = float(loss.item())
                epoch_loss += loss_val
                epoch_n += 1

                with open(csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step,
                        epoch + 1,
                        f"{loss_val:.6f}",
                        f"{pred.detach().mean().item():.6f}",
                        f"{target_batch.detach().mean().item():.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        pred=f"{pred.detach().mean().item():.4f}",
                        target=f"{target_batch.detach().mean().item():.4f}",
                    )

        scheduler.step()
        avg = epoch_loss / max(1, epoch_n)
        print(f"  Epoch {epoch + 1} | avg_loss={avg:.6f} | time={time.time() - t0:.0f}s")

    torch.save(
        {"coverage_gain_head_state_dict": head.state_dict(), "epoch": args.coverage_gain_epochs},
        os.path.join(args.out_dir, "coverage_gain_head.pt"),
    )
    print("  Coverage-gain head training complete.")
    return head


def evaluate_coverage_gain_head(
    args,
    head: nn.Module,
    z_now_pairs: torch.Tensor,
    z_future_seq_pairs: torch.Tensor,
    target_gain_pairs: torch.Tensor,
    device,
) -> dict[str, float] | None:
    if target_gain_pairs.numel() == 0:
        return None

    dataset = TensorDataset(z_now_pairs, z_future_seq_pairs, target_gain_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    preds_all: list[torch.Tensor] = []
    target_all: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for z_now_batch, z_future_seq_batch, target_batch in dataloader:
            z_now_batch = z_now_batch.to(device, non_blocking=True)
            z_future_seq_batch = z_future_seq_batch.to(device, non_blocking=True)
            pred = head(z_now_batch, z_future_seq_batch).float().cpu()
            preds_all.append(pred)
            target_all.append(target_batch.float().cpu())

    pred = torch.cat(preds_all)
    target = torch.cat(target_all)
    mse = float((pred - target).square().mean().item())
    metrics = {
        "mse": mse,
        "rmse": math.sqrt(max(0.0, mse)),
        "mae": float((pred - target).abs().mean().item()),
        "pearson": pearson_corrcoef(pred, target),
        "pred_mean_m2": float(pred.mean().item()),
        "target_mean_m2": float(target.mean().item()),
        "n_eval": int(target.numel()),
    }
    print(
        "  Held-out coverage gain | "
        f"n={metrics['n_eval']:,} "
        f"rmse={metrics['rmse']:.4f}m^2 "
        f"mae={metrics['mae']:.4f}m^2 "
        f"pearson={metrics['pearson']:.3f} "
        f"pred={metrics['pred_mean_m2']:.4f}m^2 "
        f"target={metrics['target_mean_m2']:.4f}m^2"
    )
    return metrics


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
    enc_episode_id = latent_views["encoder"]["episode_id"]
    enc_obs_step = latent_views["encoder"]["obs_step"]
    enc_robot_xy = latent_views["encoder"]["robot_xy"]
    z_rollout = latent_views["rollout"]["z"]
    safety_target_rollout = latent_views["rollout"]["safety_target"]
    rollout_collisions = latent_views["rollout"]["collisions"]
    rollout_episode_id = latent_views["rollout"]["episode_id"]
    rollout_obs_step = latent_views["rollout"]["obs_step"]
    rollout_robot_xy = latent_views["rollout"]["robot_xy"]
    print(
        f"Using rollout-view latents for safety/exploration: {tuple(z_rollout.shape)} | "
        f"encoder-view latents for goal/progress: {tuple(z_enc.shape)}"
    )

    holdout_metrics: dict[str, dict[str, float]] = {}
    z_rollout_train = z_rollout
    safety_target_train = safety_target_rollout
    rollout_collisions_eval = torch.empty((0,), dtype=torch.float32)
    z_rollout_eval = torch.empty((0, z_rollout.shape[-1]), dtype=z_rollout.dtype)
    safety_target_eval = torch.empty((0,), dtype=safety_target_rollout.dtype)
    z_enc_train = z_enc
    enc_episode_id_train = enc_episode_id
    enc_obs_step_train = enc_obs_step
    enc_robot_xy_train = enc_robot_xy
    z_enc_eval = torch.empty((0, z_enc.shape[-1]), dtype=z_enc.dtype)
    enc_episode_id_eval = torch.empty((0,), dtype=enc_episode_id.dtype)
    enc_obs_step_eval = torch.empty((0,), dtype=enc_obs_step.dtype)
    enc_robot_xy_eval = torch.empty((0, 2), dtype=enc_robot_xy.dtype)
    rollout_episode_id_train = rollout_episode_id
    rollout_obs_step_train = rollout_obs_step
    rollout_robot_xy_train = rollout_robot_xy
    rollout_episode_id_eval = torch.empty((0,), dtype=rollout_episode_id.dtype)
    rollout_obs_step_eval = torch.empty((0,), dtype=rollout_obs_step.dtype)
    rollout_robot_xy_eval = torch.empty((0, 2), dtype=rollout_robot_xy.dtype)
    if not args.skip_eval:
        split = build_rollout_holdout_masks(
            rollout_episode_id,
            holdout_fraction=float(args.eval_holdout_fraction),
            seed=int(args.eval_seed),
        )
        if split is None:
            print("Held-out eval: disabled (not enough valid rollout episodes for a split).")
        else:
            train_mask, eval_mask, split_info = split
            z_rollout_train = z_rollout[train_mask]
            safety_target_train = safety_target_rollout[train_mask]
            rollout_episode_id_train = rollout_episode_id[train_mask]
            rollout_obs_step_train = rollout_obs_step[train_mask]
            rollout_robot_xy_train = rollout_robot_xy[train_mask]
            z_rollout_eval = z_rollout[eval_mask]
            safety_target_eval = safety_target_rollout[eval_mask]
            rollout_collisions_eval = rollout_collisions[eval_mask]
            rollout_episode_id_eval = rollout_episode_id[eval_mask]
            rollout_obs_step_eval = rollout_obs_step[eval_mask]
            rollout_robot_xy_eval = rollout_robot_xy[eval_mask]
            eval_eps = torch.unique(rollout_episode_id_eval[rollout_episode_id_eval >= 0], sorted=True)
            if eval_eps.numel() > 0:
                enc_valid = enc_episode_id >= 0
                enc_eval_mask = enc_valid & torch.isin(enc_episode_id, eval_eps)
                enc_train_mask = enc_valid & (~enc_eval_mask)
                if torch.any(enc_train_mask):
                    z_enc_train = z_enc[enc_train_mask]
                    enc_episode_id_train = enc_episode_id[enc_train_mask]
                    enc_obs_step_train = enc_obs_step[enc_train_mask]
                    enc_robot_xy_train = enc_robot_xy[enc_train_mask]
                if torch.any(enc_eval_mask):
                    z_enc_eval = z_enc[enc_eval_mask]
                    enc_episode_id_eval = enc_episode_id[enc_eval_mask]
                    enc_obs_step_eval = enc_obs_step[enc_eval_mask]
                    enc_robot_xy_eval = enc_robot_xy[enc_eval_mask]
            print(
                "Held-out eval split: "
                f"episodes train/eval={split_info['n_train_episodes']}/{split_info['n_eval_episodes']} "
                f"samples train/eval={split_info['n_train_samples']:,}/{split_info['n_eval_samples']:,}"
            )

    # Phase 2: safety head
    safety_head = train_safety_head(args, z_rollout_train, safety_target_train, device)
    if z_rollout_eval.numel() > 0:
        metrics = evaluate_safety_head(
            args,
            safety_head,
            z_rollout_eval,
            safety_target_eval,
            rollout_collisions_eval,
            device,
        )
        if metrics is not None:
            holdout_metrics["safety"] = metrics

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
        exploration = train_exploration(args, z_rollout_train, device)

    # Phase 6: place head
    place_head = None
    place_snippet_len = int(
        args.place_snippet_len if args.place_snippet_len is not None else max(1, args.seq_len - 1)
    )
    if not args.skip_place:
        rollout_snippets, snippet_episode_id, snippet_start, snippet_xy = build_rollout_snippet_bank(
            z_rollout_train,
            rollout_episode_id_train,
            rollout_obs_step_train,
            snippet_len=place_snippet_len,
            temporal_stride=args.temporal_stride,
            robot_xy=rollout_robot_xy_train,
        )
        place_head = train_place_head(
            args,
            rollout_snippets,
            snippet_episode_id,
            snippet_start,
            snippet_xy,
            device,
        )
        if rollout_episode_id_eval.numel() > 0:
            try:
                eval_snippets, eval_snippet_epid, _eval_snippet_start, eval_snippet_xy = build_rollout_snippet_bank(
                    z_rollout_eval,
                    rollout_episode_id_eval,
                    rollout_obs_step_eval,
                    snippet_len=place_snippet_len,
                    temporal_stride=args.temporal_stride,
                    robot_xy=rollout_robot_xy_eval,
                )
            except ValueError:
                eval_snippets = None
            if eval_snippets is not None:
                metrics = evaluate_place_head(
                    args,
                    place_head,
                    eval_snippets,
                    eval_snippet_epid,
                    eval_snippet_xy,
                    device,
                )
                if metrics is not None:
                    holdout_metrics["place"] = metrics

    # Phase 7: displacement head
    displacement_head = None
    if not args.skip_displacement:
        try:
            z_now_disp_train, z_future_disp_train, target_disp_train = build_displacement_pairs(
                z_enc_train,
                enc_episode_id_train,
                enc_obs_step_train,
                enc_robot_xy_train,
                z_rollout_train,
                rollout_episode_id_train,
                rollout_obs_step_train,
                rollout_robot_xy_train,
                displacement_hops=int(args.displacement_hops),
                temporal_stride=int(args.temporal_stride),
            )
        except ValueError as exc:
            print(f"  Skipping displacement head: {exc}")
        else:
            displacement_head = train_displacement_head(
                args,
                z_now_disp_train,
                z_future_disp_train,
                target_disp_train,
                device,
            )
            if z_enc_eval.numel() > 0 and z_rollout_eval.numel() > 0:
                try:
                    z_now_disp_eval, z_future_disp_eval, target_disp_eval = build_displacement_pairs(
                        z_enc_eval,
                        enc_episode_id_eval,
                        enc_obs_step_eval,
                        enc_robot_xy_eval,
                        z_rollout_eval,
                        rollout_episode_id_eval,
                        rollout_obs_step_eval,
                        rollout_robot_xy_eval,
                        displacement_hops=int(args.displacement_hops),
                        temporal_stride=int(args.temporal_stride),
                    )
                except ValueError:
                    z_now_disp_eval = None
                if z_now_disp_eval is not None:
                    metrics = evaluate_displacement_head(
                        args,
                        displacement_head,
                        z_now_disp_eval,
                        z_future_disp_eval,
                        target_disp_eval,
                        device,
                    )
                    if metrics is not None:
                        holdout_metrics["displacement"] = metrics

    # Phase 8: sequence-level coverage-gain head
    coverage_gain_head = None
    if not args.skip_coverage_gain:
        try:
            z_now_cov_train, z_future_cov_train, target_cov_train = build_coverage_gain_pairs(
                z_enc_train,
                enc_episode_id_train,
                enc_obs_step_train,
                enc_robot_xy_train,
                z_rollout_train,
                rollout_episode_id_train,
                rollout_obs_step_train,
                rollout_robot_xy_train,
                coverage_gain_hops=int(args.coverage_gain_hops),
                coverage_gain_target_hops=int(args.coverage_gain_target_hops),
                coverage_gain_context_hops=int(args.coverage_gain_context_hops),
                temporal_stride=int(args.temporal_stride),
                coverage_gain_radius_m=float(args.coverage_gain_radius_m),
                coverage_gain_densify_step_m=float(args.coverage_gain_densify_step_m),
            )
        except ValueError as exc:
            print(f"  Skipping coverage-gain head: {exc}")
        else:
            coverage_gain_head = train_coverage_gain_head(
                args,
                z_now_cov_train,
                z_future_cov_train,
                target_cov_train,
                device,
            )
            if z_enc_eval.numel() > 0 and z_rollout_eval.numel() > 0:
                try:
                    z_now_cov_eval, z_future_cov_eval, target_cov_eval = build_coverage_gain_pairs(
                        z_enc_eval,
                        enc_episode_id_eval,
                        enc_obs_step_eval,
                        enc_robot_xy_eval,
                        z_rollout_eval,
                        rollout_episode_id_eval,
                        rollout_obs_step_eval,
                        rollout_robot_xy_eval,
                        coverage_gain_hops=int(args.coverage_gain_hops),
                        coverage_gain_target_hops=int(args.coverage_gain_target_hops),
                        coverage_gain_context_hops=int(args.coverage_gain_context_hops),
                        temporal_stride=int(args.temporal_stride),
                        coverage_gain_radius_m=float(args.coverage_gain_radius_m),
                        coverage_gain_densify_step_m=float(args.coverage_gain_densify_step_m),
                    )
                except ValueError:
                    z_now_cov_eval = None
                if z_now_cov_eval is not None:
                    metrics = evaluate_coverage_gain_head(
                        args,
                        coverage_gain_head,
                        z_now_cov_eval,
                        z_future_cov_eval,
                        target_cov_eval,
                        device,
                    )
                    if metrics is not None:
                        holdout_metrics["coverage_gain"] = metrics

    # Save combined TrajectoryScorer checkpoint
    scorer_path = os.path.join(args.out_dir, "trajectory_scorer.pt")
    scorer_data = {
        "safety_head": safety_head.state_dict(),
        "goal_head": goal_head.state_dict() if goal_head is not None else None,
        "progress_head": progress_head.state_dict() if progress_head is not None else None,
        "exploration": exploration.state_dict() if exploration is not None else None,
        "place_head": place_head.state_dict() if place_head is not None else None,
        "displacement_head": displacement_head.state_dict() if displacement_head is not None else None,
        "coverage_gain_head": coverage_gain_head.state_dict() if coverage_gain_head is not None else None,
        "goal_weight": args.goal_weight,
        "progress_weight": args.progress_weight,
        "exploration_weight": args.exploration_weight,
        "displacement_weight": args.displacement_weight,
        "coverage_gain_weight": args.coverage_gain_weight,
        "displacement_hops": args.displacement_hops,
        "coverage_gain_hops": args.coverage_gain_hops,
        "coverage_gain_target_hops": args.coverage_gain_target_hops,
        "coverage_gain_context_hops": args.coverage_gain_context_hops,
        "coverage_gain_radius_m": args.coverage_gain_radius_m,
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
        "displacement_latent_source": "enc_projector_to_pred_projector_teacher_forced",
        "coverage_gain_latent_source": "enc_projector_to_pred_projector_teacher_forced_snippet",
        "place_supervision": (
            "pose_xy_same_episode"
            if args.raw_data_dir is not None
            else "temporal_proxy"
        ),
        "displacement_supervision": (
            "pose_xy_delta_same_episode"
            if args.raw_data_dir is not None
            else None
        ),
        "coverage_gain_supervision": (
            "pose_xy_novel_path_area_proxy_same_episode"
            if args.raw_data_dir is not None
            else None
        ),
        "goal_latent_source": "enc_projector",
        "progress_latent_source": "enc_projector",
        "cache_version": CACHE_VERSION,
        "holdout_metrics": holdout_metrics,
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
    if holdout_metrics:
        torch.save(holdout_metrics, os.path.join(args.out_dir, "heldout_metrics.pt"))
    print(f"\nTrajectoryScorer checkpoint saved: {scorer_path}")
    print("All training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
