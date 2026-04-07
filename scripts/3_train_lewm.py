#!/usr/bin/env python3
"""LeWorldModel training loop.

Replaces the EMA student-teacher JEPA training with the LeWM approach:
  - Single encoder (no target encoder, no EMA).
  - Transformer predictor with AdaLN action conditioning.
  - Two-term loss: MSE prediction + λ·SIGReg anti-collapse regulariser.
  - All parameters optimised jointly end-to-end.

Usage:
    python scripts/3_train_lewm.py --data_dir jepa_final_dataset_224 --batch_size 256 --sigreg_lambda 0.045
    python scripts/3_train_lewm.py --data_dir jepa_final_dataset_224 --batch_size 256 --sigreg_lambda 0.045
    --resume_from lewm_checkpoints/step_3000.pt
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lewm.models import LeWorldModel
from lewm.data import StreamingJEPADataset
from lewm.checkpoint_utils import clean_state_dict

torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeWorldModel (SIGReg JEPA)")
    p.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--temporal_stride", type=int, default=1,
                   help="Raw-step spacing between model observations.")
    p.add_argument("--action_block_size", type=int, default=None,
                   help="Raw-step action-block size per model step. Defaults to --temporal_stride.")
    p.add_argument("--command_representation", type=str, default="mean_scaled",
                   choices=["mean_scaled", "mean_active", "active_block"],
                   help="How each action block is represented for the predictor.")
    p.add_argument("--command_latency", type=int, default=2,
                   help="Deterministic command delay used to reconstruct executed commands.")
    p.add_argument("--window_stride", type=int, default=None,
                   help="Raw-step spacing between sequence starts. Defaults to seq_len * temporal_stride.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=12,
                   help="DataLoader worker count for streamed HDF5 batches.")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Number of batches prefetched per worker.")
    p.add_argument("--warmup_steps", type=int, default=1000,
                    help="Number of linear LR warmup steps before cosine decay.")
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="lewm_checkpoints_keyframe_exec")
    p.add_argument("--log_dir", type=str, default="lewm_logs_keyframe_exec")
    # LeWM-specific hypers
    p.add_argument("--sigreg_lambda", type=float, default=0.09,
                    help="Weight λ for SIGReg regularisation (only tunable hyper).")
    p.add_argument("--sigreg_projections", type=int, default=1024,
                    help="Number of random projections M in SIGReg.")
    p.add_argument("--sigreg_knots", type=int, default=17,
                    help="Number of quadrature knots for Epps-Pulley test.")
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--pred_layers", type=int, default=6)
    p.add_argument("--pred_heads", type=int, default=16)
    p.add_argument("--pred_dim_head", type=int, default=64)
    p.add_argument("--pred_mlp_dim", type=int, default=2048)
    p.add_argument("--pred_dropout", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=None,
                   help="Input image size. Defaults to the rendered dataset size.")
    p.add_argument("--patch_size", type=int, default=None,
                   help="ViT patch size. Defaults to 14 for 224px data and 4 for 64px data.")
    p.add_argument("--use_proprio", action="store_true",
                   help="Fuse proprioception instead of the paper's vision-only encoder.")
    p.add_argument("--skip_eval", action="store_true",
                   help="Disable held-out scene evaluation for the base world model.")
    p.add_argument("--eval_holdout_fraction", type=float, default=0.10,
                   help="Fraction of scene groups held out for world-model validation.")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--no_progress", action="store_true",
                   help="Disable animated tqdm progress bars and emit plain logs only.")
    return p.parse_args()


# --------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------- #

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    *,
    epoch_completed: bool = False,
    metadata: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "epoch_completed": epoch_completed,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)


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


def build_transition_mask(
    dones: torch.Tensor,
    labels: dict | None,
    device: torch.device,
) -> torch.Tensor:
    if labels is not None and "episode_id" in labels:
        episode_id = labels["episode_id"].to(device, non_blocking=True)
        return episode_id[:, :-1] == episode_id[:, 1:]
    return ~dones[:, 1:]


def split_scene_ids(
    scene_ids: list[int],
    holdout_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]] | None:
    if holdout_fraction <= 0.0 or len(scene_ids) < 2:
        return None
    n_scenes = len(scene_ids)
    n_eval = max(1, int(round(float(holdout_fraction) * n_scenes)))
    n_eval = min(n_eval, n_scenes - 1)
    if n_eval <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_scenes, generator=generator).tolist()
    eval_ids = {int(scene_ids[idx]) for idx in perm[:n_eval]}
    train_ids = {int(scene_id) for scene_id in scene_ids if int(scene_id) not in eval_ids}
    if not train_ids or not eval_ids:
        return None
    return train_ids, eval_ids


# --------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    print(f"Initialising LeWorldModel training on {device}")
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
    cmd_dim = 3 * action_block_size if args.command_representation == "active_block" else 3
    print(
        "Command representation: "
        f"{args.command_representation} (latency={int(args.command_latency)}, cmd_dim={cmd_dim})"
    )

    # ---- Dataset / DataLoader ----------------------------------------
    num_workers = max(1, int(args.num_workers))
    dataset_kwargs = dict(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        temporal_stride=args.temporal_stride,
        action_block_size=args.action_block_size,
        command_representation=args.command_representation,
        command_latency=args.command_latency,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=num_workers,
    )
    dataset = StreamingJEPADataset(**dataset_kwargs)
    train_dataset = dataset
    eval_dataset = None
    split = None if args.skip_eval else split_scene_ids(
        sorted(set(int(scene_id) for scene_id in dataset._scene_ids.values())),
        holdout_fraction=float(args.eval_holdout_fraction),
        seed=int(args.eval_seed),
    )
    if split is None:
        if args.skip_eval:
            print("Held-out scene eval: disabled (--skip_eval).")
        else:
            print("Held-out scene eval: disabled (not enough scenes for a split).")
    else:
        train_scene_ids, eval_scene_ids = split
        train_dataset = StreamingJEPADataset(
            **dataset_kwargs,
            allowed_scene_ids=train_scene_ids,
        )
        eval_dataset = StreamingJEPADataset(
            **dataset_kwargs,
            allowed_scene_ids=eval_scene_ids,
        )
        print(
            "Held-out scene eval split: "
            f"scenes train/eval={len(train_scene_ids)}/{len(eval_scene_ids)} "
            f"windows train/eval={len(train_dataset._all_indices):,}/{len(eval_dataset._all_indices):,}"
        )

    channels, height, width = dataset.vision_shape
    if channels != 3 or height != width:
        raise ValueError(f"Expected square RGB inputs, got shape {dataset.vision_shape}")

    image_size = args.image_size if args.image_size is not None else height
    if image_size != height:
        raise ValueError(
            f"--image_size={image_size} does not match dataset resolution {height}"
        )

    if args.patch_size is not None:
        patch_size = args.patch_size
    elif image_size == 224:
        patch_size = 14
    elif image_size == 64:
        patch_size = 4
    else:
        raise ValueError(
            f"No default patch size for image_size={image_size}; pass --patch_size explicitly."
        )

    dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=True,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=max(1, int(args.prefetch_factor)),
            persistent_workers=True,
        )

    # ---- Model -------------------------------------------------------
    model = LeWorldModel(
        latent_dim=args.latent_dim,
        cmd_dim=cmd_dim,
        pred_layers=args.pred_layers,
        pred_heads=args.pred_heads,
        pred_dim_head=args.pred_dim_head,
        pred_mlp_dim=args.pred_mlp_dim,
        pred_dropout=args.pred_dropout,
        max_seq_len=args.seq_len,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
        sigreg_knots=args.sigreg_knots,
        image_size=image_size,
        patch_size=patch_size,
        use_proprio=args.use_proprio,
    ).to(device)
    checkpoint_meta = {
        "cmd_dim": int(cmd_dim),
        "command_representation": args.command_representation,
        "command_latency": int(args.command_latency),
        "temporal_stride": int(args.temporal_stride),
        "action_block_size": int(action_block_size),
        "window_stride": int(window_stride),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "use_proprio": bool(args.use_proprio),
        "max_seq_len": int(args.seq_len),
        "eval_holdout_fraction": float(args.eval_holdout_fraction),
        "eval_split_group": "scene",
    }

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    start_epoch = 0
    global_step = 0

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        cleaned_sd = clean_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(cleaned_sd)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        epoch_completed = ckpt.get("epoch_completed")
        if epoch_completed is True:
            start_epoch += 1
        elif epoch_completed is None:
            base = os.path.basename(args.resume_from)
            if base.startswith("epoch_"):
                start_epoch += 1

    model = torch.compile(model)

    # ---- Optimizer (ALL parameters — end-to-end) ---------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume_from and os.path.exists(args.resume_from):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Restored optimiser state. Resuming at epoch {start_epoch}, step {global_step}.")
        except Exception as e:
            print(f"Warning: could not restore optimiser state: {e}")

    # ---- Logging setup -----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, "training_metrics.csv")
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "total_loss", "pred_loss", "sigreg_loss",
                "lr", "z_proj_std", "grad_norm",
            ])
    eval_csv_path = os.path.join(args.log_dir, "eval_metrics.csv")
    if eval_dataloader is not None and not os.path.exists(eval_csv_path):
        with open(eval_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "global_step", "total_loss", "pred_loss",
                "sigreg_loss", "z_proj_std",
            ])
    multistep_csv_path = os.path.join(args.log_dir, "multistep_eval_metrics.csv")
    if eval_dataloader is not None and not os.path.exists(multistep_csv_path):
        with open(multistep_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            step_headers = [f"mse_step_{s}" for s in range(1, args.seq_len)]
            writer.writerow(["epoch", "global_step"] + step_headers + ["n_seqs"])

    def run_eval(epoch_idx: int) -> None:
        if eval_dataloader is None:
            return
        model.eval()
        total_loss_sum = 0.0
        pred_loss_sum = 0.0
        sigreg_loss_sum = 0.0
        z_std_sum = 0.0
        n_batches = 0
        with torch.inference_mode():
            for batch in eval_dataloader:
                if len(batch) == 6:
                    vision, proprio, cmds, dones, collisions, labels = batch
                else:
                    vision, proprio, cmds, dones, collisions = batch
                    labels = None

                vision = vision.to(device, non_blocking=True).float().div_(255.0)
                proprio = proprio.to(device, non_blocking=True)
                cmds = cmds.to(device, non_blocking=True)
                dones = dones.to(device, non_blocking=True)
                mask = build_transition_mask(dones, labels, device)

                with autocast(amp_device, dtype=torch.bfloat16):
                    out = model(vision, proprio, cmds, mask=mask)

                total_loss_sum += float(out["loss"].item())
                pred_loss_sum += float(out["pred_loss"].item())
                sigreg_loss_sum += float(out["sigreg_loss"].item())
                z_std_sum += float(out["z_proj_std"].item())
                n_batches += 1

        if n_batches == 0:
            print(f"Eval epoch {epoch_idx + 1}: no held-out batches.")
            model.train()
            return

        metrics = {
            "total_loss": total_loss_sum / n_batches,
            "pred_loss": pred_loss_sum / n_batches,
            "sigreg_loss": sigreg_loss_sum / n_batches,
            "z_proj_std": z_std_sum / n_batches,
        }
        with open(eval_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_idx + 1,
                global_step,
                f"{metrics['total_loss']:.6f}",
                f"{metrics['pred_loss']:.6f}",
                f"{metrics['sigreg_loss']:.6f}",
                f"{metrics['z_proj_std']:.6f}",
            ])
        print(
            f"Held-out eval epoch {epoch_idx + 1} | "
            f"loss={metrics['total_loss']:.4f} "
            f"pred={metrics['pred_loss']:.4f} "
            f"sig={metrics['sigreg_loss']:.4f} "
            f"z_std={metrics['z_proj_std']:.3f}"
        )
        model.train()

    def run_multistep_eval(epoch_idx: int) -> None:
        """Autoregressive multi-step prediction error on held-out scenes.

        Runs the predictor step-by-step using its own outputs (not ground-truth
        encoder latents) and reports per-step MSE in projected space.  This
        directly measures the regime the CEM planner operates in.
        """
        if eval_dataloader is None:
            return
        model.eval()
        max_steps = args.seq_len - 1
        step_mse_sums = [0.0] * max_steps
        n_seqs = 0
        with torch.inference_mode():
            for batch in eval_dataloader:
                if len(batch) == 6:
                    vision, proprio, cmds, dones, collisions, labels = batch
                else:
                    vision, proprio, cmds, dones, collisions = batch
                    labels = None

                vision = vision.to(device, non_blocking=True).float().div_(255.0)
                proprio = proprio.to(device, non_blocking=True)
                cmds = cmds.to(device, non_blocking=True)

                mask = build_transition_mask(dones, labels, device)

                B, T = vision.shape[:2]
                if T < 2:
                    continue

                # Unwrap torch.compile / DataParallel
                _model = model
                if hasattr(_model, "_orig_mod"):
                    _model = _model._orig_mod
                if hasattr(_model, "module"):
                    _model = _model.module

                with autocast(amp_device, dtype=torch.bfloat16):
                    z_raw, z_proj = _model.encode_seq(vision, proprio)

                z_proj_gt = z_proj.float()
                z_raw_gt = z_raw.float()

                # Autoregressive rollout from first frame
                z_ar = z_raw_gt[:, 0:1, :]
                for step in range(max_steps):
                    t = step + 1
                    if t >= T:
                        break
                    z_ctx = z_ar
                    a_ctx = cmds[:, :t, :]
                    if z_ctx.shape[1] > args.seq_len:
                        z_ctx = z_ctx[:, -args.seq_len:]
                        a_ctx = a_ctx[:, -args.seq_len:]
                    with autocast(amp_device, dtype=torch.bfloat16):
                        pred_raw = _model.predictor(z_ctx, a_ctx)
                    z_next_raw = pred_raw[:, -1:, :].float()
                    z_ar = torch.cat([z_ar, z_next_raw], dim=1)

                    with autocast(amp_device, dtype=torch.bfloat16):
                        z_next_proj = _model.pred_projector(z_next_raw.reshape(B, -1)).reshape(B, 1, -1)
                    z_next_proj = z_next_proj.float()

                    per_sample = (z_next_proj[:, 0] - z_proj_gt[:, t]).square().mean(dim=-1)
                    if mask is not None and t - 1 < mask.shape[1]:
                        valid = mask[:, :t].all(dim=1)
                        n_valid = valid.float().sum().clamp(min=1.0)
                        step_mse_sums[step] += float((per_sample * valid.float()).sum() / n_valid)
                    else:
                        step_mse_sums[step] += float(per_sample.mean())
                    n_seqs += 1 if step == 0 else 0

        if n_seqs == 0:
            model.train()
            return

        n_batches = max(1, n_seqs)
        with open(multistep_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [epoch_idx + 1, global_step]
            row += [f"{step_mse_sums[s] / n_batches:.6f}" for s in range(max_steps)]
            row.append(n_seqs)
            writer.writerow(row)
        step_strs = " ".join(
            f"s{s+1}={step_mse_sums[s] / n_batches:.4f}" for s in range(max_steps)
        )
        print(f"Multi-step eval epoch {epoch_idx + 1} | {step_strs}")
        model.train()

    # ---- Training loop -----------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()

        epoch_loss_sum = 0.0
        epoch_batches = 0
        t_epoch_start = time.time()

        with make_progress(args, dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for batch in pbar:
                # Dataset returns 5-tuple (old) or 6-tuple (new, with labels dict)
                if len(batch) == 6:
                    vision, proprio, cmds, dones, collisions, labels = batch
                else:
                    vision, proprio, cmds, dones, collisions = batch
                    labels = None

                vision = vision.to(device, non_blocking=True).float().div_(255.0)
                proprio = proprio.to(device, non_blocking=True)
                cmds = cmds.to(device, non_blocking=True)
                dones = dones.to(device, non_blocking=True)
                collisions = collisions.to(device, non_blocking=True)

                # Keep soft-collision transitions in the loss so the predictor
                # learns wall-contact dynamics; only reset/teleport boundaries are masked.
                mask = build_transition_mask(dones, labels, device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(amp_device, dtype=torch.bfloat16):
                    out = model(vision, proprio, cmds, mask=mask)

                total_loss = out["loss"]
                pred_loss_val = out["pred_loss"].item()
                sigreg_loss_val = out["sigreg_loss"].item()
                z_std = out["z_proj_std"].item()

                # Warmup: linearly ramp LR from near-zero to args.lr over warmup_steps
                if global_step < args.warmup_steps:
                    warmup_lr = args.lr * (global_step + 1) / args.warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip,
                ).item()

                if not torch.isfinite(total_loss) or not math.isfinite(grad_norm):
                    progress_write(
                        "  WARNING: "
                        f"non-finite loss={total_loss.item():.4f} or "
                        f"grad_norm={grad_norm:.4f} at step {global_step}, "
                        "skipping update.",
                        pbar,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                # NOTE: No EMA update — that's the whole point of LeWM.

                # ---- Collapse monitoring ---------------------------------
                if z_std < 0.1:
                    progress_write(
                        f"  WARNING: z_proj_std={z_std:.4f} < 0.1; possible collapse.",
                        pbar,
                    )

                # ---- Logging ---------------------------------------------
                global_step += 1
                current_lr = optimizer.param_groups[0]["lr"]
                loss_val = total_loss.item()
                epoch_loss_sum += loss_val
                epoch_batches += 1

                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_step,
                        epoch + 1,
                        f"{loss_val:.6f}",
                        f"{pred_loss_val:.6f}",
                        f"{sigreg_loss_val:.6f}",
                        f"{current_lr:.2e}",
                        f"{z_std:.6f}",
                        f"{grad_norm:.4f}",
                    ])

                if global_step % 5 == 0:
                    pbar.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "pred": f"{pred_loss_val:.4f}",
                        "sig": f"{sigreg_loss_val:.4f}",
                        "z_std": f"{z_std:.3f}",
                        "gnorm": f"{grad_norm:.3f}",
                        "lr": f"{current_lr:.1e}",
                    })

                # ---- Intra-epoch checkpoint ------------------------------
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.out_dir, f"step_{global_step}.pt")
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        epoch_completed=False,
                        metadata=checkpoint_meta,
                    )
                    progress_write(f"  Checkpoint saved: {ckpt_path}", pbar)
                    import glob as _glob
                    step_ckpts = sorted(_glob.glob(os.path.join(args.out_dir, "step_*.pt")))
                    for old in step_ckpts[:-3]:
                        os.remove(old)

        # ---- End of epoch --------------------------------------------
        scheduler.step()

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = time.time() - t_epoch_start
        print(
            f"Epoch {epoch + 1} complete | "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"time={elapsed:.0f}s"
        )
        run_eval(epoch)
        run_multistep_eval(epoch)

        epoch_ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch + 1}.pt")
        save_checkpoint(
            epoch_ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            epoch_completed=True,
            metadata=checkpoint_meta,
        )
        print(f"  Epoch checkpoint saved: {epoch_ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
