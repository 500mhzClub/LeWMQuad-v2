#!/usr/bin/env python3
"""Export short MP4 clips around collision events from rendered HDF5 chunks.

Usage:
    python scripts/export_collision_clips.py \
        --data_dir runs/wall_collision_smoke/rendered \
        --out_dir runs/wall_collision_smoke/collision_clips
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CollisionRun:
    file_path: str
    env_idx: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def discover_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*_rgb.h5")))
    if not files:
        raise FileNotFoundError(f"No rendered HDF5 files found in {data_dir}")
    return files


def find_collision_runs(collisions: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (env_idx, start, end) for contiguous collision runs."""
    runs: list[tuple[int, int, int]] = []
    n_envs, steps = collisions.shape
    for env_idx in range(n_envs):
        active = collisions[env_idx].astype(bool)
        start = None
        for step in range(steps):
            if active[step] and start is None:
                start = step
            elif not active[step] and start is not None:
                runs.append((env_idx, start, step))
                start = None
        if start is not None:
            runs.append((env_idx, start, steps))
    return runs


def collect_candidates(files: list[str]) -> list[CollisionRun]:
    candidates: list[CollisionRun] = []
    for file_path in files:
        with h5py.File(file_path, "r") as h5f:
            if "collisions" not in h5f:
                continue
            collisions = h5f["collisions"][:]
        for env_idx, start, end in find_collision_runs(collisions):
            candidates.append(CollisionRun(file_path=file_path, env_idx=env_idx, start=start, end=end))
    candidates.sort(
        key=lambda run: (
            -run.length,
            os.path.basename(run.file_path),
            run.env_idx,
            run.start,
        )
    )
    return candidates


def annotate_frame(
    frame_chw: np.ndarray,
    font: ImageFont.ImageFont,
    text_lines: list[str],
) -> np.ndarray:
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
    img = Image.fromarray(frame_hwc, mode="RGB")
    draw = ImageDraw.Draw(img)

    line_height = 12
    pad = 6
    box_height = pad * 2 + line_height * len(text_lines)
    box_width = min(img.width - 8, 210)
    draw.rectangle((4, 4, 4 + box_width, 4 + box_height), fill=(0, 0, 0))

    y = 4 + pad
    for line in text_lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    return np.asarray(img, dtype=np.uint8)


def encode_mp4(frames_hwc: list[np.ndarray], out_path: str, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if not frames_hwc:
        raise ValueError("No frames provided for MP4 export")

    height, width = frames_hwc[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames_hwc:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        ret = proc.wait()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}:\n{stderr}")


def export_clip(
    run: CollisionRun,
    out_dir: str,
    clip_idx: int,
    pre_frames: int,
    post_frames: int,
    fps: int,
    font: ImageFont.ImageFont,
) -> dict[str, object]:
    with h5py.File(run.file_path, "r") as h5f:
        vision = h5f["vision"]
        collisions = h5f["collisions"][run.env_idx]
        total_steps = int(vision.shape[1])
        clip_start = max(0, run.start - pre_frames)
        clip_end = min(total_steps, run.end + post_frames)
        clip_frames = vision[run.env_idx, clip_start:clip_end]

    frames_hwc: list[np.ndarray] = []
    basename = os.path.splitext(os.path.basename(run.file_path))[0]
    for absolute_step, frame_chw in zip(range(clip_start, clip_end), clip_frames):
        rel = absolute_step - run.start
        lines = [
            f"{basename} env={run.env_idx}",
            f"t={absolute_step} rel={rel:+d}",
            f"collision={'yes' if collisions[absolute_step] else 'no'}",
            f"run={run.start}:{run.end} len={run.length}",
        ]
        frames_hwc.append(annotate_frame(frame_chw, font, lines))

    clip_name = (
        f"clip_{clip_idx:03d}_{basename}_env{run.env_idx:03d}"
        f"_t{run.start:04d}_len{run.length:03d}.mp4"
    )
    out_path = os.path.join(out_dir, clip_name)
    encode_mp4(frames_hwc, out_path, fps=fps)

    return {
        "clip_path": out_path,
        "source_file": run.file_path,
        "env_idx": run.env_idx,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "collision_start": run.start,
        "collision_end": run.end,
        "collision_len": run.length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MP4 clips around collision events.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing *_rgb.h5 files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for exported MP4 clips.")
    parser.add_argument("--max_clips", type=int, default=8, help="Maximum number of clips to export.")
    parser.add_argument("--pre_frames", type=int, default=20, help="Frames before collision onset to include.")
    parser.add_argument("--post_frames", type=int, default=40, help="Frames after collision run to include.")
    parser.add_argument("--fps", type=int, default=12, help="Output video FPS.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = discover_files(args.data_dir)
    candidates = collect_candidates(files)
    if not candidates:
        print("No collision runs found; nothing to export.")
        return

    font = ImageFont.load_default()
    manifest_path = os.path.join(args.out_dir, "clips_manifest.csv")

    exported = []
    for clip_idx, run in enumerate(candidates[: max(1, args.max_clips)], start=1):
        meta = export_clip(
            run,
            out_dir=args.out_dir,
            clip_idx=clip_idx,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
            fps=args.fps,
            font=font,
        )
        exported.append(meta)
        print(
            f"[{clip_idx}/{min(len(candidates), args.max_clips)}] "
            f"{os.path.basename(meta['clip_path'])} "
            f"(env={run.env_idx}, collision={run.start}:{run.end}, len={run.length})"
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_path",
                "source_file",
                "env_idx",
                "clip_start",
                "clip_end",
                "collision_start",
                "collision_end",
                "collision_len",
            ],
        )
        writer.writeheader()
        writer.writerows(exported)

    print(f"Exported {len(exported)} clip(s) to {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
