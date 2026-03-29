#!/usr/bin/env python3
"""Export third-person chase-camera clips around collision events from raw chunks.

This replays saved rollout trajectories from ``chunk_*.npz`` in Genesis, so it
does not require recollecting physics data. It is intended for spot checks,
especially when first-person clips make robot motion hard to interpret.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import h5py
import numpy as np
from PIL import Image, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from export_collision_clips import annotate_frame, encode_gif, encode_mp4, find_collision_runs, resolve_output_format
from lewm.beacon_utils import BeaconLayout
from lewm.genesis_utils import init_genesis_once
from lewm.math_utils import forward_up_from_quat
from lewm.obstacle_utils import ObstacleLayout, add_obstacles_to_scene


URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
JOINTS_ACTUATED = [
    "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
    "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]


@dataclass(frozen=True)
class CollisionRun:
    file_path: str
    env_idx: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def discover_raw_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError(f"No raw rollout chunks found in {data_dir}")
    return files


def collect_candidates(files: list[str]) -> list[CollisionRun]:
    candidates: list[CollisionRun] = []
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as data:
            if "collisions" not in data:
                continue
            collisions = np.asarray(data["collisions"])
        for env_idx, start, end in find_collision_runs(collisions):
            candidates.append(CollisionRun(file_path=file_path, env_idx=env_idx, start=start, end=end))
    candidates.sort(
        key=lambda run: (-run.length, os.path.basename(run.file_path), run.env_idx, run.start)
    )
    return candidates


def build_scene(gs, layout: ObstacleLayout, beacon_layout: BeaconLayout, img_res: int, fov: float):
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    add_obstacles_to_scene(scene, layout)

    for obs in beacon_layout.all_obstacles():
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )

    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(img_res, img_res), fov=fov, near=0.01, GUI=False)
    scene.build(n_envs=1)

    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]

    import torch

    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)
    return scene, robot, cam, act_dofs


def resize_frame(frame_hwc: np.ndarray, target_res: int) -> np.ndarray:
    if frame_hwc.shape[0] == target_res and frame_hwc.shape[1] == target_res:
        return frame_hwc
    return np.asarray(Image.fromarray(frame_hwc).resize((target_res, target_res), Image.Resampling.BILINEAR))


def matching_first_person_file(first_person_dir: str, raw_file_path: str) -> str:
    basename = os.path.splitext(os.path.basename(raw_file_path))[0]
    return os.path.join(first_person_dir, f"{basename}_rgb.h5")


def load_first_person_frames(
    first_person_dir: str,
    raw_file_path: str,
    env_idx: int,
    clip_start: int,
    clip_end: int,
    target_res: int,
) -> list[np.ndarray]:
    h5_path = matching_first_person_file(first_person_dir, raw_file_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Matching rendered HDF5 not found for {raw_file_path}: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        clip_frames = np.asarray(h5f["vision"][env_idx, clip_start:clip_end], dtype=np.uint8)

    frames_hwc: list[np.ndarray] = []
    for frame_chw in clip_frames:
        frame_hwc = np.transpose(frame_chw, (1, 2, 0))
        frames_hwc.append(resize_frame(frame_hwc, target_res))
    return frames_hwc


def build_side_by_side_frame(first_person_hwc: np.ndarray, third_person_hwc: np.ndarray) -> np.ndarray:
    divider = np.full((third_person_hwc.shape[0], 8, 3), 12, dtype=np.uint8)
    return np.concatenate([first_person_hwc, divider, third_person_hwc], axis=1)


def render_clip(
    gs,
    torch,
    robot,
    cam,
    act_dofs,
    arrays: dict[str, np.ndarray],
    run: CollisionRun,
    pre_frames: int,
    post_frames: int,
    img_res: int,
    chase_dist: float,
    chase_height: float,
    side_offset: float,
    lookahead: float,
    font: ImageFont.ImageFont,
    first_person_dir: str | None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    base_pos = arrays["base_pos"]
    base_quat = arrays["base_quat"]
    joint_pos = arrays["joint_pos"]
    collisions = arrays["collisions"]

    total_steps = int(base_pos.shape[1])
    clip_start = max(0, run.start - pre_frames)
    clip_end = min(total_steps, run.end + post_frames)
    first_person_frames = None
    if first_person_dir:
        first_person_frames = load_first_person_frames(
            first_person_dir=first_person_dir,
            raw_file_path=run.file_path,
            env_idx=run.env_idx,
            clip_start=clip_start,
            clip_end=clip_end,
            target_res=img_res,
        )

    frames_hwc: list[np.ndarray] = []
    for absolute_step in range(clip_start, clip_end):
        pos_np = np.asarray(base_pos[run.env_idx, absolute_step], dtype=np.float32)
        quat_np = np.asarray(base_quat[run.env_idx, absolute_step], dtype=np.float32)
        joint_np = np.asarray(joint_pos[run.env_idx, absolute_step], dtype=np.float32)

        robot.set_pos(torch.tensor(pos_np, device=gs.device, dtype=torch.float32).unsqueeze(0))
        robot.set_quat(torch.tensor(quat_np, device=gs.device, dtype=torch.float32).unsqueeze(0))
        robot.set_dofs_position(
            torch.tensor(joint_np, device=gs.device, dtype=torch.float32).unsqueeze(0),
            act_dofs,
        )

        fw, _up = forward_up_from_quat(quat_np)
        fw_xy = np.asarray([fw[0], fw[1], 0.0], dtype=np.float32)
        fw_norm = float(np.linalg.norm(fw_xy[:2]))
        if fw_norm < 1e-6:
            fw_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            fw_xy /= fw_norm
        side = np.array([-fw_xy[1], fw_xy[0], 0.0], dtype=np.float32)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        cam_pos = pos_np - chase_dist * fw_xy + chase_height * world_up + side_offset * side
        cam_lookat = pos_np + lookahead * fw_xy + 0.18 * world_up
        cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=world_up)

        render_out = cam.render(rgb=True, force_render=True)
        rgb = render_out[0]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb, dtype=np.uint8)

        rel = absolute_step - run.start
        lines = [
            f"{os.path.basename(run.file_path)} env={run.env_idx}",
            f"t={absolute_step} rel={rel:+d}",
            f"collision={'yes' if collisions[run.env_idx, absolute_step] else 'no'}",
            f"run={run.start}:{run.end} len={run.length} third-person",
        ]
        third_person_frame = annotate_frame(np.transpose(rgb, (2, 0, 1)), font, lines)
        if first_person_frames is None:
            frames_hwc.append(third_person_frame)
            continue

        fp_lines = [
            f"{os.path.basename(run.file_path)} env={run.env_idx}",
            f"t={absolute_step} rel={rel:+d}",
            f"collision={'yes' if collisions[run.env_idx, absolute_step] else 'no'}",
            f"run={run.start}:{run.end} len={run.length} first-person",
        ]
        fp_idx = absolute_step - clip_start
        first_person_frame = annotate_frame(
            np.transpose(first_person_frames[fp_idx], (2, 0, 1)),
            font,
            fp_lines,
        )
        frames_hwc.append(build_side_by_side_frame(first_person_frame, third_person_frame))

    meta = {
        "clip_start": clip_start,
        "clip_end": clip_end,
        "collision_start": run.start,
        "collision_end": run.end,
        "collision_len": run.length,
    }
    return frames_hwc, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export third-person collision spot-check clips.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw chunk_*.npz files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for output clips.")
    parser.add_argument("--first_person_dir", type=str, default="",
                        help="Optional directory containing rendered *_rgb.h5 files for side-by-side export.")
    parser.add_argument("--max_clips", type=int, default=8, help="Maximum number of clips to export.")
    parser.add_argument("--pre_frames", type=int, default=20, help="Frames before collision onset.")
    parser.add_argument("--post_frames", type=int, default=40, help="Frames after the collision run.")
    parser.add_argument("--fps", type=int, default=12, help="Output FPS.")
    parser.add_argument("--img_res", type=int, default=384, help="Square render resolution.")
    parser.add_argument("--fov", type=float, default=60.0, help="Third-person camera field of view.")
    parser.add_argument("--sim_backend", type=str, default="auto")
    parser.add_argument("--format", type=str, default="auto", help="auto | mp4 | gif")
    parser.add_argument("--chase_dist", type=float, default=1.0, help="Distance behind the robot.")
    parser.add_argument("--chase_height", type=float, default=0.55, help="Camera height above the ground.")
    parser.add_argument("--side_offset", type=float, default=0.25, help="Lateral shoulder-cam offset.")
    parser.add_argument("--lookahead", type=float, default=0.2, help="Look-ahead distance in front of robot.")
    args = parser.parse_args()

    output_format, extension = resolve_output_format(args.format)
    os.makedirs(args.out_dir, exist_ok=True)
    first_person_dir = args.first_person_dir.strip() or None
    if first_person_dir is not None and not os.path.isdir(first_person_dir):
        raise FileNotFoundError(f"First-person render dir not found: {first_person_dir}")

    files = discover_raw_files(args.raw_dir)
    candidates = collect_candidates(files)
    if not candidates:
        print("No collision runs found; nothing to export.")
        return

    selected = candidates[: max(1, args.max_clips)]
    grouped: dict[str, list[CollisionRun]] = defaultdict(list)
    for run in selected:
        grouped[run.file_path].append(run)

    import genesis as gs
    import torch

    init_genesis_once(args.sim_backend)
    font = ImageFont.load_default()
    manifest_path = os.path.join(args.out_dir, "clips_manifest.csv")
    print(f"Clip output format: {output_format}")
    if first_person_dir:
        print(f"Export mode: side-by-side (first-person + third-person)")
    else:
        print(f"Export mode: third-person only")

    exported: list[dict[str, object]] = []
    clip_idx = 0
    try:
        for file_path in sorted(grouped):
            with np.load(file_path, allow_pickle=True) as data:
                arrays = {
                    "base_pos": np.asarray(data["base_pos"]),
                    "base_quat": np.asarray(data["base_quat"]),
                    "joint_pos": np.asarray(data["joint_pos"]),
                    "collisions": np.asarray(data["collisions"]),
                }
                obstacle_json = str(data["obstacle_layout"].item() if hasattr(data["obstacle_layout"], "item") else data["obstacle_layout"])
                beacon_json = str(data["beacon_layout"].item() if hasattr(data["beacon_layout"], "item") else data["beacon_layout"])

            layout = ObstacleLayout.from_json(obstacle_json)
            beacon_layout = BeaconLayout.from_json(beacon_json)
            scene, robot, cam, act_dofs = build_scene(gs, layout, beacon_layout, args.img_res, args.fov)
            try:
                for run in grouped[file_path]:
                    clip_idx += 1
                    frames_hwc, meta = render_clip(
                        gs=gs,
                        torch=torch,
                        robot=robot,
                        cam=cam,
                        act_dofs=act_dofs,
                        arrays=arrays,
                        run=run,
                        pre_frames=args.pre_frames,
                        post_frames=args.post_frames,
                        img_res=args.img_res,
                        chase_dist=args.chase_dist,
                        chase_height=args.chase_height,
                        side_offset=args.side_offset,
                        lookahead=args.lookahead,
                        font=font,
                        first_person_dir=first_person_dir,
                    )

                    basename = os.path.splitext(os.path.basename(run.file_path))[0]
                    prefix = "side_by_side" if first_person_dir else "third_person"
                    clip_name = (
                        f"{prefix}_{clip_idx:03d}_{basename}_env{run.env_idx:03d}"
                        f"_t{run.start:04d}_len{run.length:03d}{extension}"
                    )
                    out_path = os.path.join(args.out_dir, clip_name)
                    if output_format == "mp4":
                        encode_mp4(frames_hwc, out_path, fps=args.fps)
                    else:
                        encode_gif(frames_hwc, out_path, fps=args.fps)

                    meta.update({
                        "clip_path": out_path,
                        "source_file": run.file_path,
                        "env_idx": run.env_idx,
                        "first_person_file": matching_first_person_file(first_person_dir, run.file_path) if first_person_dir else "",
                    })
                    exported.append(meta)
                    print(
                        f"[{clip_idx}/{len(selected)}] {clip_name} "
                        f"(env={run.env_idx}, collision={run.start}:{run.end}, len={run.length})"
                    )
            finally:
                scene.destroy()
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()

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
                "first_person_file",
            ],
        )
        writer.writeheader()
        writer.writerows(exported)

    if first_person_dir:
        print(f"Exported {len(exported)} side-by-side clip(s) to {args.out_dir}")
    else:
        print(f"Exported {len(exported)} third-person clip(s) to {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
