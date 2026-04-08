#!/usr/bin/env python3
"""Visual renderer: renders egocentric RGB from recorded physics rollouts.

Egocentric camera protections:
  - Shared camera model, reused by the renderer and validation scripts
  - Smaller near plane to avoid cutting away nearby wall surfaces
  - Frustum-aware multi-ray clipping detection (9 rays spanning the full FOV)
  - Camera retraction along -forward when frustum check detects imminent clipping
  - Depth-buffer validation after render to confirm no near-plane pixels remain
  - Frame substitution as last-resort fallback only

Other features (from v1):
  - Beacon panel rendering (coloured panels added to the scene)
  - Beacon-like wall colour randomization (confusable wall colours)
  - Camera pose jitter (slight random offset each frame for robustness)
  - Wall/box texture variation (per-obstacle material randomization)
  - Preservation of new label fields (clearance, beacon_*, traversability, etc.)

Reads .npz chunk files produced by 1_physics_rollout.py, replays the recorded
trajectories in isolated Genesis render scenes (one per worker process), applies
visual domain randomization, and writes the final dataset to HDF5.

Usage:
    python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset --workers 4
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import queue
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lewm.genesis_utils import to_numpy
from lewm.texture_utils import generate_texture_set
from lewm.obstacle_utils import ObstacleLayout
from lewm.beacon_utils import BeaconLayout, beacon_like_wall_color
from lewm.label_utils import compute_episode_labels
from lewm.camera_utils import (
    add_egocentric_camera_args,
    camera_rotation_matrix,
    camera_safety_metrics,
    depth_buffer_has_clipping,
    ego_camera_config_from_args,
    egocentric_camera_pose,
    retract_camera_to_safe,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

URDF_PATH = "assets/mini_pupper/mini_pupper_render.urdf"

JOINTS_ACTUATED = [
    "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
    "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

DEFAULT_IMG_RES = 224
DEFAULT_TEXTURE_COUNT = 27
DEFAULT_TEXTURE_VARIANTS_PER_WORKER = 4
VULKAN_SAFE_WORKER_LIMIT = 4
VULKAN_SAFE_TEXTURE_VARIANT_LIMIT = 1
HIP_SAFE_WORKER_LIMIT = 1
HIP_SAFE_TEXTURE_VARIANT_LIMIT = 1

# --------------------------------------------------------------------------- #
# Visual domain randomization
# --------------------------------------------------------------------------- #

def pick_backend(gs, backend_str: str):
    """Resolve a Genesis backend across API variants without hard-failing."""
    backend_str = backend_str.lower().strip()
    available = {
        "cpu": getattr(gs, "cpu", None),
        "gpu": getattr(gs, "gpu", None),
        "cuda": getattr(gs, "cuda", None),
        "vulkan": getattr(gs, "vulkan", None),
        "metal": getattr(gs, "metal", None),
        "amdgpu": getattr(gs, "amdgpu", None),
    }

    explicit = {
        "cpu": ("cpu",),
        "gpu": ("gpu", "amdgpu", "cuda", "vulkan", "metal", "cpu"),
        "cuda": ("cuda", "gpu", "cpu"),
        "vulkan": ("vulkan", "gpu", "amdgpu", "cuda", "metal", "cpu"),
        "metal": ("metal", "gpu", "cpu"),
        "amdgpu": ("amdgpu", "gpu", "vulkan", "cpu"),
        "amd": ("amdgpu", "gpu", "vulkan", "cpu"),
        "hip": ("amdgpu", "gpu", "vulkan", "cpu"),
        "auto": ("vulkan", "amdgpu", "gpu", "cuda", "metal", "cpu"),
    }

    candidates = explicit.get(backend_str, explicit["auto"])
    for name in candidates:
        backend = available.get(name)
        if backend is not None:
            if name == backend_str:
                return backend, name
            return backend, f"{name} (requested {backend_str})"

    return gs.cpu, "cpu (last-resort fallback)"


def opened_render_nodes() -> list[str]:
    """Best-effort introspection of DRM render nodes opened by this process."""
    fd_dir = "/proc/self/fd"
    nodes = []
    try:
        for entry in os.listdir(fd_dir):
            path = os.path.join(fd_dir, entry)
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if "/dev/dri/renderD" in target:
                nodes.append(target)
    except OSError:
        return []
    return sorted(set(nodes))


def normalize_h5_compression(name: str | None) -> str | None:
    """Map CLI compression names onto h5py's expected values."""
    if name is None:
        return None
    name = str(name).strip().lower()
    if name in {"", "none", "off", "false", "0"}:
        return None
    if name in {"gzip", "lzf"}:
        return name
    raise ValueError(f"Unsupported HDF5 compression '{name}'. Use one of: none, gzip, lzf.")


def is_rocm_runtime() -> bool:
    """Return True when the local PyTorch runtime is backed by ROCm/HIP."""
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def apply_visual_domain_randomization(rgb: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply per-frame visual domain randomization to an (H, W, 3) uint8 image."""
    img = rgb.astype(np.float32) / 255.0

    # Brightness
    brightness = rng.uniform(-0.4, 0.4)
    img = img + brightness

    # Contrast
    contrast = rng.uniform(0.5, 1.5)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean

    # Gaussian noise
    sigma = rng.uniform(0.02, 0.08)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    img = img + noise

    # Hue shift
    hue_angle = rng.uniform(-0.08, 0.08)
    cos_a = math.cos(hue_angle)
    sin_a = math.sin(hue_angle)
    one_third = 1.0 / 3.0
    sqrt_third = math.sqrt(one_third)
    hue_mat = np.array([
        [cos_a + one_third * (1 - cos_a),
         one_third * (1 - cos_a) - sqrt_third * sin_a,
         one_third * (1 - cos_a) + sqrt_third * sin_a],
        [one_third * (1 - cos_a) + sqrt_third * sin_a,
         cos_a + one_third * (1 - cos_a),
         one_third * (1 - cos_a) - sqrt_third * sin_a],
        [one_third * (1 - cos_a) - sqrt_third * sin_a,
         one_third * (1 - cos_a) + sqrt_third * sin_a,
         cos_a + one_third * (1 - cos_a)],
    ], dtype=np.float32)
    img = img @ hue_mat.T

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def sample_obstacle_color(rng: np.random.RandomState) -> Tuple[float, float, float]:
    """Muted random obstacle colour."""
    base = rng.uniform(0.3, 0.7)
    tint = rng.uniform(-0.1, 0.1, size=3)
    color = np.clip(base + tint, 0.1, 0.9)
    return (float(color[0]), float(color[1]), float(color[2]))


def sample_wall_color(rng: np.random.RandomState, beacon_confuse_prob: float = 0.3) -> Tuple[float, float, float]:
    """Wall colour — sometimes beacon-like to force shape/context attention."""
    if rng.rand() < beacon_confuse_prob:
        return beacon_like_wall_color(rng)
    return sample_obstacle_color(rng)


def build_render_bundle(
    gs, torch,
    texture_path: str,
    layout: ObstacleLayout,
    beacon_layout: BeaconLayout,
    rng: np.random.RandomState,
    img_res: int,
    cam_fov: float,
    cam_near: float,
    beacon_confuse_prob: float = 0.3,
):
    """Construct one textured render scene variant with beacon panels."""
    scene = gs.Scene(show_viewer=False)

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=texture_path),
        ),
    )

    # Obstacles (walls, boxes) with randomized colours
    for obs in layout.obstacles:
        color = sample_wall_color(rng, beacon_confuse_prob)
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=color),
        )

    # Beacon panels — use their actual colour (identity-linked)
    for beacon_obs in beacon_layout.all_obstacles():
        scene.add_entity(
            gs.morphs.Box(pos=beacon_obs.pos, size=beacon_obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=beacon_obs.color),
        )

    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(img_res, img_res), fov=cam_fov, near=cam_near, GUI=False)
    scene.build(n_envs=1)

    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    return {
        "scene": scene,
        "robot": robot,
        "cam": cam,
        "act_dofs": act_dofs,
        "texture_path": texture_path,
    }


# --------------------------------------------------------------------------- #
# Render worker
# --------------------------------------------------------------------------- #

def render_worker(args_tuple):
    """Each worker renders a subset of environments from one .npz chunk."""
    (worker_id, chunk_file, start_env, end_env,
     tmp_file, sim_backend, texture_dir, obstacle_json, beacon_json,
     texture_count, texture_variants, beacon_confuse_prob, img_res,
     tmp_vision_compression, skip_physics_step, camera_cfg, progress_queue) = args_tuple

    N_subset = end_env - start_env

    # Skip if already fully rendered
    if os.path.exists(tmp_file):
        try:
            with h5py.File(tmp_file, "r") as f:
                if "vision" in f and f["vision"].shape[0] == N_subset:
                    return tmp_file
        except Exception:
            pass

    import genesis as gs
    import torch

    backend_obj, backend_desc = pick_backend(gs, sim_backend)
    gs.init(backend=backend_obj, logging_level="warning")
    print(
        f"[worker {worker_id}] Genesis backend: requested={sim_backend} resolved={backend_desc}",
        flush=True,
    )
    texture_count = max(1, int(texture_count))
    texture_variants = max(1, int(texture_variants))
    worker_seed = worker_id + int.from_bytes(os.urandom(4), "little")
    worker_rng = np.random.RandomState(worker_seed)

    texture_paths = sorted(glob.glob(os.path.join(texture_dir, "*.png")))
    if len(texture_paths) < texture_count:
        texture_paths = generate_texture_set(texture_dir, count=texture_count)
    else:
        texture_paths = texture_paths[:texture_count]
    if not texture_paths:
        raise RuntimeError(f"No textures available in {texture_dir}")

    layout = ObstacleLayout.from_json(obstacle_json)
    beacon_layout = BeaconLayout.from_json(beacon_json)

    variant_count = max(1, min(texture_variants, len(texture_paths)))
    variant_ids = worker_rng.choice(len(texture_paths), size=variant_count, replace=False)
    bundles = [
        build_render_bundle(
            gs, torch,
            texture_paths[int(texture_idx)],
            layout,
            beacon_layout,
            np.random.RandomState(worker_seed + 1009 * (variant_offset + 1)),
            img_res=img_res,
            cam_fov=camera_cfg.fov_deg,
            cam_near=camera_cfg.near_plane,
            beacon_confuse_prob=beacon_confuse_prob,
        )
        for variant_offset, texture_idx in enumerate(variant_ids)
    ]
    render_nodes = opened_render_nodes()
    if render_nodes:
        print(f"[worker {worker_id}] DRM render nodes: {', '.join(render_nodes)}", flush=True)
    env_bundle_ids = worker_rng.randint(0, len(bundles), size=N_subset)

    with np.load(chunk_file, allow_pickle=True) as data_npz:
        # `.npz` archives are lazy zip members; repeated `data_npz["key"]`
        # lookups re-read and decompress. Materialize the hot arrays once.
        base_pos_all = np.asarray(data_npz["base_pos"])
        base_quat_all = np.asarray(data_npz["base_quat"])
        joint_pos_all = np.asarray(data_npz["joint_pos"])
    T = base_pos_all.shape[1]

    with h5py.File(tmp_file, "w") as f:
        h5_vision = f.create_dataset(
            "vision",
            (N_subset, T, 3, img_res, img_res),
            dtype="uint8",
            compression=tmp_vision_compression,
        )

        for local_idx, env_idx in enumerate(range(start_env, end_env)):
            bundle = bundles[int(env_bundle_ids[local_idx])]
            scene = bundle["scene"]
            robot = bundle["robot"]
            cam = bundle["cam"]
            act_dofs = bundle["act_dofs"]
            env_rng = np.random.RandomState(worker_seed + env_idx * 7919 + 17)

            base_pos_seq = torch.tensor(
                base_pos_all[env_idx], device=gs.device, dtype=torch.float32,
            )
            base_quat_seq = torch.tensor(
                base_quat_all[env_idx], device=gs.device, dtype=torch.float32,
            )
            joint_pos_seq = torch.tensor(
                joint_pos_all[env_idx], device=gs.device, dtype=torch.float32,
            )

            env_video = np.zeros((T, 3, img_res, img_res), dtype=np.uint8)
            last_clean_frame = None  # last-resort fallback only
            retracted_count = 0
            substituted_count = 0
            depth_clipped_count = 0
            depth_available = True  # try depth on first frame; disable if unsupported

            for step in range(T):
                base_pos = base_pos_seq[step].unsqueeze(0)
                base_quat = base_quat_seq[step].unsqueeze(0)

                robot.set_pos(base_pos)
                robot.set_quat(base_quat)
                robot.set_dofs_position(joint_pos_seq[step].unsqueeze(0), act_dofs)

                if not skip_physics_step:
                    scene.step(update_visualizer=False)

                # ---- Camera placement with jitter ---- #
                q_np = to_numpy(base_quat_seq[step])
                pos_np = to_numpy(base_pos_seq[step])
                cam_pos, cam_lookat, cam_up, cam_forward = egocentric_camera_pose(pos_np, q_np, camera_cfg)
                cam_rot = camera_rotation_matrix(q_np, camera_cfg.pitch_rad)

                # Camera pose jitter for visual robustness.
                cam_pos = cam_pos + env_rng.uniform(-camera_cfg.pos_jitter, camera_cfg.pos_jitter, size=3)
                cam_lookat = cam_lookat + env_rng.uniform(-camera_cfg.lookat_jitter, camera_cfg.lookat_jitter, size=3)
                cam_forward = cam_lookat - cam_pos
                forward_norm = float(np.linalg.norm(cam_forward))
                if forward_norm > 1e-8:
                    cam_forward = cam_forward / forward_norm

                # ---- Frustum-aware safety check ---- #
                safety = camera_safety_metrics(cam_pos, cam_forward, layout, camera_cfg, cam_rot=cam_rot)

                if safety["unsafe"]:
                    # Try camera retraction before falling back to substitution
                    cam_pos, cam_lookat, cam_up, cam_forward, retract_dist = retract_camera_to_safe(
                        cam_pos, cam_forward, cam_up, cam_rot, layout, camera_cfg,
                    )
                    if retract_dist > 0:
                        # Re-check after retraction
                        safety = camera_safety_metrics(cam_pos, cam_forward, layout, camera_cfg, cam_rot=cam_rot)
                        if safety["unsafe"]:
                            # Retraction wasn't enough — last-resort substitution
                            substituted_count += 1
                            if last_clean_frame is not None:
                                env_video[step] = last_clean_frame
                            continue
                        retracted_count += 1
                    else:
                        # Retraction returned 0 but was still unsafe — substitute
                        substituted_count += 1
                        if last_clean_frame is not None:
                            env_video[step] = last_clean_frame
                        continue

                cam.set_pose(
                    pos=cam_pos,
                    lookat=cam_lookat,
                    up=cam_up,
                )

                # Render with depth when available for clipping validation
                if depth_available:
                    try:
                        render_out = cam.render(rgb=True, depth=True, force_render=skip_physics_step)
                    except TypeError:
                        depth_available = False
                        render_out = cam.render(rgb=True, force_render=skip_physics_step)
                else:
                    render_out = cam.render(rgb=True, force_render=skip_physics_step)
                rgb = render_out[0]
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                rgb = np.asarray(rgb, dtype=np.uint8)

                # ---- Depth-buffer clipping validation (log only, no frame drop) ---- #
                # The frustum+retraction check is the primary guard. Depth stats
                # are logged so we can audit, but frames are NOT dropped here —
                # the floor and robot legs legitimately produce near-plane depth
                # pixels that would cause massive false-positive rates (~20%).
                if depth_available and len(render_out) > 1 and render_out[1] is not None:
                    depth_buf = render_out[1]
                    if hasattr(depth_buf, "cpu"):
                        depth_buf = depth_buf.cpu().numpy()
                    depth_buf = np.asarray(depth_buf, dtype=np.float32)
                    if depth_buffer_has_clipping(depth_buf, camera_cfg.near_plane):
                        depth_clipped_count += 1

                rgb = apply_visual_domain_randomization(rgb, env_rng)

                frame = np.transpose(rgb, (2, 0, 1))
                env_video[step] = frame
                last_clean_frame = frame

            total_issues = retracted_count + substituted_count + depth_clipped_count
            if total_issues > 0:
                print(
                    f"[worker {worker_id}] env {env_idx}: "
                    f"{retracted_count} retracted-ok, {substituted_count} substituted, "
                    f"{depth_clipped_count} depth-clipped out of {T} frames",
                    flush=True,
                )

            h5_vision[local_idx] = env_video
            progress_queue.put(1)

    return tmp_file


def run_render_processes(tasks: list[tuple], tmp_files: list[str], expected_envs: int, pbar) -> tuple[list[int], list[str], int]:
    """Launch worker processes, update progress, and collect failure details."""
    progress_queue = mp.Queue()
    tasks_with_q = [(*task, progress_queue) for task in tasks]
    processes = [mp.Process(target=render_worker, args=(task,)) for task in tasks_with_q]
    for proc in processes:
        proc.start()

    envs_received = 0
    try:
        while envs_received < expected_envs:
            try:
                progress_queue.get(timeout=5)
                pbar.update(1)
                envs_received += 1
            except queue.Empty:
                if not any(proc.is_alive() for proc in processes):
                    break
    finally:
        for proc in processes:
            proc.join()
        progress_queue.close()
        progress_queue.join_thread()

    failed = [proc.pid for proc in processes if proc.exitcode not in (0, None)]
    missing_tmp = [tmp for tmp in tmp_files if not os.path.exists(tmp)]
    return failed, missing_tmp, envs_received


# --------------------------------------------------------------------------- #
# Stitch worker outputs into one HDF5
# --------------------------------------------------------------------------- #

# Label fields to pass through from .npz to HDF5
LABEL_FIELDS = [
    "collisions", "clearance", "near_miss", "traversability",
    "beacon_visible", "beacon_identity", "beacon_bearing", "beacon_range",
    "cmd_pattern",
]

RECOMPUTED_LABEL_FIELDS = [
    "clearance",
    "near_miss",
    "traversability",
    "beacon_visible",
    "beacon_identity",
    "beacon_bearing",
    "beacon_range",
]


def recompute_chunk_labels(data: dict, obstacle_json: str, beacon_json: str) -> dict[str, np.ndarray]:
    """Recompute geometry-derived labels from recorded base state."""
    base_pos = np.asarray(data["base_pos"])
    base_quat = np.asarray(data["base_quat"])
    n_envs, steps = int(base_pos.shape[0]), int(base_pos.shape[1])

    w = base_quat[..., 0]
    x = base_quat[..., 1]
    y = base_quat[..., 2]
    z = base_quat[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)

    obstacle_layout = ObstacleLayout.from_json(obstacle_json)
    beacon_layout = BeaconLayout.from_json(beacon_json)
    beacon_layout_or_none = beacon_layout if len(beacon_layout.beacons) > 0 else None

    labels_out: dict[str, np.ndarray] = {
        "clearance": np.zeros((n_envs, steps), dtype=np.float32),
        "near_miss": np.zeros((n_envs, steps), dtype=bool),
        "traversability": np.zeros((n_envs, steps), dtype=np.int32),
        "beacon_visible": np.zeros((n_envs, steps), dtype=bool),
        "beacon_identity": np.full((n_envs, steps), -1, dtype=np.int32),
        "beacon_bearing": np.zeros((n_envs, steps), dtype=np.float32),
        "beacon_range": np.full((n_envs, steps), float("inf"), dtype=np.float32),
    }

    for env_i in range(n_envs):
        labels = compute_episode_labels(
            robot_xy=base_pos[env_i, :, :2].astype(np.float32),
            robot_yaw=yaw[env_i],
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout_or_none,
        )
        for field in RECOMPUTED_LABEL_FIELDS:
            labels_out[field][env_i] = labels[field]
    return labels_out


def stitch_hdf5(
    out_path: str,
    tmp_files: List[str],
    tasks: list,
    data: dict,
    N: int,
    T: int,
    img_res: int,
    vision_compression: str | None,
    vision_source_path: str | None = None,
) -> None:
    """Merge per-worker HDF5 shards and raw data into one final HDF5 file."""
    tmp_out = out_path + ".stitching"
    try:
        with h5py.File(tmp_out, "w") as h5f:
            h5_vision = h5f.create_dataset(
                "vision", (N, T, 3, img_res, img_res),
                dtype="uint8",
                chunks=(1, T, 3, img_res, img_res),
                compression=vision_compression,
            )

            h5f.create_dataset("proprio", data=data["proprio"], compression="gzip")
            h5f.create_dataset("cmds", data=data["cmds"], compression="gzip")
            h5f.create_dataset("dones", data=data["dones"], compression="gzip")

            # Pass through all label fields
            for field in LABEL_FIELDS:
                if field in data:
                    h5f.create_dataset(field, data=data[field], compression="gzip")

            # Store layout JSON as attributes
            if "obstacle_layout" in data:
                raw = data["obstacle_layout"]
                h5f.attrs["obstacle_layout"] = str(raw.item() if hasattr(raw, "item") else raw)
            if "beacon_layout" in data:
                raw = data["beacon_layout"]
                h5f.attrs["beacon_layout"] = str(raw.item() if hasattr(raw, "item") else raw)
            for attr_key in ("scene_seed", "scene_type", "scene_meta"):
                if attr_key in data:
                    raw = data[attr_key]
                    h5f.attrs[attr_key] = raw.item() if hasattr(raw, "item") else raw
            h5f.attrs["label_visibility_mode"] = "fov_range_front_los"

            # Copy vision from either temporary render shards or an existing HDF5.
            if vision_source_path is not None:
                try:
                    with h5py.File(vision_source_path, "r") as src_h5:
                        h5_vision[:] = src_h5["vision"][:]
                except Exception as e:
                    print(f"[stitch] Failed to copy vision from {vision_source_path}: {e}")
                    raise
            else:
                for tmp_file, task in zip(tmp_files, tasks):
                    start, end = task[2], task[3]
                    try:
                        with h5py.File(tmp_file, "r") as tmp_in:
                            h5_vision[start:end] = tmp_in["vision"][:]
                    except Exception as e:
                        print(f"[stitch] Failed to merge {tmp_file}: {e}")
                        raise

        os.replace(tmp_out, out_path)

        for tmp_file in tmp_files:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception as e:
                print(f"[stitch] Warning: could not remove {tmp_file}: {e}")

    except Exception:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        raise


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Render egocentric RGB from recorded physics rollouts.",
    )
    parser.add_argument("--raw_dir", type=str, default="jepa_raw_data")
    parser.add_argument("--out_dir", type=str, default="jepa_final_dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sim_backend", type=str, default="auto")
    parser.add_argument("--texture_count", type=int, default=DEFAULT_TEXTURE_COUNT)
    parser.add_argument("--texture_variants_per_worker", type=int, default=DEFAULT_TEXTURE_VARIANTS_PER_WORKER)
    parser.add_argument("--unsafe_backend_parallelism", action="store_true",
                        help="Disable backend safety caps for parallel Genesis rendering.")
    parser.add_argument("--unsafe_vulkan_parallelism", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--img_res", type=int, default=DEFAULT_IMG_RES)
    parser.add_argument("--beacon_confuse_prob", type=float, default=0.3,
                        help="Probability that a wall gets a beacon-like colour.")
    parser.add_argument(
        "--tmp_vision_compression",
        type=str,
        default="none",
        help="Compression for per-worker temporary vision shards: none | gzip | lzf",
    )
    parser.add_argument(
        "--final_vision_compression",
        type=str,
        default="none",
        help="Compression for final stitched vision datasets: none | gzip | lzf",
    )
    parser.add_argument(
        "--skip_physics_step",
        action="store_true",
        help="Replay recorded poses without advancing Genesis physics; force camera refresh on render.",
    )
    parser.add_argument(
        "--reuse_vision_from",
        type=str,
        default=None,
        help="Existing rendered HDF5 directory to copy vision from while recomputing labels.",
    )
    parser.add_argument(
        "--skip_label_recompute",
        action="store_true",
        help="Preserve label arrays from the raw rollout instead of recomputing geometry-derived labels.",
    )
    add_egocentric_camera_args(parser, include_jitter=True)
    args = parser.parse_args()

    tmp_vision_compression = normalize_h5_compression(args.tmp_vision_compression)
    final_vision_compression = normalize_h5_compression(args.final_vision_compression)
    camera_cfg = ego_camera_config_from_args(args, include_jitter=True)

    effective_workers = max(1, int(args.workers))
    effective_texture_variants = max(1, int(args.texture_variants_per_worker))
    backend_name = args.sim_backend.lower().strip()
    unsafe_parallelism = args.unsafe_backend_parallelism or args.unsafe_vulkan_parallelism
    rocm_runtime = is_rocm_runtime()

    if backend_name != "cpu" and not unsafe_parallelism:
        if rocm_runtime:
            # Genesis scene.build() still allocates rigid/SDF buffers through HIP on
            # ROCm, even when the requested backend is Vulkan. Mixed-GPU AMD hosts
            # have proven especially fragile here, so serialize worker startup.
            capped_workers = min(effective_workers, HIP_SAFE_WORKER_LIMIT)
            capped_variants = min(effective_texture_variants, HIP_SAFE_TEXTURE_VARIANT_LIMIT)
            if (capped_workers, capped_variants) != (effective_workers, effective_texture_variants):
                tqdm.write(
                    f"ROCm/AMDGPU safety caps applied: workers {effective_workers}->{capped_workers}, "
                    f"texture_variants {effective_texture_variants}->{capped_variants}. "
                    "Genesis may still touch the HIP allocator even with --sim_backend vulkan. "
                    "Use --unsafe_backend_parallelism to override."
                )
            effective_workers = capped_workers
            effective_texture_variants = capped_variants
        elif backend_name == "vulkan":
            capped_workers = min(effective_workers, VULKAN_SAFE_WORKER_LIMIT)
            capped_variants = min(effective_texture_variants, VULKAN_SAFE_TEXTURE_VARIANT_LIMIT)
            if (capped_workers, capped_variants) != (effective_workers, effective_texture_variants):
                tqdm.write(
                    f"Vulkan safety caps applied: workers {effective_workers}->{capped_workers}, "
                    f"texture_variants {effective_texture_variants}->{capped_variants}."
                )
            effective_workers = capped_workers
            effective_texture_variants = capped_variants

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "chunk_*.npz")))
    if not raw_files:
        tqdm.write(f"No raw data found in {args.raw_dir}/. Run 1_physics_rollout.py first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    texture_dir = os.path.join(args.out_dir, "_textures")
    tqdm.write(
        "Egocentric camera: "
        f"mount=({camera_cfg.mount_pos_body[0]:.3f}, {camera_cfg.mount_pos_body[1]:.3f}, {camera_cfg.mount_pos_body[2]:.3f}) "
        f"pitch={math.degrees(camera_cfg.pitch_rad):.1f}deg "
        f"near={camera_cfg.near_plane:.3f}m "
        f"safe_clearance={camera_cfg.safe_clearance:.3f}m"
    )
    if args.reuse_vision_from is None:
        tqdm.write(f"Generating ground texture set ({args.texture_count} textures) ...")
        generate_texture_set(texture_dir, count=args.texture_count)
    else:
        tqdm.write(f"Reusing vision from {args.reuse_vision_from}; skipping texture generation and rendering.")

    n_chunks = len(raw_files)

    chunk_meta = []
    for file_path in raw_files:
        chunk_name = os.path.basename(file_path).split(".")[0]
        d = np.load(file_path, allow_pickle=True)
        chunk_meta.append((file_path, chunk_name, int(d["base_pos"].shape[0]), int(d["base_pos"].shape[1])))

    pending = []
    skipped = 0
    for file_path, chunk_name, N, T in chunk_meta:
        out_path = os.path.join(args.out_dir, f"{chunk_name}_rgb.h5")
        if not args.force and os.path.exists(out_path):
            try:
                with h5py.File(out_path, "r") as h5f:
                    if "vision" in h5f and "collisions" in h5f:
                        skipped += 1
                        continue
            except Exception:
                pass
        pending.append((file_path, chunk_name, N, T))

    if skipped:
        tqdm.write(f"Skipping {skipped}/{n_chunks} already-complete chunk(s).  Pass --force to re-render.")

    if not pending:
        tqdm.write("Nothing to render.")
        return

    total_envs = sum(N for _, _, N, _ in pending)

    with tqdm(
        total=total_envs,
        unit="env",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} envs  [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:

        for chunk_idx, (file_path, chunk_name, N, T) in enumerate(pending):
            out_path = os.path.join(args.out_dir, f"{chunk_name}_rgb.h5")

            # Cleanup stale tmp files
            stale_tmps = sorted(glob.glob(os.path.join(args.out_dir, f"{chunk_name}_tmp_*.h5")))
            for stale in stale_tmps:
                try:
                    os.remove(stale)
                except Exception as e:
                    tqdm.write(f"Warning: could not remove stale {os.path.basename(stale)}: {e}")
            if stale_tmps:
                tqdm.write(f"Removed {len(stale_tmps)} stale shard(s) from previous run.")

            stitching_tmp = out_path + ".stitching"
            if os.path.exists(stitching_tmp):
                try:
                    os.remove(stitching_tmp)
                except Exception:
                    pass

            chunk_label = f"{chunk_name}  [{chunk_idx + 1}/{len(pending)}]"
            pbar.set_description(chunk_label)

            raw_npz = np.load(file_path, allow_pickle=True)
            data = {key: np.asarray(raw_npz[key]) for key in raw_npz.files}

            # Extract obstacle layout JSON
            if "obstacle_layout" in data:
                obs_raw = data["obstacle_layout"]
                obstacle_json = str(obs_raw.item() if hasattr(obs_raw, "item") else obs_raw)
            else:
                obstacle_json = "[]"

            # Extract beacon layout JSON
            if "beacon_layout" in data:
                bcn_raw = data["beacon_layout"]
                beacon_json = str(bcn_raw.item() if hasattr(bcn_raw, "item") else bcn_raw)
            else:
                beacon_json = '{"beacons": [], "distractors": []}'

            if not args.skip_label_recompute:
                data.update(recompute_chunk_labels(data, obstacle_json, beacon_json))

            tasks = []
            tmp_files = []
            source_vision_path = None
            if args.reuse_vision_from is None:
                def build_chunk_tasks(worker_count: int, sim_backend: str, texture_variants: int) -> tuple[list[tuple], list[str]]:
                    chunk_tasks = []
                    chunk_tmp_files = []
                    envs_per_worker = math.ceil(N / worker_count)
                    for i in range(worker_count):
                        start = i * envs_per_worker
                        end = min(start + envs_per_worker, N)
                        if start >= end:
                            break
                        tmp = os.path.join(args.out_dir, f"{chunk_name}_tmp_{i}.h5")
                        chunk_tmp_files.append(tmp)
                        chunk_tasks.append((
                            i, file_path, start, end, tmp,
                            sim_backend, texture_dir, obstacle_json, beacon_json,
                            args.texture_count, texture_variants,
                            args.beacon_confuse_prob, args.img_res, tmp_vision_compression,
                            args.skip_physics_step, camera_cfg,
                        ))
                    return chunk_tasks, chunk_tmp_files

                tasks, tmp_files = build_chunk_tasks(
                    effective_workers, args.sim_backend, effective_texture_variants,
                )
                failed, missing_tmp, envs_received = run_render_processes(tasks, tmp_files, N, pbar)

                cpu_retry_attempted = False
                if failed or missing_tmp:
                    for tmp in tmp_files:
                        try:
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        except Exception:
                            pass

                    should_retry_on_cpu = rocm_runtime and backend_name != "cpu" and not unsafe_parallelism
                    if should_retry_on_cpu:
                        cpu_retry_attempted = True
                        if envs_received:
                            pbar.update(-envs_received)
                        tqdm.write(
                            f"{chunk_name}: backend={args.sim_backend} failed during worker startup "
                            "on ROCm; retrying serially on CPU."
                        )
                        pbar.set_description(f"{chunk_label}  cpu-retry...")
                        tasks, tmp_files = build_chunk_tasks(1, "cpu", 1)
                        failed, missing_tmp, _ = run_render_processes(tasks, tmp_files, N, pbar)

                    if failed or missing_tmp:
                        for tmp in tmp_files:
                            try:
                                if os.path.exists(tmp):
                                    os.remove(tmp)
                            except Exception:
                                pass
                        retry_note = " automatic_cpu_retry=True," if cpu_retry_attempted else ""
                        raise RuntimeError(
                            "Render worker failure before stitching. "
                            f"backend={args.sim_backend}, workers={effective_workers}, img_res={args.img_res}, "
                            f"{retry_note} failed_pids={failed}, missing_tmp_files={len(missing_tmp)}. "
                            "On AMD/ROCm this usually means Genesis hit the HIP allocator during scene.build; "
                            "use --sim_backend cpu or only re-enable parallel Vulkan if it is known stable "
                            "on this machine."
                        )
            else:
                source_vision_path = os.path.join(args.reuse_vision_from, f"{chunk_name}_rgb.h5")
                if not os.path.isfile(source_vision_path):
                    raise FileNotFoundError(
                        f"Missing source vision file for {chunk_name}: {source_vision_path}"
                    )
                pbar.update(N)

            pbar.set_description(f"{chunk_label}  stitching...")
            stitch_hdf5(
                out_path,
                tmp_files,
                tasks,
                data,
                N,
                T,
                args.img_res,
                final_vision_compression,
                vision_source_path=source_vision_path,
            )
            tqdm.write(f"  {chunk_name} done  ({N} envs, {T} steps)  ->  {out_path}")

    tqdm.write("All chunks rendered.")


if __name__ == "__main__":
    main()
