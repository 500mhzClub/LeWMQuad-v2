"""Shared egocentric camera geometry and safety helpers."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import math

import numpy as np


# The URDF camera mount sits farther forward (x=0.09055), which is too close to
# obstacle faces during collision-heavy rollouts. Keep the URDF-like height and
# pitch, but pull the optical centre back into the body envelope.
DEFAULT_EGO_CAMERA_MOUNT_POS_BODY = (0.04, 0.0, 0.07)
DEFAULT_EGO_CAMERA_PITCH_DEG = 15.0
DEFAULT_EGO_CAMERA_LOOKAT_DIST = 1.0
DEFAULT_EGO_CAMERA_FOV_DEG = 58.0
DEFAULT_EGO_CAMERA_NEAR_PLANE = 0.01
DEFAULT_EGO_CAMERA_POS_JITTER = 0.008
DEFAULT_EGO_CAMERA_LOOKAT_JITTER = 0.012
DEFAULT_EGO_CAMERA_INSIDE_MARGIN = 0.02


@dataclass(frozen=True)
class EgoCameraConfig:
    mount_pos_body: tuple[float, float, float]
    pitch_rad: float
    lookat_dist: float
    near_plane: float
    fov_deg: float
    pos_jitter: float = 0.0
    lookat_jitter: float = 0.0
    inside_margin: float = DEFAULT_EGO_CAMERA_INSIDE_MARGIN

    @property
    def safe_clearance(self) -> float:
        return max(0.0, float(self.near_plane) + float(self.pos_jitter))


def add_egocentric_camera_args(parser: argparse.ArgumentParser, *, include_jitter: bool = False) -> None:
    """Add a shared set of egocentric camera CLI flags."""
    parser.add_argument("--cam_mount_x", type=float, default=DEFAULT_EGO_CAMERA_MOUNT_POS_BODY[0])
    parser.add_argument("--cam_mount_y", type=float, default=DEFAULT_EGO_CAMERA_MOUNT_POS_BODY[1])
    parser.add_argument("--cam_mount_z", type=float, default=DEFAULT_EGO_CAMERA_MOUNT_POS_BODY[2])
    parser.add_argument("--cam_pitch_deg", type=float, default=DEFAULT_EGO_CAMERA_PITCH_DEG)
    parser.add_argument("--cam_lookat_dist", type=float, default=DEFAULT_EGO_CAMERA_LOOKAT_DIST)
    parser.add_argument("--cam_fov", type=float, default=DEFAULT_EGO_CAMERA_FOV_DEG)
    parser.add_argument("--cam_near", type=float, default=DEFAULT_EGO_CAMERA_NEAR_PLANE)
    if include_jitter:
        parser.add_argument("--cam_pos_jitter", type=float, default=DEFAULT_EGO_CAMERA_POS_JITTER)
        parser.add_argument("--cam_lookat_jitter", type=float, default=DEFAULT_EGO_CAMERA_LOOKAT_JITTER)


def ego_camera_config_from_args(args: argparse.Namespace, *, include_jitter: bool = False) -> EgoCameraConfig:
    """Build a shared camera config from parsed CLI args."""
    return EgoCameraConfig(
        mount_pos_body=(
            float(args.cam_mount_x),
            float(args.cam_mount_y),
            float(args.cam_mount_z),
        ),
        pitch_rad=math.radians(float(args.cam_pitch_deg)),
        lookat_dist=float(args.cam_lookat_dist),
        near_plane=float(args.cam_near),
        fov_deg=float(args.cam_fov),
        pos_jitter=float(args.cam_pos_jitter) if include_jitter else 0.0,
        lookat_jitter=float(args.cam_lookat_jitter) if include_jitter else 0.0,
    )


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion into a 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


def rotmat_pitch_y(pitch_rad: float) -> np.ndarray:
    c = math.cos(float(pitch_rad))
    s = math.sin(float(pitch_rad))
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.copy()
    return vec / norm


def egocentric_camera_pose(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    config: EgoCameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cam_pos, cam_lookat, cam_up, cam_forward) from base pose."""
    base_pos = np.asarray(base_pos, dtype=np.float32)
    base_quat = np.asarray(base_quat, dtype=np.float32)
    body_rot = quat_to_rotmat_wxyz(base_quat)
    cam_rot = body_rot @ rotmat_pitch_y(config.pitch_rad)

    mount_pos_body = np.asarray(config.mount_pos_body, dtype=np.float32)
    cam_pos = base_pos + body_rot @ mount_pos_body
    cam_forward = normalize_vec(cam_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
    cam_up = normalize_vec(cam_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    cam_lookat = cam_pos + float(config.lookat_dist) * cam_forward
    return cam_pos.astype(np.float32), cam_lookat.astype(np.float32), cam_up.astype(np.float32), cam_forward.astype(np.float32)


def camera_inside_any_obstacle(
    cam_pos: np.ndarray,
    layout,
    margin: float = DEFAULT_EGO_CAMERA_INSIDE_MARGIN,
) -> bool:
    """Return True if the camera position lies inside any obstacle AABB."""
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cx, cy, cz = [float(v) for v in cam_pos[:3]]
    for obs in layout.obstacles:
        ox, oy, oz = [float(v) for v in obs.pos[:3]]
        hx = float(obs.size[0]) / 2.0 + float(margin)
        hy = float(obs.size[1]) / 2.0 + float(margin)
        hz = float(obs.size[2]) / 2.0
        if abs(cx - ox) < hx and abs(cy - oy) < hy and abs(cz - oz) < hz:
            return True
    return False


def camera_clearance_to_any_obstacle(cam_pos: np.ndarray, layout) -> float:
    """Return Euclidean clearance from the camera point to the nearest obstacle AABB."""
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cx, cy, cz = [float(v) for v in cam_pos[:3]]
    min_clearance = float("inf")
    for obs in layout.obstacles:
        ox, oy, oz = [float(v) for v in obs.pos[:3]]
        hx = float(obs.size[0]) / 2.0
        hy = float(obs.size[1]) / 2.0
        hz = float(obs.size[2]) / 2.0
        dx = max(abs(cx - ox) - hx, 0.0)
        dy = max(abs(cy - oy) - hy, 0.0)
        dz = max(abs(cz - oz) - hz, 0.0)
        min_clearance = min(min_clearance, math.sqrt(dx * dx + dy * dy + dz * dz))
    return min_clearance


def forward_hit_distance_to_any_obstacle(cam_pos: np.ndarray, cam_forward: np.ndarray, layout) -> float:
    """Return distance along the camera forward ray to the nearest obstacle AABB."""
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_forward = normalize_vec(cam_forward)
    min_distance = float("inf")
    eps = 1e-8

    for obs in layout.obstacles:
        box_min = np.asarray(obs.pos, dtype=np.float32) - 0.5 * np.asarray(obs.size, dtype=np.float32)
        box_max = np.asarray(obs.pos, dtype=np.float32) + 0.5 * np.asarray(obs.size, dtype=np.float32)

        t_min = -float("inf")
        t_max = float("inf")
        hit = True
        for axis in range(3):
            origin = float(cam_pos[axis])
            direction = float(cam_forward[axis])
            lo = float(box_min[axis])
            hi = float(box_max[axis])

            if abs(direction) < eps:
                if origin < lo or origin > hi:
                    hit = False
                    break
                continue

            t0 = (lo - origin) / direction
            t1 = (hi - origin) / direction
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                hit = False
                break

        if not hit or t_max < 0.0:
            continue
        min_distance = min(min_distance, max(0.0, t_min))

    return min_distance


def camera_safety_metrics(cam_pos: np.ndarray, cam_forward: np.ndarray, layout, config: EgoCameraConfig) -> dict[str, float | bool]:
    """Return shared camera safety metrics for frame validation."""
    inside_wall = camera_inside_any_obstacle(cam_pos, layout, margin=config.inside_margin)
    clearance = camera_clearance_to_any_obstacle(cam_pos, layout)
    forward_hit = forward_hit_distance_to_any_obstacle(cam_pos, cam_forward, layout)
    unsafe = inside_wall or clearance < config.safe_clearance or forward_hit < config.safe_clearance
    return {
        "inside_wall": inside_wall,
        "clearance": clearance,
        "forward_hit": forward_hit,
        "unsafe": unsafe,
    }
