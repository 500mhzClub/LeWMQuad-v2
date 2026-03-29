"""Shared egocentric camera geometry and safety helpers."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import math

import numpy as np


# Use the actual URDF camera mount. Clipping should be handled by camera near
# plane and frame safety checks, not by moving the optical centre inside the dog.
DEFAULT_EGO_CAMERA_MOUNT_POS_BODY = (0.09055, 0.0, 0.07)
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


def _ray_hit_distance_to_any_obstacle(origin: np.ndarray, direction: np.ndarray, layout) -> float:
    """Return distance along a single ray to the nearest obstacle AABB (slab method)."""
    origin = np.asarray(origin, dtype=np.float32)
    direction = normalize_vec(direction)
    min_distance = float("inf")
    eps = 1e-8

    for obs in layout.obstacles:
        box_min = np.asarray(obs.pos, dtype=np.float32) - 0.5 * np.asarray(obs.size, dtype=np.float32)
        box_max = np.asarray(obs.pos, dtype=np.float32) + 0.5 * np.asarray(obs.size, dtype=np.float32)

        t_min = -float("inf")
        t_max = float("inf")
        hit = True
        for axis in range(3):
            o = float(origin[axis])
            d = float(direction[axis])
            lo = float(box_min[axis])
            hi = float(box_max[axis])

            if abs(d) < eps:
                if o < lo or o > hi:
                    hit = False
                    break
                continue

            t0 = (lo - o) / d
            t1 = (hi - o) / d
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


def forward_hit_distance_to_any_obstacle(cam_pos: np.ndarray, cam_forward: np.ndarray, layout) -> float:
    """Return distance along the camera forward ray to the nearest obstacle AABB."""
    return _ray_hit_distance_to_any_obstacle(cam_pos, cam_forward, layout)


def frustum_ray_directions(fov_deg: float) -> list[np.ndarray]:
    """Return 9 unit-vector ray directions spanning the camera frustum in camera-local frame.

    Camera-local convention: +X = forward, +Y = right, +Z = up.
    Rays: center, 4 corners, 4 edge midpoints of the near-plane rectangle.
    """
    half_angle = math.radians(fov_deg / 2.0)
    t = math.tan(half_angle)

    raw = [
        np.array([1.0, 0.0, 0.0]),       # center
        np.array([1.0, -t, +t]),          # top-left
        np.array([1.0, +t, +t]),          # top-right
        np.array([1.0, -t, -t]),          # bottom-left
        np.array([1.0, +t, -t]),          # bottom-right
        np.array([1.0, -t, 0.0]),         # mid-left
        np.array([1.0, +t, 0.0]),         # mid-right
        np.array([1.0, 0.0, +t]),         # mid-top
        np.array([1.0, 0.0, -t]),         # mid-bottom
    ]
    return [normalize_vec(r.astype(np.float32)) for r in raw]


def frustum_min_hit_distance(
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fov_deg: float,
    layout,
) -> float:
    """Cast rays through 9 frustum sample points; return the minimum hit distance.

    ``cam_rot`` is the 3×3 camera rotation matrix (columns = camera-local axes
    expressed in world frame).  The frustum ray directions are transformed to
    world space via this matrix before ray-AABB intersection.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_rot = np.asarray(cam_rot, dtype=np.float32)
    local_dirs = frustum_ray_directions(fov_deg)
    min_hit = float("inf")
    for local_d in local_dirs:
        world_d = cam_rot @ local_d
        hit = _ray_hit_distance_to_any_obstacle(cam_pos, world_d, layout)
        if hit < min_hit:
            min_hit = hit
    return min_hit


def camera_rotation_matrix(base_quat: np.ndarray, pitch_rad: float) -> np.ndarray:
    """Return the 3×3 world-frame camera rotation from base quaternion and pitch."""
    body_rot = quat_to_rotmat_wxyz(base_quat)
    return body_rot @ rotmat_pitch_y(pitch_rad)


# ---- Camera retraction ---------------------------------------------------- #

MAX_RETRACT_M = 0.06  # never retract further than 6 cm (keeps camera in body)
RETRACT_MARGIN_M = 0.005  # extra margin on top of safe_clearance


def retract_camera_to_safe(
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    cam_up: np.ndarray,
    cam_rot: np.ndarray,
    layout,
    config: EgoCameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Pull the camera backward along -cam_forward until the frustum clears.

    Returns (new_cam_pos, new_cam_lookat, cam_up, cam_forward, retract_dist).
    cam_forward and cam_up are unchanged (translation only).
    retract_dist is 0.0 if no retraction was needed.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_forward = np.asarray(cam_forward, dtype=np.float32)
    cam_up = np.asarray(cam_up, dtype=np.float32)
    cam_rot = np.asarray(cam_rot, dtype=np.float32)
    safe = config.safe_clearance

    # Quick check: is frustum already safe?
    inside = camera_inside_any_obstacle(cam_pos, layout, margin=config.inside_margin)
    fmin = frustum_min_hit_distance(cam_pos, cam_rot, config.fov_deg, layout)
    if fmin >= safe and not inside:
        lookat = cam_pos + float(config.lookat_dist) * cam_forward
        return cam_pos, lookat, cam_up, cam_forward, 0.0

    # Camera inside obstacle — need full retraction to escape
    if inside:
        retract = MAX_RETRACT_M
    elif fmin < float("inf"):
        needed = safe - fmin + RETRACT_MARGIN_M
        retract = min(max(needed, 0.005), MAX_RETRACT_M)
    else:
        retract = MAX_RETRACT_M

    new_pos = cam_pos - retract * cam_forward
    lookat = new_pos + float(config.lookat_dist) * cam_forward
    return new_pos.astype(np.float32), lookat.astype(np.float32), cam_up, cam_forward, float(retract)


# ---- Depth-buffer clipping check ----------------------------------------- #

def depth_buffer_has_clipping(depth: np.ndarray, near_plane: float, frac_threshold: float = 0.005) -> bool:
    """Return True if a meaningful fraction of depth-buffer pixels are at or below the near plane.

    ``depth`` is an (H, W) float array of per-pixel depth values from the renderer.
    ``frac_threshold`` controls how many near-plane pixels constitute clipping (default 0.5%).
    """
    depth = np.asarray(depth, dtype=np.float32)
    at_near = (depth <= near_plane * 1.05)  # 5% tolerance
    return float(at_near.mean()) >= frac_threshold


def camera_safety_metrics(
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    layout,
    config: EgoCameraConfig,
    cam_rot: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Return shared camera safety metrics for frame validation.

    If ``cam_rot`` (3×3 camera rotation matrix) is provided, a full frustum
    multi-ray check is used.  Otherwise falls back to single forward ray
    (backward compatible).
    """
    inside_wall = camera_inside_any_obstacle(cam_pos, layout, margin=config.inside_margin)
    clearance = camera_clearance_to_any_obstacle(cam_pos, layout)
    forward_hit = forward_hit_distance_to_any_obstacle(cam_pos, cam_forward, layout)

    if cam_rot is not None:
        fmin = frustum_min_hit_distance(cam_pos, cam_rot, config.fov_deg, layout)
    else:
        fmin = forward_hit

    unsafe = inside_wall or clearance < config.safe_clearance or fmin < config.safe_clearance
    return {
        "inside_wall": inside_wall,
        "clearance": clearance,
        "forward_hit": forward_hit,
        "frustum_min_hit": fmin,
        "unsafe": unsafe,
    }
