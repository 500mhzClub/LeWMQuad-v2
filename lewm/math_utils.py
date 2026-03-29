"""Quaternion and coordinate frame utilities."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# Numpy quaternion / rotation helpers
# --------------------------------------------------------------------------- #

def yaw_to_quat(yaw_rad: float) -> np.ndarray:
    """Yaw angle (rad) -> wxyz quaternion."""
    half = 0.5 * yaw_rad
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_to_yaw(q: np.ndarray) -> float:
    """wxyz quaternion -> yaw angle (rad)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_to_world_xy(yaw: float, v_body_xy: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy = float(v_body_xy[0]), float(v_body_xy[1])
    return np.array([c * vx - s * vy, s * vx + c * vy], dtype=np.float32)


def world_to_body_xy(yaw: float, v_world_xy: np.ndarray) -> np.ndarray:
    return body_to_world_xy(-yaw, v_world_xy)


def forward_up_from_quat(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """wxyz quaternion -> (forward, up) unit vectors."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    fw = np.array([
        1 - 2 * (y ** 2 + z ** 2),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dtype=np.float32)
    up = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x ** 2 + y ** 2),
    ], dtype=np.float32)
    return fw, up


# --------------------------------------------------------------------------- #
# Torch quaternion helpers (batched, wxyz convention)
# --------------------------------------------------------------------------- #

def quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0],
    ], dim=-1)


def world_to_body_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame 3-vector into the body frame (batched)."""
    q_conj = quat_conj_wxyz(quat)
    vq = torch.cat([torch.zeros((vec.shape[0], 1), device=vec.device), vec], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q_conj, vq), quat)[:, 1:4]
