"""Label computation for training data enrichment.

Computes per-timestep supervision signals that go beyond the raw JEPA
prediction objective:

- **clearance**: min distance to any obstacle AABB (near-miss detection)
- **beacon_visible**: binary mask — is any beacon panel in the camera FOV?
- **beacon_identity**: one-hot or index of the closest visible beacon
- **beacon_bearing**: relative angle from robot heading to beacon centre
- **beacon_range**: Euclidean distance to closest visible beacon
- **traversability**: can the robot move forward N steps without collision?

These labels are stored alongside vision/proprio/cmds in HDF5 and optionally
used as auxiliary prediction targets or reward shaping signals.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .obstacle_utils import ObstacleSpec, ObstacleLayout
from .beacon_utils import BeaconSpec, BeaconLayout, BEACON_FAMILIES


# --------------------------------------------------------------------------- #
# Clearance (min distance to obstacle AABBs)
# --------------------------------------------------------------------------- #

def _aabb_distance_2d(
    px: float, py: float,
    cx: float, cy: float,
    hx: float, hy: float,
) -> float:
    """Signed distance from point (px,py) to AABB centre (cx,cy) half-extents (hx,hy).

    Returns 0 if inside the box, positive distance otherwise.
    """
    dx = max(abs(px - cx) - hx, 0.0)
    dy = max(abs(py - cy) - hy, 0.0)
    return math.sqrt(dx * dx + dy * dy)


def compute_clearance(
    robot_xy: np.ndarray,
    layout: ObstacleLayout,
) -> np.ndarray:
    """Min 2-D distance from each robot position to any obstacle AABB.

    Args:
        robot_xy: (N, 2) array of robot XY positions.
        layout: obstacle layout for the scene.

    Returns:
        (N,) array of minimum clearance values (metres).  0 means inside an
        obstacle (collision).
    """
    N = robot_xy.shape[0]
    min_dist = np.full(N, float("inf"), dtype=np.float32)

    for obs in layout.obstacles:
        cx, cy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
        dx = np.maximum(np.abs(robot_xy[:, 0] - cx) - hx, 0.0)
        dy = np.maximum(np.abs(robot_xy[:, 1] - cy) - hy, 0.0)
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def compute_near_miss(
    clearance: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """Boolean mask: is the robot within *threshold* of an obstacle but not colliding?

    Args:
        clearance: (N,) clearance array from :func:`compute_clearance`.
        threshold: near-miss distance (metres).

    Returns:
        (N,) bool array.
    """
    return (clearance > 0.0) & (clearance < threshold)


# --------------------------------------------------------------------------- #
# Beacon visibility and bearing/range
# --------------------------------------------------------------------------- #

def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def compute_beacon_labels(
    robot_xy: np.ndarray,
    robot_yaw: np.ndarray,
    beacon_layout: BeaconLayout,
    fov_deg: float = 58.0,
    max_range: float = 5.0,
) -> dict:
    """Compute per-timestep beacon observation labels.

    For each timestep, finds the closest *visible* beacon (within FOV and
    range) and records its identity, bearing, and range.

    Args:
        robot_xy: (N, 2) robot XY positions.
        robot_yaw: (N,) robot heading in radians.
        beacon_layout: layout containing beacon placements.
        fov_deg: camera horizontal field of view in degrees.
        max_range: maximum beacon detection range (metres).

    Returns:
        dict with:
            beacon_visible: (N,) bool
            beacon_identity: (N,) int — index into BEACON_FAMILIES keys (-1 if none)
            beacon_bearing: (N,) float — relative bearing in radians (0 = straight ahead)
            beacon_range: (N,) float — distance in metres (inf if none visible)
    """
    N = robot_xy.shape[0]
    identity_names = list(BEACON_FAMILIES.keys())

    visible = np.zeros(N, dtype=bool)
    identity = np.full(N, -1, dtype=np.int32)
    bearing = np.zeros(N, dtype=np.float32)
    brange = np.full(N, float("inf"), dtype=np.float32)

    half_fov = math.radians(fov_deg / 2.0)

    for beacon in beacon_layout.beacons:
        bx, by = beacon.pos[0], beacon.pos[1]
        bid = identity_names.index(beacon.identity) if beacon.identity in identity_names else -1

        dx = bx - robot_xy[:, 0]
        dy = by - robot_xy[:, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        # Bearing relative to robot heading
        abs_angle = np.arctan2(dy, dx)
        rel_bearing = np.array([_angle_diff(float(a), float(y))
                                for a, y in zip(abs_angle, robot_yaw)],
                               dtype=np.float32)

        # Visibility: within FOV and within range
        in_fov = np.abs(rel_bearing) < half_fov
        in_range = dist < max_range
        is_visible = in_fov & in_range

        # Update closest visible beacon
        closer = is_visible & (dist < brange)
        visible[closer] = True
        identity[closer] = bid
        bearing[closer] = rel_bearing[closer]
        brange[closer] = dist[closer]

    return {
        "beacon_visible": visible,
        "beacon_identity": identity,
        "beacon_bearing": bearing,
        "beacon_range": brange,
    }


# --------------------------------------------------------------------------- #
# Traversability (forward clearance over future steps)
# --------------------------------------------------------------------------- #

def compute_traversability(
    robot_xy: np.ndarray,
    robot_yaw: np.ndarray,
    layout: ObstacleLayout,
    horizon: int = 10,
    step_size: float = 0.02,
    collision_margin: float = 0.15,
) -> np.ndarray:
    """How many steps the robot can move forward before hitting an obstacle.

    Simulates straight-line forward motion from each position and counts steps
    until the clearance drops below ``collision_margin``.

    Args:
        robot_xy: (N, 2) robot positions.
        robot_yaw: (N,) heading in radians.
        layout: obstacle layout.
        horizon: max look-ahead steps.
        step_size: forward distance per step (metres).
        collision_margin: clearance threshold for "blocked".

    Returns:
        (N,) int array — number of clear steps (0 to horizon).
    """
    N = robot_xy.shape[0]
    traversable = np.full(N, horizon, dtype=np.int32)

    cos_y = np.cos(robot_yaw)
    sin_y = np.sin(robot_yaw)

    for step in range(1, horizon + 1):
        future_xy = robot_xy + step * step_size * np.stack([cos_y, sin_y], axis=-1)
        clearance = compute_clearance(future_xy, layout)
        blocked = clearance < collision_margin
        # For robots that just became blocked at this step
        first_block = blocked & (traversable == horizon)
        traversable[first_block] = step - 1

    return traversable


# --------------------------------------------------------------------------- #
# Batch label computation for full episode
# --------------------------------------------------------------------------- #

def compute_episode_labels(
    robot_xy: np.ndarray,
    robot_yaw: np.ndarray,
    obstacle_layout: ObstacleLayout,
    beacon_layout: Optional[BeaconLayout] = None,
    near_miss_threshold: float = 0.20,
    fov_deg: float = 58.0,
    traversability_horizon: int = 10,
) -> dict:
    """Compute all labels for a full episode.

    Args:
        robot_xy: (T, 2) robot XY positions over time.
        robot_yaw: (T,) robot heading over time.
        obstacle_layout: obstacles in the scene.
        beacon_layout: optional beacon placements.
        near_miss_threshold: clearance threshold for near-miss flag.
        fov_deg: camera FOV for beacon visibility.
        traversability_horizon: forward look-ahead steps.

    Returns:
        dict with all label arrays, each of shape (T,) or (T, ...).
    """
    labels = {}

    # Clearance
    clearance = compute_clearance(robot_xy, obstacle_layout)
    labels["clearance"] = clearance
    labels["near_miss"] = compute_near_miss(clearance, near_miss_threshold)

    # Traversability
    labels["traversability"] = compute_traversability(
        robot_xy, robot_yaw, obstacle_layout,
        horizon=traversability_horizon,
    )

    # Beacon labels
    if beacon_layout is not None and len(beacon_layout.beacons) > 0:
        beacon_labels = compute_beacon_labels(
            robot_xy, robot_yaw, beacon_layout, fov_deg=fov_deg,
        )
        labels.update(beacon_labels)
    else:
        T = robot_xy.shape[0]
        labels["beacon_visible"] = np.zeros(T, dtype=bool)
        labels["beacon_identity"] = np.full(T, -1, dtype=np.int32)
        labels["beacon_bearing"] = np.zeros(T, dtype=np.float32)
        labels["beacon_range"] = np.full(T, float("inf"), dtype=np.float32)

    return labels
