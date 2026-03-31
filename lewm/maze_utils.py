"""Maze topology generators for structured navigation training.

Each generator returns an ``ObstacleLayout`` (walls) and an optional list of
``BeaconSpec`` placements for goal / landmark beacons within the maze.

Maze types
----------
- T-junction, 4-way crossroads, S-bend, zig-zag corridor
- One-turn / two-turn mazes, multi-room with loops
- Branching corridors, cul-de-sacs (dead-end pockets)
- Symmetric corridors, long straight corridors, doorway/narrow-gap passages

All coordinates are in the arena frame with the robot spawning near the origin.
Walls are axis-aligned boxes compatible with Genesis ``gs.morphs.Box``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .obstacle_utils import ObstacleSpec, ObstacleLayout, _random_color
from .beacon_utils import BeaconSpec, BeaconLayout, make_beacon_panel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _wall(
    cx: float, cy: float, length: float, thickness: float, height: float,
    axis: str, color: Tuple[float, float, float],
) -> ObstacleSpec:
    """Create a single axis-aligned wall segment."""
    hz = height / 2.0
    if axis == "x":
        return ObstacleSpec(pos=(cx, cy, hz), size=(length, thickness, height), color=color)
    else:
        return ObstacleSpec(pos=(cx, cy, hz), size=(thickness, length, height), color=color)


def _clears_origin(obs: ObstacleSpec, clearance: float) -> bool:
    cx, cy = obs.pos[0], obs.pos[1]
    hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
    dx = max(abs(cx) - hx, 0.0)
    dy = max(abs(cy) - hy, 0.0)
    return (dx * dx + dy * dy) >= clearance * clearance


def _point_clearance_xy(point_xy: np.ndarray, obstacles: List[ObstacleSpec]) -> float:
    """Minimum XY clearance from a point to any obstacle AABB."""
    px, py = float(point_xy[0]), float(point_xy[1])
    min_dist = float("inf")
    for obs in obstacles:
        cx, cy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
        dx = max(abs(px - cx) - hx, 0.0)
        dy = max(abs(py - cy) - hy, 0.0)
        min_dist = min(min_dist, math.sqrt(dx * dx + dy * dy))
    return min_dist


def _wall_candidate_normals(obs: ObstacleSpec) -> List[Tuple[float, float]]:
    """Two possible outward-facing normals for an axis-aligned wall."""
    if obs.size[0] < obs.size[1]:
        return [(1.0, 0.0), (-1.0, 0.0)]
    return [(0.0, 1.0), (0.0, -1.0)]


def _build_free_space_grid(
    obstacles: List[ObstacleSpec],
    arena_half: float,
    clearance_margin: float,
    resolution: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(-arena_half, arena_half + 0.5 * resolution, resolution, dtype=np.float32)
    ys = np.arange(-arena_half, arena_half + 0.5 * resolution, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    min_dist = np.full(points.shape[0], float("inf"), dtype=np.float32)

    for obs in obstacles:
        cx, cy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
        dx = np.maximum(np.abs(points[:, 0] - cx) - hx, 0.0)
        dy = np.maximum(np.abs(points[:, 1] - cy) - hy, 0.0)
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = np.minimum(min_dist, dist)

    free = (min_dist >= float(clearance_margin)).reshape(len(ys), len(xs))
    return xs, ys, free


def _nearest_free_cell(
    free: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    point_xy: np.ndarray,
    max_search_radius: int = 8,
) -> Optional[Tuple[int, int]]:
    ix = int(np.argmin(np.abs(xs - float(point_xy[0]))))
    iy = int(np.argmin(np.abs(ys - float(point_xy[1]))))
    if bool(free[iy, ix]):
        return iy, ix

    h, w = free.shape
    best_cell: Optional[Tuple[int, int]] = None
    best_dist = float("inf")
    for radius in range(1, max_search_radius + 1):
        y0 = max(0, iy - radius)
        y1 = min(h, iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(w, ix + radius + 1)
        for cy in range(y0, y1):
            for cx in range(x0, x1):
                if not bool(free[cy, cx]):
                    continue
                dist = float((xs[cx] - point_xy[0]) ** 2 + (ys[cy] - point_xy[1]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = (cy, cx)
        if best_cell is not None:
            return best_cell
    return None


def _has_free_path(
    free: np.ndarray,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
) -> bool:
    if start_cell == goal_cell:
        return True

    h, w = free.shape
    visited = np.zeros((h, w), dtype=bool)
    frontier: List[Tuple[int, int]] = [start_cell]
    visited[start_cell] = True
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while frontier:
        next_frontier: List[Tuple[int, int]] = []
        for cy, cx in frontier:
            for dy, dx in neighbors:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not bool(free[ny, nx]):
                    continue
                if (ny, nx) == goal_cell:
                    return True
                visited[ny, nx] = True
                next_frontier.append((ny, nx))
        frontier = next_frontier

    return False


def _choose_accessible_wall_face(
    obs: ObstacleSpec,
    obstacles: List[ObstacleSpec],
    robot_clearance: float,
    arena_half: float,
) -> Tuple[float, float]:
    """Pick the wall face with the most open free space in front of it."""
    half_thickness = 0.5 * min(float(obs.size[0]), float(obs.size[1]))
    clearance_margin = max(0.14, min(float(robot_clearance), 0.15))
    standoff = max(clearance_margin, 0.05)
    xs, ys, free = _build_free_space_grid(
        obstacles=obstacles,
        arena_half=float(arena_half),
        clearance_margin=clearance_margin,
    )
    start_cell = _nearest_free_cell(
        free=free,
        xs=xs,
        ys=ys,
        point_xy=np.zeros(2, dtype=np.float32),
    )

    best_normal = _wall_candidate_normals(obs)[0]
    best_score = -float("inf")
    for nx, ny in _wall_candidate_normals(obs):
        sample_xy = np.asarray(
            [
                float(obs.pos[0]) + float(nx) * (half_thickness + standoff),
                float(obs.pos[1]) + float(ny) * (half_thickness + standoff),
            ],
            dtype=np.float32,
        )
        clearance = _point_clearance_xy(sample_xy, obstacles)
        goal_cell = _nearest_free_cell(free=free, xs=xs, ys=ys, point_xy=sample_xy)
        reachable = (
            start_cell is not None
            and goal_cell is not None
            and _has_free_path(free, start_cell, goal_cell)
        )
        # Tie-break toward the side closer to the origin corridor the mazes are
        # designed around, while still preferring the more open face first.
        score = (
            (10.0 if reachable else 0.0)
            + float(clearance)
            - 0.10 * float(np.linalg.norm(sample_xy))
        )
        if score > best_score:
            best_score = score
            best_normal = (float(nx), float(ny))

    return best_normal


@dataclass
class MazeResult:
    """Output of a maze generator: walls + optional beacon placements."""
    layout: ObstacleLayout
    beacons: List[BeaconSpec] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Individual maze generators
# --------------------------------------------------------------------------- #

def _t_junction(
    rng: np.random.RandomState,
    cx: float, cy: float,
    arm_len: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """T-junction: a main corridor with a perpendicular branch.

    The main corridor runs along X, the branch goes in +Y from the midpoint.
    ::

        ═══════╦═══════
               ║
               ║
    """
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # Main corridor walls (left and right of branch)
    # Top wall left segment
    walls.append(_wall(cx - arm_len / 2.0, cy + hw, arm_len, thickness, height, "x", color))
    # Top wall right segment
    walls.append(_wall(cx + arm_len / 2.0, cy + hw, arm_len, thickness, height, "x", color))
    # Bottom wall — full span
    walls.append(_wall(cx, cy - hw, arm_len * 2.0, thickness, height, "x", color))
    # Branch walls (going +Y)
    branch_len = arm_len * rng.uniform(0.6, 1.0)
    walls.append(_wall(cx - hw, cy + hw + branch_len / 2.0, branch_len, thickness, height, "y", color))
    walls.append(_wall(cx + hw, cy + hw + branch_len / 2.0, branch_len, thickness, height, "y", color))
    # Cap at end of branch
    walls.append(_wall(cx, cy + hw + branch_len, corridor_w, thickness, height, "x", color))

    return walls


def _crossroads(
    rng: np.random.RandomState,
    cx: float, cy: float,
    arm_len: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """4-way crossroads (+ shape): four corridors meeting at an intersection.

    Each arm extends ``arm_len`` from the centre.  The intersection itself is
    a ``corridor_w × corridor_w`` open square.
    """
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # Four L-shaped corner pieces
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # Horizontal piece (runs along X from corner outward)
        walls.append(_wall(
            cx + sx * (hw + arm_len / 2.0), cy + sy * hw,
            arm_len, thickness, height, "x", color,
        ))
        # Vertical piece (runs along Y from corner outward)
        walls.append(_wall(
            cx + sx * hw, cy + sy * (hw + arm_len / 2.0),
            arm_len, thickness, height, "y", color,
        ))

    # End caps for each arm
    walls.append(_wall(cx - hw - arm_len, cy, corridor_w, thickness, height, "y", color))  # -X cap
    walls.append(_wall(cx + hw + arm_len, cy, corridor_w, thickness, height, "y", color))  # +X cap
    walls.append(_wall(cx, cy - hw - arm_len, corridor_w, thickness, height, "x", color))  # -Y cap
    walls.append(_wall(cx, cy + hw + arm_len, corridor_w, thickness, height, "x", color))  # +Y cap

    return walls


def _s_bend(
    rng: np.random.RandomState,
    cx: float, cy: float,
    seg_len: float, corridor_w: float, offset: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """S-bend: two straight segments connected by a lateral shift.

    ::

        ══════╗
              ║
              ╚══════
    """
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # First straight (going +X)
    walls.append(_wall(cx - seg_len / 2.0, cy + hw, seg_len, thickness, height, "x", color))
    walls.append(_wall(cx - seg_len / 2.0, cy - hw, seg_len, thickness, height, "x", color))

    # Connecting jog (going +Y by offset)
    jog_x = cx
    walls.append(_wall(jog_x - hw, cy + offset / 2.0, offset + corridor_w, thickness, height, "y", color))
    walls.append(_wall(jog_x + hw, cy + offset / 2.0, offset + corridor_w, thickness, height, "y", color))

    # Second straight (shifted by offset in Y)
    cy2 = cy + offset
    walls.append(_wall(cx + seg_len / 2.0, cy2 + hw, seg_len, thickness, height, "x", color))
    walls.append(_wall(cx + seg_len / 2.0, cy2 - hw, seg_len, thickness, height, "x", color))

    return walls


def _zigzag(
    rng: np.random.RandomState,
    cx: float, cy: float,
    n_zigs: int, seg_len: float, corridor_w: float, offset: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """Zig-zag corridor: alternating lateral shifts."""
    walls: List[ObstacleSpec] = []
    hw = corridor_w / 2.0
    cur_x = cx
    cur_y = cy
    direction = 1.0

    for i in range(n_zigs):
        # Straight segment along X
        walls.append(_wall(cur_x + seg_len / 2.0, cur_y + hw, seg_len, thickness, height, "x", color))
        walls.append(_wall(cur_x + seg_len / 2.0, cur_y - hw, seg_len, thickness, height, "x", color))
        cur_x += seg_len

        if i < n_zigs - 1:
            # Lateral jog
            jog = direction * offset
            walls.append(_wall(cur_x - hw, cur_y + jog / 2.0, abs(jog) + corridor_w, thickness, height, "y", color))
            walls.append(_wall(cur_x + hw, cur_y + jog / 2.0, abs(jog) + corridor_w, thickness, height, "y", color))
            cur_y += jog
            direction *= -1.0

    return walls


def _one_turn(
    rng: np.random.RandomState,
    cx: float, cy: float,
    leg1_len: float, leg2_len: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """Single 90-degree turn corridor (L-shaped passage)."""
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # First leg along +X
    walls.append(_wall(cx + leg1_len / 2.0, cy + hw, leg1_len, thickness, height, "x", color))
    walls.append(_wall(cx + leg1_len / 2.0, cy - hw, leg1_len + corridor_w, thickness, height, "x", color))

    # Turn corner: second leg along +Y
    turn_x = cx + leg1_len
    walls.append(_wall(turn_x + hw, cy + leg2_len / 2.0, leg2_len, thickness, height, "y", color))
    walls.append(_wall(turn_x - hw, cy + leg2_len / 2.0, leg2_len + corridor_w, thickness, height, "y", color))

    return walls


def _two_turn(
    rng: np.random.RandomState,
    cx: float, cy: float,
    leg_lens: Tuple[float, float, float], corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """Two 90-degree turns forming a Z or U-shaped corridor."""
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []
    l1, l2, l3 = leg_lens

    # Leg 1 along +X
    walls.append(_wall(cx + l1 / 2.0, cy + hw, l1, thickness, height, "x", color))
    walls.append(_wall(cx + l1 / 2.0, cy - hw, l1 + corridor_w, thickness, height, "x", color))

    # Turn up: Leg 2 along +Y
    t1x = cx + l1
    walls.append(_wall(t1x + hw, cy + l2 / 2.0, l2, thickness, height, "y", color))
    walls.append(_wall(t1x - hw, cy + l2 / 2.0, l2 + corridor_w, thickness, height, "y", color))

    # Turn right again: Leg 3 along +X
    t2y = cy + l2
    walls.append(_wall(t1x + l3 / 2.0, t2y + hw, l3 + corridor_w, thickness, height, "x", color))
    walls.append(_wall(t1x + l3 / 2.0, t2y - hw, l3, thickness, height, "x", color))

    return walls


def _multi_room(
    rng: np.random.RandomState,
    cx: float, cy: float,
    n_rooms: int, room_size: float, doorway_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """Chain of rectangular rooms connected by doorways.

    Rooms are placed in a grid-like row along X with connecting doors.
    Creates a loop by connecting the last room back towards the first when
    n_rooms >= 3.
    """
    walls: List[ObstacleSpec] = []
    half = room_size / 2.0
    door_half = doorway_w / 2.0

    for i in range(n_rooms):
        rx = cx + i * room_size
        ry = cy

        # Top wall (with doorway if not first room)
        walls.append(_wall(rx, ry + half, room_size, thickness, height, "x", color))
        # Bottom wall
        walls.append(_wall(rx, ry - half, room_size, thickness, height, "x", color))

        # Left wall (with doorway for room-to-room connection)
        if i == 0:
            walls.append(_wall(rx - half, ry, room_size, thickness, height, "y", color))
        else:
            # Left wall with door hole — two segments above and below door
            seg_len = (room_size - doorway_w) / 2.0
            walls.append(_wall(rx - half, ry + door_half + seg_len / 2.0, seg_len, thickness, height, "y", color))
            walls.append(_wall(rx - half, ry - door_half - seg_len / 2.0, seg_len, thickness, height, "y", color))

    # Right wall of last room
    last_rx = cx + (n_rooms - 1) * room_size
    walls.append(_wall(last_rx + half, cy, room_size, thickness, height, "y", color))

    # Loop connection: top passage from last room back toward first (if >= 3 rooms)
    if n_rooms >= 3:
        loop_y = cy + half + room_size * 0.6
        loop_len = (n_rooms - 1) * room_size
        walls.append(_wall(cx + loop_len / 2.0, loop_y + doorway_w / 2.0, loop_len, thickness, height, "x", color))
        walls.append(_wall(cx + loop_len / 2.0, loop_y - doorway_w / 2.0, loop_len, thickness, height, "x", color))

    return walls


def _branching_corridors(
    rng: np.random.RandomState,
    cx: float, cy: float,
    main_len: float, n_branches: int, branch_len_range: Tuple[float, float],
    corridor_w: float, thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """A main corridor with perpendicular branches at random intervals."""
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # Main corridor along X
    walls.append(_wall(cx + main_len / 2.0, cy + hw, main_len, thickness, height, "x", color))
    walls.append(_wall(cx + main_len / 2.0, cy - hw, main_len, thickness, height, "x", color))

    # Branches
    branch_positions = sorted(rng.uniform(0.15, 0.85, size=n_branches) * main_len + cx)
    for bx in branch_positions:
        b_len = rng.uniform(branch_len_range[0], branch_len_range[1])
        side = rng.choice([-1.0, 1.0])
        by = cy + side * (hw + b_len / 2.0)
        walls.append(_wall(bx - hw, by, b_len, thickness, height, "y", color))
        walls.append(_wall(bx + hw, by, b_len, thickness, height, "y", color))
        # Cap the branch
        cap_y = cy + side * (hw + b_len)
        walls.append(_wall(bx, cap_y, corridor_w, thickness, height, "x", color))

    # Cap main corridor ends
    walls.append(_wall(cx, cy, corridor_w, thickness, height, "y", color))  # start
    walls.append(_wall(cx + main_len, cy, corridor_w, thickness, height, "y", color))  # end

    return walls


def _cul_de_sac(
    rng: np.random.RandomState,
    cx: float, cy: float,
    depth: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """A dead-end pocket the robot must back out of.

    Narrow corridor with a closed end — forces retreat/reverse behaviour.
    """
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    # Side walls along Y
    walls.append(_wall(cx - hw, cy + depth / 2.0, depth, thickness, height, "y", color))
    walls.append(_wall(cx + hw, cy + depth / 2.0, depth, thickness, height, "y", color))
    # Back wall
    walls.append(_wall(cx, cy + depth, corridor_w + thickness, thickness, height, "x", color))

    return walls


def _symmetric_corridor(
    rng: np.random.RandomState,
    cx: float, cy: float,
    length: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """A perfectly straight, symmetric corridor.

    Useful for testing forward locomotion and wall-following without turns.
    """
    hw = corridor_w / 2.0
    walls: List[ObstacleSpec] = []

    walls.append(_wall(cx + length / 2.0, cy + hw, length, thickness, height, "x", color))
    walls.append(_wall(cx + length / 2.0, cy - hw, length, thickness, height, "x", color))
    # End caps
    walls.append(_wall(cx, cy, corridor_w, thickness, height, "y", color))
    walls.append(_wall(cx + length, cy, corridor_w, thickness, height, "y", color))

    return walls


def _long_corridor(
    rng: np.random.RandomState,
    cx: float, cy: float,
    length: float, corridor_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """An extra-long corridor (>2m) for testing sustained forward motion."""
    return _symmetric_corridor(rng, cx, cy, length, corridor_w, thickness, height, color)


def _doorway(
    rng: np.random.RandomState,
    cx: float, cy: float,
    wall_len: float, gap_w: float,
    thickness: float, height: float,
    color: Tuple[float, float, float],
) -> List[ObstacleSpec]:
    """A wall with a narrow gap (doorway) the robot must pass through."""
    seg_len = (wall_len - gap_w) / 2.0
    if seg_len <= 0:
        seg_len = 0.1
    walls: List[ObstacleSpec] = []

    # Left segment
    walls.append(_wall(cx - gap_w / 2.0 - seg_len / 2.0, cy, seg_len, thickness, height, "x", color))
    # Right segment
    walls.append(_wall(cx + gap_w / 2.0 + seg_len / 2.0, cy, seg_len, thickness, height, "x", color))

    return walls


# --------------------------------------------------------------------------- #
# Maze style catalogue
# --------------------------------------------------------------------------- #

MAZE_STYLES = [
    "t_junction",
    "crossroads",
    "s_bend",
    "zigzag",
    "one_turn",
    "two_turn",
    "multi_room",
    "branching",
    "cul_de_sac",
    "symmetric_corridor",
    "long_corridor",
    "doorway_passage",
]


def generate_maze(
    style: Optional[str] = None,
    seed: int | None = None,
    arena_half: float = 3.0,
    wall_thickness: float = 0.06,
    wall_height_range: Tuple[float, float] = (0.15, 0.40),
    corridor_width_range: Tuple[float, float] = (0.30, 0.55),
    robot_clearance: float = 0.40,
    beacon_identities: Optional[List[str]] = None,
    n_beacons: int = 0,
) -> MazeResult:
    """Generate a maze layout of a given style.

    Args:
        style: one of :data:`MAZE_STYLES` or ``None`` (random).
        seed: RNG seed.
        arena_half: half-extent of the arena.
        wall_thickness: wall panel thickness.
        wall_height_range: (min, max) wall height.
        corridor_width_range: (min, max) corridor gap.
        robot_clearance: minimum XY clearance from origin.
        beacon_identities: beacon colour names to use (random subset if None).
        n_beacons: number of beacons to place in the maze.

    Returns:
        :class:`MazeResult` with walls and optional beacon placements.
    """
    rng = np.random.RandomState(seed)

    if style is None:
        style = rng.choice(MAZE_STYLES)

    height = float(rng.uniform(wall_height_range[0], wall_height_range[1]))
    cw = float(rng.uniform(corridor_width_range[0], corridor_width_range[1]))
    color = _random_color(rng)

    # Place maze centre offset from origin so the robot (at origin) enters it
    offset = rng.uniform(0.3, 1.0)
    angle = rng.uniform(0, 2 * math.pi)
    cx = offset * math.cos(angle)
    cy = offset * math.sin(angle)

    walls: List[ObstacleSpec] = []

    if style == "t_junction":
        arm = rng.uniform(0.6, 1.5)
        walls = _t_junction(rng, cx, cy, arm, cw, wall_thickness, height, color)

    elif style == "crossroads":
        arm = rng.uniform(0.5, 1.2)
        walls = _crossroads(rng, cx, cy, arm, cw, wall_thickness, height, color)

    elif style == "s_bend":
        seg = rng.uniform(0.5, 1.2)
        off = rng.uniform(0.4, 1.0) * rng.choice([-1.0, 1.0])
        walls = _s_bend(rng, cx, cy, seg, cw, off, wall_thickness, height, color)

    elif style == "zigzag":
        n_zigs = rng.randint(3, 6)
        seg = rng.uniform(0.4, 0.8)
        off = rng.uniform(0.3, 0.7)
        walls = _zigzag(rng, cx, cy, n_zigs, seg, cw, off, wall_thickness, height, color)

    elif style == "one_turn":
        l1 = rng.uniform(0.5, 1.5)
        l2 = rng.uniform(0.5, 1.5)
        walls = _one_turn(rng, cx, cy, l1, l2, cw, wall_thickness, height, color)

    elif style == "two_turn":
        legs = (rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.0))
        walls = _two_turn(rng, cx, cy, legs, cw, wall_thickness, height, color)

    elif style == "multi_room":
        n_rooms = rng.randint(2, 5)
        room_sz = rng.uniform(0.6, 1.2)
        door_w = rng.uniform(0.25, 0.45)
        walls = _multi_room(rng, cx, cy, n_rooms, room_sz, door_w, wall_thickness, height, color)

    elif style == "branching":
        main_l = rng.uniform(1.5, 3.0)
        n_br = rng.randint(2, 5)
        walls = _branching_corridors(rng, cx, cy, main_l, n_br, (0.4, 1.0),
                                     cw, wall_thickness, height, color)

    elif style == "cul_de_sac":
        depth = rng.uniform(0.6, 1.5)
        walls = _cul_de_sac(rng, cx, cy, depth, cw, wall_thickness, height, color)

    elif style == "symmetric_corridor":
        length = rng.uniform(1.0, 2.0)
        walls = _symmetric_corridor(rng, cx, cy, length, cw, wall_thickness, height, color)

    elif style == "long_corridor":
        length = rng.uniform(2.0, 4.0)
        walls = _long_corridor(rng, cx, cy, length, cw, wall_thickness, height, color)

    elif style == "doorway_passage":
        wall_len = rng.uniform(1.5, 3.0)
        gap = rng.uniform(0.25, 0.45)
        walls = _doorway(rng, cx, cy, wall_len, gap, wall_thickness, height, color)

    # Filter walls that would block the spawn
    walls = [w for w in walls if _clears_origin(w, robot_clearance)]

    # --- Optional beacon placement on maze walls ---
    beacons: List[BeaconSpec] = []
    if n_beacons > 0 and walls:
        from .beacon_utils import BEACON_FAMILIES
        if beacon_identities is None:
            all_ids = list(BEACON_FAMILIES.keys())
            beacon_identities = list(rng.choice(all_ids, size=min(n_beacons, len(all_ids)), replace=False))

        # Pick random walls to host beacons
        host_indices = rng.choice(len(walls), size=min(n_beacons, len(walls)), replace=False)
        for i, wi in enumerate(host_indices):
            w = walls[wi]
            identity = beacon_identities[i % len(beacon_identities)]
            normal = _choose_accessible_wall_face(
                obs=w,
                obstacles=walls,
                robot_clearance=float(robot_clearance),
                arena_half=float(arena_half),
            )
            b = make_beacon_panel(w.pos, normal, identity, rng)
            beacons.append(b)

    return MazeResult(layout=ObstacleLayout(walls), beacons=beacons)


def generate_random_maze(
    seed: int | None = None,
    n_beacons: int = 0,
    **kwargs,
) -> MazeResult:
    """Convenience: pick a random maze style and generate it."""
    return generate_maze(style=None, seed=seed, n_beacons=n_beacons, **kwargs)


# --------------------------------------------------------------------------- #
# Composite: maze + free obstacles + beacons
# --------------------------------------------------------------------------- #

def generate_composite_scene(
    seed: int | None = None,
    maze_style: Optional[str] = None,
    n_free_obstacles: int = 0,
    n_beacons: int = 0,
    n_distractors: int = 0,
    arena_half: float = 3.0,
    perimeter_prob: float = 0.4,
) -> Tuple[ObstacleLayout, BeaconLayout]:
    """Generate a full scene: maze walls + free obstacles + beacons + distractors.

    Returns:
        (obstacle_layout, beacon_layout) — the obstacle layout includes maze
        walls and free obstacles; the beacon layout includes true beacons and
        distractor patches.
    """
    from .obstacle_utils import generate_random_layout, _generate_perimeter
    from .beacon_utils import generate_beacon_layout, make_distractor_patch, BEACON_FAMILIES

    rng = np.random.RandomState(seed)

    # 1) Maze
    maze = generate_maze(style=maze_style, seed=int(rng.randint(0, 2**31)),
                         n_beacons=n_beacons, arena_half=arena_half)

    all_obstacles = list(maze.layout.obstacles)

    # 2) Free obstacles (sprinkled in remaining space)
    if n_free_obstacles > 0:
        free_layout = generate_random_layout(
            n_range=(n_free_obstacles, n_free_obstacles + 1),
            seed=int(rng.randint(0, 2**31)),
        )
        all_obstacles.extend(free_layout.obstacles)

    # 3) Optional perimeter
    if rng.rand() < perimeter_prob:
        perimeter_h = rng.uniform(0.15, 0.35)
        all_obstacles.extend(_generate_perimeter(rng, arena_half, perimeter_h))

    obstacle_layout = ObstacleLayout(all_obstacles)

    # 4) Beacon layout (beacons from maze + distractors)
    from .beacon_utils import BeaconLayout as BL
    distractors = []
    if n_distractors > 0:
        identities = list(BEACON_FAMILIES.keys())
        for _ in range(n_distractors):
            pos = (float(rng.uniform(-arena_half + 0.3, arena_half - 0.3)),
                   float(rng.uniform(-arena_half + 0.3, arena_half - 0.3)),
                   float(rng.uniform(0.10, 0.30)))
            near = rng.choice(identities)
            distractors.append(make_distractor_patch(pos, rng, near_identity=near))

    beacon_layout = BL(beacons=maze.beacons, distractors=distractors)

    return obstacle_layout, beacon_layout
