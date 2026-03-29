"""Random obstacle layout generation and collision detection.

Supports three obstacle primitives:
  - **boxes**: free-standing blocks (original v1 behaviour)
  - **walls**: long thin barriers that constrain movement
  - **perimeter**: arena boundary walls that keep the robot in-bounds

Layouts are composed by stochastically mixing these primitives so the JEPA
learns to handle corridors, dead-ends, and open-field clutter alike.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ObstacleSpec:
    """Axis-aligned box obstacle."""
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.55, 0.55, 0.60)


@dataclass
class ObstacleLayout:
    """A full obstacle configuration for one chunk / scene."""
    obstacles: List[ObstacleSpec] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps([asdict(o) for o in self.obstacles])

    @staticmethod
    def from_json(s: str) -> "ObstacleLayout":
        data = json.loads(s)
        return ObstacleLayout([ObstacleSpec(**d) for d in data])


def _random_color(rng: np.random.RandomState) -> Tuple[float, float, float]:
    """Sample a muted obstacle color."""
    base = rng.uniform(0.3, 0.7)
    tint = rng.uniform(-0.1, 0.1, size=3)
    c = np.clip(base + tint, 0.1, 0.9)
    return (float(c[0]), float(c[1]), float(c[2]))


# Keep the old public name working (used by render_worker colour randomization).
def random_obstacle_color() -> Tuple[float, float, float]:
    base = np.random.uniform(0.3, 0.7)
    tint = np.random.uniform(-0.1, 0.1, size=3)
    c = np.clip(base + tint, 0.1, 0.9)
    return (float(c[0]), float(c[1]), float(c[2]))


def _clears_origin(obs: ObstacleSpec, clearance: float) -> bool:
    """Return True when an obstacle's XY AABB stays clear of the spawn zone."""
    cx, cy = obs.pos[0], obs.pos[1]
    hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
    dx = max(abs(cx) - hx, 0.0)
    dy = max(abs(cy) - hy, 0.0)
    return (dx * dx + dy * dy) >= clearance * clearance


def _all_clear_of_origin(obstacles: List[ObstacleSpec], clearance: float) -> bool:
    return all(_clears_origin(obs, clearance) for obs in obstacles)


# --------------------------------------------------------------------------- #
# Primitive generators
# --------------------------------------------------------------------------- #

def _generate_boxes(
    rng: np.random.RandomState,
    n: int,
    size_range: Tuple[float, float],
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """Free-standing box obstacles placed in a ring around the origin."""
    boxes: List[ObstacleSpec] = []
    attempts = 0
    max_attempts = max(8 * n, 8)
    while len(boxes) < n and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(spawn_radius[0], spawn_radius[1])
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        sx = rng.uniform(size_range[0], size_range[1])
        sy = rng.uniform(size_range[0], size_range[1])
        sz = rng.uniform(height_range[0], height_range[1])
        obstacle = ObstacleSpec(
            pos=(float(x), float(y), float(sz / 2.0)),
            size=(float(sx), float(sy), float(sz)),
            color=_random_color(rng),
        )
        if _clears_origin(obstacle, robot_clearance):
            boxes.append(obstacle)
    return boxes


def _generate_walls(
    rng: np.random.RandomState,
    n: int,
    length_range: Tuple[float, float],
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """Long thin wall segments (axis-aligned)."""
    walls: List[ObstacleSpec] = []
    attempts = 0
    max_attempts = max(8 * n, 8)
    while len(walls) < n and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(spawn_radius[0], spawn_radius[1])
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)

        length = rng.uniform(length_range[0], length_range[1])
        sz = rng.uniform(height_range[0], height_range[1])

        # Randomly orient along X or Y axis.
        if rng.rand() < 0.5:
            sx, sy = length, thickness
        else:
            sx, sy = thickness, length

        obstacle = ObstacleSpec(
            pos=(float(x), float(y), float(sz / 2.0)),
            size=(float(sx), float(sy), float(sz)),
            color=_random_color(rng),
        )
        if _clears_origin(obstacle, robot_clearance):
            walls.append(obstacle)
    return walls


def _generate_corridor(
    rng: np.random.RandomState,
    length_range: Tuple[float, float],
    width_range: Tuple[float, float],
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """A pair of parallel walls forming a corridor the robot must pass through."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    cx = dist * math.cos(angle)
    cy = dist * math.sin(angle)

    length = rng.uniform(length_range[0], length_range[1])
    gap = rng.uniform(width_range[0], width_range[1])
    sz = rng.uniform(height_range[0], height_range[1])

    # Corridor orientation: random angle
    orient = rng.uniform(0, math.pi)
    cos_o, sin_o = math.cos(orient), math.sin(orient)

    # Perpendicular offset to place the two walls on either side
    perp_x, perp_y = -sin_o, cos_o
    half_gap = gap / 2.0

    walls = []
    for sign in (-1.0, 1.0):
        wx = cx + sign * half_gap * perp_x
        wy = cy + sign * half_gap * perp_y

        # Axis-aligned bounding box that approximates the rotated wall.
        # For simplicity we use axis-aligned boxes; the corridor effect comes
        # from the pair placement rather than exact rotation.
        if abs(cos_o) > abs(sin_o):
            sx, sy = length, thickness
        else:
            sx, sy = thickness, length

        walls.append(ObstacleSpec(
            pos=(float(wx), float(wy), float(sz / 2.0)),
            size=(float(sx), float(sy), float(sz)),
            color=_random_color(rng),
        ))

    return walls if _all_clear_of_origin(walls, robot_clearance) else []


def _generate_l_shape(
    rng: np.random.RandomState,
    arm_length_range: Tuple[float, float],
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """An L-shaped wall: two perpendicular wall segments meeting at a corner."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    corner_x = dist * math.cos(angle)
    corner_y = dist * math.sin(angle)

    arm1_len = rng.uniform(arm_length_range[0], arm_length_range[1])
    arm2_len = rng.uniform(arm_length_range[0], arm_length_range[1])
    sz = rng.uniform(height_range[0], height_range[1])
    color = _random_color(rng)

    # Pick a random rotation quadrant for the L shape
    flip_x = rng.choice([-1.0, 1.0])
    flip_y = rng.choice([-1.0, 1.0])

    pieces: List[ObstacleSpec] = []

    # Horizontal arm (extends along X from corner)
    pieces.append(ObstacleSpec(
        pos=(float(corner_x + flip_x * arm1_len / 2.0), float(corner_y), float(sz / 2.0)),
        size=(float(arm1_len), float(thickness), float(sz)),
        color=color,
    ))

    # Vertical arm (extends along Y from corner)
    pieces.append(ObstacleSpec(
        pos=(float(corner_x), float(corner_y + flip_y * arm2_len / 2.0), float(sz / 2.0)),
        size=(float(thickness), float(arm2_len), float(sz)),
        color=color,
    ))

    return pieces if _all_clear_of_origin(pieces, robot_clearance) else []


def _generate_dead_end(
    rng: np.random.RandomState,
    width_range: Tuple[float, float],
    depth_range: Tuple[float, float],
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """A U-shaped dead end: three walls forming a pocket."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    cx = dist * math.cos(angle)
    cy = dist * math.sin(angle)

    width = rng.uniform(width_range[0], width_range[1])
    depth = rng.uniform(depth_range[0], depth_range[1])
    sz = rng.uniform(height_range[0], height_range[1])
    color = _random_color(rng)

    # Random orientation: open along +X, -X, +Y, or -Y
    orient = rng.randint(0, 4)
    hw, hd = width / 2.0, depth / 2.0

    pieces: List[ObstacleSpec] = []

    if orient == 0:  # open +X
        # Back wall
        pieces.append(ObstacleSpec(
            pos=(float(cx - hd), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(width), float(sz)), color=color))
        # Side walls
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy + hw), float(sz / 2.0)),
            size=(float(depth), float(thickness), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy - hw), float(sz / 2.0)),
            size=(float(depth), float(thickness), float(sz)), color=color))
    elif orient == 1:  # open -X
        pieces.append(ObstacleSpec(
            pos=(float(cx + hd), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(width), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy + hw), float(sz / 2.0)),
            size=(float(depth), float(thickness), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy - hw), float(sz / 2.0)),
            size=(float(depth), float(thickness), float(sz)), color=color))
    elif orient == 2:  # open +Y
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy - hd), float(sz / 2.0)),
            size=(float(width), float(thickness), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx + hw), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(depth), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx - hw), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(depth), float(sz)), color=color))
    else:  # open -Y
        pieces.append(ObstacleSpec(
            pos=(float(cx), float(cy + hd), float(sz / 2.0)),
            size=(float(width), float(thickness), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx + hw), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(depth), float(sz)), color=color))
        pieces.append(ObstacleSpec(
            pos=(float(cx - hw), float(cy), float(sz / 2.0)),
            size=(float(thickness), float(depth), float(sz)), color=color))

    return pieces if _all_clear_of_origin(pieces, robot_clearance) else []


def _generate_doorway(
    rng: np.random.RandomState,
    wall_length: float,
    gap_width: float,
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """A wall with a narrow gap (doorway) the robot must squeeze through."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    cx = dist * math.cos(angle)
    cy = dist * math.sin(angle)
    sz = rng.uniform(height_range[0], height_range[1])
    color = _random_color(rng)

    seg_len = max((wall_length - gap_width) / 2.0, 0.1)
    orient = rng.choice(["x", "y"])

    walls: List[ObstacleSpec] = []
    if orient == "x":
        # Wall runs along X with gap in the middle
        walls.append(ObstacleSpec(
            pos=(float(cx - gap_width / 2.0 - seg_len / 2.0), float(cy), float(sz / 2.0)),
            size=(float(seg_len), float(thickness), float(sz)), color=color))
        walls.append(ObstacleSpec(
            pos=(float(cx + gap_width / 2.0 + seg_len / 2.0), float(cy), float(sz / 2.0)),
            size=(float(seg_len), float(thickness), float(sz)), color=color))
    else:
        walls.append(ObstacleSpec(
            pos=(float(cx), float(cy - gap_width / 2.0 - seg_len / 2.0), float(sz / 2.0)),
            size=(float(thickness), float(seg_len), float(sz)), color=color))
        walls.append(ObstacleSpec(
            pos=(float(cx), float(cy + gap_width / 2.0 + seg_len / 2.0), float(sz / 2.0)),
            size=(float(thickness), float(seg_len), float(sz)), color=color))

    return walls if _all_clear_of_origin(walls, robot_clearance) else []


def _generate_long_wall(
    rng: np.random.RandomState,
    length_range: Tuple[float, float],
    thickness: float,
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """A single very long wall (>2 m) that forces large detours."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    cx = dist * math.cos(angle)
    cy = dist * math.sin(angle)
    length = rng.uniform(length_range[0], length_range[1])
    sz = rng.uniform(height_range[0], height_range[1])
    color = _random_color(rng)

    if rng.rand() < 0.5:
        sx, sy = length, thickness
    else:
        sx, sy = thickness, length

    wall = ObstacleSpec(
        pos=(float(cx), float(cy), float(sz / 2.0)),
        size=(float(sx), float(sy), float(sz)),
        color=color,
    )
    return [wall] if _clears_origin(wall, robot_clearance) else []


def _generate_slalom(
    rng: np.random.RandomState,
    n_boxes: int,
    spacing_range: Tuple[float, float],
    size_range: Tuple[float, float],
    stagger_range: Tuple[float, float],
    height_range: Tuple[float, float],
    spawn_radius: Tuple[float, float],
    robot_clearance: float,
) -> List[ObstacleSpec]:
    """Staggered boxes forming a slalom the robot must weave through."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(spawn_radius[0], spawn_radius[1])
    start_x = dist * math.cos(angle)
    start_y = dist * math.sin(angle)

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    boxes: List[ObstacleSpec] = []
    side = 1.0

    for i in range(n_boxes):
        spacing = rng.uniform(spacing_range[0], spacing_range[1])
        fwd = spacing * (i + 1)
        stagger = side * rng.uniform(stagger_range[0], stagger_range[1])
        bx = start_x + fwd * cos_a - stagger * sin_a
        by = start_y + fwd * sin_a + stagger * cos_a
        sx = rng.uniform(size_range[0], size_range[1])
        sy = rng.uniform(size_range[0], size_range[1])
        sz = rng.uniform(height_range[0], height_range[1])
        obs = ObstacleSpec(
            pos=(float(bx), float(by), float(sz / 2.0)),
            size=(float(sx), float(sy), float(sz)),
            color=_random_color(rng),
        )
        if _clears_origin(obs, robot_clearance):
            boxes.append(obs)
        side *= -1.0

    return boxes


def _generate_perimeter(
    rng: np.random.RandomState,
    arena_half: float,
    wall_height: float,
    thickness: float = 0.20,
) -> List[ObstacleSpec]:
    """Four walls forming a square arena perimeter."""
    color = _random_color(rng)
    side = arena_half * 2.0
    hz = wall_height / 2.0
    walls = [
        # +Y wall
        ObstacleSpec(pos=(0.0, float(arena_half), float(hz)),
                     size=(float(side + thickness), float(thickness), float(wall_height)),
                     color=color),
        # -Y wall
        ObstacleSpec(pos=(0.0, float(-arena_half), float(hz)),
                     size=(float(side + thickness), float(thickness), float(wall_height)),
                     color=color),
        # +X wall
        ObstacleSpec(pos=(float(arena_half), 0.0, float(hz)),
                     size=(float(thickness), float(side + thickness), float(wall_height)),
                     color=color),
        # -X wall
        ObstacleSpec(pos=(float(-arena_half), 0.0, float(hz)),
                     size=(float(thickness), float(side + thickness), float(wall_height)),
                     color=color),
    ]
    return walls


# --------------------------------------------------------------------------- #
# Layout composition
# --------------------------------------------------------------------------- #

# Layout style weights — each chunk randomly picks one.
_LAYOUT_STYLES = [
    "mixed",        # boxes + walls (balanced)
    "corridor",     # corridors + scattered boxes
    "cluttered",    # many boxes, few walls
    "structured",   # L-shapes + dead-ends + corridors
    "open",         # few large obstacles (mostly walls)
    "doorway",      # walls with narrow gaps
    "slalom",       # staggered boxes to weave through
    "long_walls",   # very long walls (>2m) forcing detours
]


def generate_random_layout(
    n_range: Tuple[int, int] = (3, 8),
    size_range: Tuple[float, float] = (0.15, 0.40),
    height_range: Tuple[float, float] = (0.10, 0.50),
    spawn_radius: Tuple[float, float] = (0.5, 2.5),
    robot_clearance: float = 0.40,
    seed: int | None = None,
    wall_thickness: float = 0.20,
    perimeter_prob: float = 0.4,
    arena_half: float = 3.0,
) -> ObstacleLayout:
    """Generate a random obstacle layout mixing boxes, walls, and structures.

    Each call randomly selects a layout style and populates the scene with an
    appropriate mix of obstacle primitives.  ~40% of layouts also get a
    perimeter wall to teach the robot about arena boundaries.
    """
    rng = np.random.RandomState(seed)

    style = rng.choice(_LAYOUT_STYLES)
    obstacles: List[ObstacleSpec] = []

    common = dict(
        height_range=height_range,
        spawn_radius=spawn_radius,
        robot_clearance=robot_clearance,
    )

    if style == "mixed":
        n_boxes = rng.randint(2, 5)
        n_walls = rng.randint(1, 4)
        obstacles += _generate_boxes(rng, n_boxes, size_range, **common)
        obstacles += _generate_walls(rng, n_walls,
                                     length_range=(0.5, 1.5),
                                     thickness=wall_thickness, **common)
        if rng.rand() < 0.4:
            obstacles += _generate_corridor(rng,
                                            length_range=(0.8, 1.5),
                                            width_range=(0.50, 0.70),
                                            thickness=wall_thickness, **common)

    elif style == "corridor":
        n_corridors = rng.randint(1, 3)
        for _ in range(n_corridors):
            obstacles += _generate_corridor(rng,
                                            length_range=(0.8, 2.0),
                                            width_range=(0.50, 0.70),
                                            thickness=wall_thickness, **common)
        obstacles += _generate_boxes(rng, rng.randint(1, 4), size_range, **common)

    elif style == "cluttered":
        n_boxes = rng.randint(5, 10)
        n_walls = rng.randint(0, 2)
        obstacles += _generate_boxes(rng, n_boxes, size_range, **common)
        obstacles += _generate_walls(rng, n_walls,
                                     length_range=(0.4, 1.0),
                                     thickness=wall_thickness, **common)

    elif style == "structured":
        if rng.rand() < 0.5:
            obstacles += _generate_l_shape(rng,
                                           arm_length_range=(0.5, 1.2),
                                           thickness=wall_thickness, **common)
        if rng.rand() < 0.5:
            obstacles += _generate_dead_end(rng,
                                            width_range=(0.50, 0.70),
                                            depth_range=(0.4, 0.9),
                                            thickness=wall_thickness, **common)
        obstacles += _generate_corridor(rng,
                                        length_range=(0.6, 1.5),
                                        width_range=(0.50, 0.70),
                                        thickness=wall_thickness, **common)
        obstacles += _generate_boxes(rng, rng.randint(1, 3), size_range, **common)

    elif style == "open":
        n_walls = rng.randint(2, 5)
        obstacles += _generate_walls(rng, n_walls,
                                     length_range=(0.8, 2.0),
                                     thickness=wall_thickness, **common)
        obstacles += _generate_boxes(rng, rng.randint(0, 2), size_range, **common)

    elif style == "doorway":
        n_doorways = rng.randint(1, 4)
        for _ in range(n_doorways):
            obstacles += _generate_doorway(rng,
                                           wall_length=rng.uniform(1.0, 2.5),
                                           gap_width=rng.uniform(0.45, 0.65),
                                           thickness=wall_thickness, **common)
        obstacles += _generate_boxes(rng, rng.randint(1, 3), size_range, **common)

    elif style == "slalom":
        n_slalom = rng.randint(4, 8)
        obstacles += _generate_slalom(rng, n_slalom,
                                      spacing_range=(0.35, 0.60),
                                      size_range=size_range,
                                      stagger_range=(0.20, 0.45), **common)

    elif style == "long_walls":
        n_lw = rng.randint(1, 4)
        for _ in range(n_lw):
            obstacles += _generate_long_wall(rng,
                                             length_range=(2.0, 4.0),
                                             thickness=wall_thickness, **common)
        obstacles += _generate_boxes(rng, rng.randint(0, 3), size_range, **common)

    # Optionally add a perimeter wall to bound the arena.
    if rng.rand() < perimeter_prob:
        perimeter_height = rng.uniform(0.15, 0.35)
        obstacles += _generate_perimeter(rng, arena_half, perimeter_height)

    return ObstacleLayout(obstacles)


def add_obstacles_to_scene(scene, layout: ObstacleLayout) -> None:
    """Add all obstacles from a layout to a Genesis scene (before build)."""
    import genesis as gs  # noqa: delay import
    for obs in layout.obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )


def detect_collisions(
    robot_pos_xy: torch.Tensor,
    layout: ObstacleLayout,
    margin: float = 0.15,
) -> torch.Tensor:
    """Per-env boolean: is the robot within margin of any obstacle AABB?

    Args:
        robot_pos_xy: (N, 2) tensor of robot XY positions.
        layout: obstacle layout for this scene.
        margin: clearance buffer (metres).

    Returns:
        (N,) bool tensor — True if the robot is clipping / too close.
    """
    N = robot_pos_xy.shape[0]
    colliding = torch.zeros(N, dtype=torch.bool, device=robot_pos_xy.device)

    for obs in layout.obstacles:
        cx, cy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0 + margin, obs.size[1] / 2.0 + margin
        in_x = (robot_pos_xy[:, 0] > cx - hx) & (robot_pos_xy[:, 0] < cx + hx)
        in_y = (robot_pos_xy[:, 1] > cy - hy) & (robot_pos_xy[:, 1] < cy + hy)
        colliding |= in_x & in_y

    return colliding
