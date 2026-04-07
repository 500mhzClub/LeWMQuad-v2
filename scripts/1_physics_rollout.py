#!/usr/bin/env python3
"""Collect trajectory data from a frozen PPO walking policy in Genesis.

v2: Walls are 0.20m thick (up from 0.06m) to prevent camera clipping.
Corridors are wider (0.50-0.70m) to remain navigable with thicker walls.

Extended from TinyQuadJEPA-v2 with:
  - Maze topology integration (T-junctions, crossroads, S-bends, etc.)
  - Beacon panel placement and metadata recording
  - Diverse command patterns (retreat, recovery, dead-end backout, wall-follow)
  - Clearance / near-miss / traversability labels
  - Soft collision handling (don't always terminate — allow recovery)
  - Composite scene generation (maze + free obstacles + beacons + distractors)

Output: one .npz per chunk in --out_dir, each containing:
    proprio       (n_envs, steps, 47)   noisy proprioceptive observation
    cmds          (n_envs, steps, 3)    velocity commands (vx, vy, wz)
    dones         (n_envs, steps)       episode termination flags
    base_pos      (n_envs, steps, 3)    world-frame base position
    base_quat     (n_envs, steps, 4)    world-frame base orientation (wxyz)
    joint_pos     (n_envs, steps, 12)   actuated joint positions
    collisions    (n_envs, steps)       per-step collision flags
    clearance     (n_envs, steps)       min distance to nearest obstacle
    near_miss     (n_envs, steps)       close-but-not-colliding flag
    traversability(n_envs, steps)       forward look-ahead steps clear
    beacon_visible(n_envs, steps)       any beacon in camera FOV
    beacon_identity(n_envs, steps)      closest visible beacon index (-1 = none)
    beacon_bearing(n_envs, steps)       relative bearing to closest beacon (rad)
    beacon_range  (n_envs, steps)       distance to closest beacon (m)
    obstacle_layout                     JSON of obstacle layout
    beacon_layout                       JSON of beacon layout
    cmd_pattern   (n_envs, steps)       command pattern name index
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import genesis as gs
from tqdm import tqdm

from lewm.models.ppo import ActorCritic
from lewm.math_utils import world_to_body_vec, yaw_to_quat
from lewm.genesis_utils import init_genesis_once, to_genesis_target
from lewm.checkpoint_utils import load_ppo_checkpoint
from lewm.obstacle_utils import (
    generate_random_layout,
    add_obstacles_to_scene,
    detect_collisions,
    ObstacleLayout,
)
from lewm.maze_utils import (
    generate_maze,
    generate_composite_scene,
    generate_enclosed_maze,
    MAZE_STYLES,
)
from lewm.beacon_utils import BeaconLayout
from lewm.command_utils import (
    OUProcess,
    sample_command_pattern,
    build_mixed_command_sequence,
    COMMAND_PATTERNS,
)
from lewm.label_utils import compute_episode_labels

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

JOINTS_ACTUATED = [
    "lf_hip_joint",  "lh_hip_joint",  "rf_hip_joint",  "rh_hip_joint",
    "lf_thigh_joint","lh_thigh_joint","rf_thigh_joint","rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

Q0_VALUES = [
    0.06,  0.06, -0.06, -0.06,   # hips (LF, LH, RF, RH)
    0.85,  0.85,  0.85,  0.85,   # thighs
   -1.75, -1.75, -1.75, -1.75,   # calves
]

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
ROBOT_SPAWN = (0.0, 0.0, 0.12)
DEFAULT_SCENE_BATCH_ENVS = 32768

# Command pattern name → integer index for storage
EXTRA_CMD_PATTERN_NAMES = [
    "maze_teacher_beacon",
    "maze_teacher_frontier",
    "maze_teacher_explore",
]
CMD_PATTERN_NAMES = list(COMMAND_PATTERNS.keys()) + EXTRA_CMD_PATTERN_NAMES
CMD_PATTERN_INDEX = {name: i for i, name in enumerate(CMD_PATTERN_NAMES)}

# --------------------------------------------------------------------------- #
# Simulation config
# --------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    n_envs: int = 2048
    dt: float = 0.01
    substeps: int = 4
    decimation: int = 4
    kp: float = 5.0
    kv: float = 0.5
    action_scale: float = 0.30
    # Mini Pupper regularly dips just below 5 cm during nominal gait, so a
    # 4 cm threshold is a less trigger-happy "fallen" heuristic.
    min_z: float = 0.04
    max_tilt: float = 1.0
    collision_margin: float = 0.15
    safe_clearance: float = 0.40
    # New: soft collision mode — don't always terminate on collision
    soft_collision_prob: float = 0.3   # probability of NOT terminating on collision
    near_miss_threshold: float = 0.20

# --------------------------------------------------------------------------- #
# OU noise (GPU batched, matches original)
# --------------------------------------------------------------------------- #

class OUNoiseBatched:
    """Batched Ornstein-Uhlenbeck process for correlated command exploration."""

    def __init__(
        self,
        n_envs: int,
        dim: int,
        device: str,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.n_envs = n_envs
        self.dim = dim
        self.device = device
        self.theta = theta
        self.sigma = sigma
        self.state = torch.zeros((n_envs, dim), device=device)

    def step(self) -> torch.Tensor:
        noise = torch.randn((self.n_envs, self.dim), device=self.device)
        self.state = self.state - self.theta * self.state + self.sigma * noise
        return self.state

    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = 0.0


# --------------------------------------------------------------------------- #
# Safe respawn
# --------------------------------------------------------------------------- #

def sample_safe_positions(
    n: int,
    layout: ObstacleLayout,
    clearance: float,
    spawn_range: float = 2.0,
    bounds_xy: tuple[float, float, float, float] | None = None,
    max_attempts: int = 200,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample n (x, y) positions that are >= clearance from every obstacle."""
    positions = torch.zeros((n, 2), device=device)
    filled = 0
    attempts = 0

    while filled < n and attempts < max_attempts * n:
        batch_size = min(n - filled, 256)
        if bounds_xy is not None:
            x_min, x_max, y_min, y_max = [float(v) for v in bounds_xy]
            candidates = torch.empty((batch_size, 2), device=device)
            candidates[:, 0] = torch.rand(batch_size, device=device) * (x_max - x_min) + x_min
            candidates[:, 1] = torch.rand(batch_size, device=device) * (y_max - y_min) + y_min
        else:
            candidates = (torch.rand((batch_size, 2), device=device) * 2 - 1) * spawn_range
        colliding = detect_collisions(candidates, layout, margin=clearance)
        safe_mask = ~colliding
        safe_pts = candidates[safe_mask]
        take = min(safe_pts.shape[0], n - filled)
        if take > 0:
            positions[filled : filled + take] = safe_pts[:take]
            filled += take
        attempts += batch_size

    if filled < n:
        print(f"  [WARN] Could only find {filled}/{n} safe spawn points; "
              f"placing remainder at origin.")

    return positions


def sample_episode_spawn_positions(
    n: int,
    layout: ObstacleLayout,
    scene_meta: dict,
    clearance: float,
    strategy: str,
    start_jitter_radius: float,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample respawn positions aligned with the scene topology.

    ``uniform_safe`` reproduces the old behavior but can optionally be bounded
    to the scene interior. ``scene_start`` and ``scene_start_jitter`` keep
    episodes near the designated maze start region, which better matches the
    enclosed-maze inference task.
    """
    scene_type = str(scene_meta.get("scene_type", "unknown"))
    if strategy == "auto":
        strategy = "scene_start_jitter" if scene_type == "enclosed" else "uniform_safe"

    bounds_xy_raw = scene_meta.get("spawn_bounds_xy")
    bounds_xy = None
    if bounds_xy_raw is not None and len(bounds_xy_raw) == 4:
        bounds_xy = tuple(float(v) for v in bounds_xy_raw)

    if strategy == "uniform_safe":
        spawn_range = float(scene_meta.get("spawn_range", 2.0))
        return sample_safe_positions(
            n,
            layout,
            clearance=clearance,
            spawn_range=spawn_range,
            bounds_xy=bounds_xy,
            device=device,
        )

    start_xy_raw = scene_meta.get("start_xy")
    if start_xy_raw is None:
        raise ValueError(
            f"Respawn strategy {strategy!r} requires scene_meta['start_xy'], "
            f"but scene_type={scene_type!r} does not provide it."
        )
    start_xy = torch.tensor(start_xy_raw, device=device, dtype=torch.float32).view(1, 2)

    if strategy == "scene_start":
        colliding = detect_collisions(start_xy, layout, margin=clearance)
        if bool(colliding[0].item()):
            raise ValueError(
                f"Configured start_xy {start_xy_raw} is not safe under clearance={clearance:.3f}",
            )
        return start_xy.expand(n, -1).clone()

    if strategy == "scene_start_jitter":
        positions = torch.zeros((n, 2), device=device, dtype=torch.float32)
        filled = 0
        attempts = 0
        radius = max(0.0, float(start_jitter_radius))
        while filled < n and attempts < 400 * max(1, n):
            batch_size = min(n - filled, 256)
            jitter = (torch.rand((batch_size, 2), device=device) * 2.0 - 1.0) * radius
            candidates = start_xy + jitter
            if bounds_xy is not None:
                x_min, x_max, y_min, y_max = bounds_xy
                candidates[:, 0].clamp_(x_min, x_max)
                candidates[:, 1].clamp_(y_min, y_max)
            colliding = detect_collisions(candidates, layout, margin=clearance)
            safe_pts = candidates[~colliding]
            take = min(int(safe_pts.shape[0]), n - filled)
            if take > 0:
                positions[filled:filled + take] = safe_pts[:take]
                filled += take
            attempts += batch_size
        if filled < n:
            fallback = sample_safe_positions(
                n - filled,
                layout,
                clearance=clearance,
                spawn_range=float(scene_meta.get("spawn_range", 2.0)),
                bounds_xy=bounds_xy,
                device=device,
            )
            positions[filled:] = fallback
        return positions

    raise ValueError(
        "respawn strategy must be one of: auto, uniform_safe, scene_start, scene_start_jitter"
    )


def load_frozen_policy(ckpt_path: str) -> ActorCritic:
    """Load the PPO actor-critic onto the current Genesis torch device."""
    model = ActorCritic(obs_dim=50, act_dim=12).to(gs.device)
    ppo_sd = load_ppo_checkpoint(ckpt_path, device=gs.device)
    model.load_state_dict(ppo_sd, strict=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Scene generation
# --------------------------------------------------------------------------- #

def generate_scene(
    seed: int,
    scene_distribution: str = "mixed",
    use_maze: bool = True,
    n_beacons: int = 2,
    n_distractors: int = 1,
    n_free_obstacles: int = 2,
    arena_half: float = 3.0,
    enclosed_grid_rows_min: int = 4,
    enclosed_grid_rows_max: int = 7,
    enclosed_grid_cols_min: int = 4,
    enclosed_grid_cols_max: int = 7,
    enclosed_cell_size_min: float = 0.50,
    enclosed_cell_size_max: float = 0.70,
    enclosed_wall_thickness_min: float = 0.18,
    enclosed_wall_thickness_max: float = 0.24,
) -> tuple:
    """Generate a scene (obstacles + beacons) for one chunk.

    Returns:
        (obstacle_layout, beacon_layout, scene_meta)
    """
    rng = np.random.RandomState(seed)
    scene_distribution = str(scene_distribution).strip().lower()

    if scene_distribution == "legacy":
        scene_kind = "composite" if (use_maze and rng.rand() < 0.7) else "free"
    elif scene_distribution == "mixed":
        if use_maze:
            scene_kind = str(
                rng.choice(
                    ["enclosed", "composite", "free"],
                    p=[0.45, 0.40, 0.15],
                )
            )
        else:
            scene_kind = "free"
    elif scene_distribution in {"composite", "enclosed", "free"}:
        scene_kind = scene_distribution
    else:
        raise ValueError(
            "scene_distribution must be one of: legacy, mixed, composite, enclosed, free"
        )

    if scene_kind == "composite":
        maze_style = rng.choice(MAZE_STYLES)
        obstacle_layout, beacon_layout = generate_composite_scene(
            seed=seed,
            maze_style=maze_style,
            n_free_obstacles=n_free_obstacles,
            n_beacons=n_beacons,
            n_distractors=n_distractors,
            arena_half=arena_half,
        )
        return obstacle_layout, beacon_layout, {
            "scene_seed": int(seed),
            "scene_distribution": scene_distribution,
            "scene_type": "composite",
            "scene_label": str(maze_style),
            "n_obstacles": int(len(obstacle_layout.obstacles)),
            "n_beacons": int(len(beacon_layout.beacons)),
            "n_distractors": int(len(beacon_layout.distractors)),
        }

    if scene_kind == "enclosed":
        grid_rows = int(
            rng.randint(
                max(2, int(enclosed_grid_rows_min)),
                max(2, int(enclosed_grid_rows_max)) + 1,
            )
        )
        grid_cols = int(
            rng.randint(
                max(2, int(enclosed_grid_cols_min)),
                max(2, int(enclosed_grid_cols_max)) + 1,
            )
        )
        cell_size = float(
            rng.uniform(
                float(enclosed_cell_size_min),
                float(enclosed_cell_size_max),
            )
        )
        wall_thickness = float(
            rng.uniform(
                float(enclosed_wall_thickness_min),
                float(enclosed_wall_thickness_max),
            )
        )
        obstacle_layout, beacon_layout, start_cell, maze_meta = generate_enclosed_maze(
            seed=seed,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            cell_size=cell_size,
            wall_thickness=wall_thickness,
            n_beacons=n_beacons,
            n_distractors=n_distractors,
            return_metadata=True,
        )
        step = cell_size + wall_thickness
        ox = -(grid_cols - 1) * step / 2.0
        oy = -(grid_rows - 1) * step / 2.0
        start_xy = [
            float(ox + start_cell[1] * step),
            float(oy + start_cell[0] * step),
        ]
        x_left = ox - step / 2.0
        x_right = ox + (grid_cols - 1) * step + step / 2.0
        y_bottom = oy - step / 2.0
        y_top = oy + (grid_rows - 1) * step + step / 2.0
        return obstacle_layout, beacon_layout, {
            "scene_seed": int(seed),
            "scene_distribution": scene_distribution,
            "scene_type": "enclosed",
            "scene_label": f"enclosed_{grid_rows}x{grid_cols}",
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "cell_size": cell_size,
            "wall_thickness": wall_thickness,
            "start_cell": [int(start_cell[0]), int(start_cell[1])],
            "start_xy": start_xy,
            "spawn_bounds_xy": [
                float(x_left + 0.5 * wall_thickness),
                float(x_right - 0.5 * wall_thickness),
                float(y_bottom + 0.5 * wall_thickness),
                float(y_top - 0.5 * wall_thickness),
            ],
            "spawn_range": float(max(abs(x_left), abs(x_right), abs(y_bottom), abs(y_top))),
            "n_obstacles": int(len(obstacle_layout.obstacles)),
            "n_beacons": int(len(beacon_layout.beacons)),
            "n_distractors": int(len(beacon_layout.distractors)),
            "maze_meta": maze_meta,
        }

    if scene_kind == "free":
        obstacle_layout = generate_random_layout(seed=seed)
        beacon_layout = BeaconLayout()  # empty
        return obstacle_layout, beacon_layout, {
            "scene_seed": int(seed),
            "scene_distribution": scene_distribution,
            "scene_type": "free",
            "scene_label": "free_obstacles",
            "spawn_range": 2.0,
            "n_obstacles": int(len(obstacle_layout.obstacles)),
            "n_beacons": 0,
            "n_distractors": 0,
        }

    raise AssertionError(f"Unhandled scene_kind={scene_kind}")


# --------------------------------------------------------------------------- #
# Command generation per environment
# --------------------------------------------------------------------------- #

def generate_env_commands(
    n_envs: int,
    steps: int,
    seed: int,
    ou_fraction: float = 0.5,
) -> tuple:
    """Pre-generate mixed command sequences for all envs.

    Half the envs use OU noise (original behaviour), the other half use
    structured command patterns (retreat, recovery, etc.).

    Returns:
        (cmds_array (n_envs, steps, 3), pattern_index (n_envs, steps))
    """
    rng = np.random.RandomState(seed)
    cmds = np.zeros((n_envs, steps, 3), dtype=np.float32)
    pattern_idx = np.zeros((n_envs, steps), dtype=np.int32)

    n_ou = int(n_envs * ou_fraction)

    # OU envs: fill with simple OU random walk
    for i in range(n_ou):
        ou = OUProcess(n_envs=1, dim=3, theta=0.15, sigma=0.3)
        for t in range(steps):
            c = ou.sample(rng)
            cmds[i, t] = np.tanh(c[0]) * np.array([0.40, 0.25, 0.60])
        pattern_idx[i, :] = CMD_PATTERN_INDEX.get("ou_explore", 0)

    # Structured envs: mixed command patterns
    for i in range(n_ou, n_envs):
        env_cmds, segments = build_mixed_command_sequence(
            rng, total_steps=steps,
        )
        cmds[i] = env_cmds
        for start, end, name in segments:
            pattern_idx[i, start:end] = CMD_PATTERN_INDEX.get(name, 0)

    return cmds, pattern_idx


def build_maze_teacher(
    scene_meta: dict,
    n_envs: int,
    teacher_fraction: float,
    seed: int,
    device: str,
) -> dict | None:
    """Build a closed-loop privileged teacher for enclosed-maze collection.

    The teacher follows shortest-path cell routes toward beacon cells or
    dead-end/frontier cells. It is used only during data collection.
    """
    maze_meta = scene_meta.get("maze_meta")
    if maze_meta is None or str(scene_meta.get("scene_type")) != "enclosed":
        return None

    grid_rows = int(maze_meta["grid_rows"])
    grid_cols = int(maze_meta["grid_cols"])
    n_cells = grid_rows * grid_cols

    def rc_to_idx(r: int, c: int) -> int:
        return int(r) * grid_cols + int(c)

    cell_centers = torch.zeros((n_cells, 2), device=device, dtype=torch.float32)
    adjacency: list[list[int]] = [[] for _ in range(n_cells)]
    for key, xy in maze_meta["cell_centers_xy"].items():
        r, c = [int(v) for v in key.split(",")]
        idx = rc_to_idx(r, c)
        cell_centers[idx] = torch.tensor(xy, device=device, dtype=torch.float32)
        adjacency[idx] = [rc_to_idx(nr, nc) for nr, nc in maze_meta["adjacency"][key]]

    next_hop = torch.full((n_cells, n_cells), -1, dtype=torch.long)
    hop_dist = torch.full((n_cells, n_cells), 10_000, dtype=torch.long)
    for src in range(n_cells):
        queue: list[int] = [src]
        parent = {src: src}
        hop_dist[src, src] = 0
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt in adjacency[cur]:
                if nxt in parent:
                    continue
                parent[nxt] = cur
                hop_dist[src, nxt] = hop_dist[src, cur] + 1
                queue.append(nxt)
        for dst in range(n_cells):
            if hop_dist[src, dst] >= 10_000:
                continue
            if src == dst:
                next_hop[src, dst] = src
                continue
            cur = dst
            while parent[cur] != src:
                cur = parent[cur]
            next_hop[src, dst] = cur

    dead_end_cells = [
        rc_to_idx(int(r), int(c))
        for r, c in maze_meta.get("dead_end_cells", [])
    ]
    beacon_cells = [
        rc_to_idx(int(rc[0]), int(rc[1]))
        for rc in maze_meta.get("beacon_cells", {}).values()
    ]
    start_cell = maze_meta.get("start_cell", [0, 0])
    start_cell_idx = rc_to_idx(int(start_cell[0]), int(start_cell[1]))
    all_cells = list(range(n_cells))

    teacher_envs = int(round(float(teacher_fraction) * float(n_envs)))
    teacher_envs = max(0, min(n_envs, teacher_envs))
    if teacher_envs <= 0:
        return None

    mask = torch.zeros((n_envs,), device=device, dtype=torch.bool)
    mask[:teacher_envs] = True
    teacher_slot_by_env = torch.full((n_envs,), -1, device=device, dtype=torch.long)
    teacher_slot_by_env[mask] = torch.arange(teacher_envs, device=device, dtype=torch.long)

    return {
        "rng": torch.Generator(device=device).manual_seed(int(seed)),
        "mask": mask,
        "env_indices": torch.nonzero(mask).squeeze(-1),
        "teacher_slot_by_env": teacher_slot_by_env,
        "target_idx": torch.full((teacher_envs,), start_cell_idx, device=device, dtype=torch.long),
        "target_kind": torch.zeros((teacher_envs,), device=device, dtype=torch.long),
        "target_age": torch.zeros((teacher_envs,), device=device, dtype=torch.long),
        "cell_centers": cell_centers,
        "next_hop": next_hop.to(device=device),
        "hop_dist": hop_dist.to(device=device),
        "all_cells": torch.tensor(all_cells, device=device, dtype=torch.long),
        "beacon_cells": torch.tensor(beacon_cells or dead_end_cells or all_cells, device=device, dtype=torch.long),
        "frontier_cells": torch.tensor(dead_end_cells or all_cells, device=device, dtype=torch.long),
        "start_cell_idx": int(start_cell_idx),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "step": float(maze_meta["step"]),
        "origin_xy": tuple(float(v) for v in maze_meta["origin_xy"]),
    }


def _maze_teacher_choose_targets(
    teacher: dict,
    current_cells: torch.Tensor,
    refresh_mask: torch.Tensor,
    beacon_prob: float,
    frontier_prob: float,
) -> None:
    """Assign new cell targets for the selected teacher envs."""
    if refresh_mask.numel() == 0 or not bool(torch.any(refresh_mask).item()):
        return

    rng = teacher["rng"]
    env_sel = torch.nonzero(refresh_mask).squeeze(-1)
    if env_sel.numel() == 0:
        return

    curr = current_cells[env_sel]
    n = int(env_sel.numel())
    mode_rand = torch.rand((n,), generator=rng, device=curr.device)
    use_beacon = mode_rand < float(beacon_prob)
    use_frontier = (~use_beacon) & (mode_rand < float(beacon_prob + frontier_prob))

    def choose_far_target(candidate_cells: torch.Tensor, subset_mask: torch.Tensor) -> None:
        subset_idx = torch.nonzero(subset_mask).squeeze(-1)
        if subset_idx.numel() == 0:
            return
        cand = candidate_cells
        dist = teacher["hop_dist"][curr[subset_idx]][:, cand].float()
        noise = 0.01 * torch.rand(dist.shape, device=dist.device, generator=rng)
        score = dist + noise
        best = cand[score.argmax(dim=1)]
        teacher["target_idx"][env_sel[subset_idx]] = best

    choose_far_target(teacher["beacon_cells"], use_beacon)
    choose_far_target(teacher["frontier_cells"], use_frontier)
    choose_far_target(teacher["all_cells"], ~(use_beacon | use_frontier))

    teacher["target_kind"][env_sel[use_beacon]] = CMD_PATTERN_INDEX["maze_teacher_beacon"]
    teacher["target_kind"][env_sel[use_frontier]] = CMD_PATTERN_INDEX["maze_teacher_frontier"]
    teacher["target_kind"][env_sel[~(use_beacon | use_frontier)]] = CMD_PATTERN_INDEX["maze_teacher_explore"]
    teacher["target_age"][env_sel] = 0


def compute_maze_teacher_commands(
    teacher: dict | None,
    pos: torch.Tensor,
    quat: torch.Tensor,
    teacher_max_target_steps: int,
    teacher_beacon_prob: float,
    teacher_frontier_prob: float,
    teacher_noise_scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute closed-loop teacher commands for a subset of envs."""
    if teacher is None:
        return None, None

    env_idx = teacher["env_indices"]
    if env_idx.numel() == 0:
        return None, None

    pos_sel = pos[env_idx, :2]
    quat_sel = quat[env_idx]
    ox, oy = teacher["origin_xy"]
    step = float(teacher["step"])
    grid_cols = int(teacher["grid_cols"])
    grid_rows = int(teacher["grid_rows"])

    col = torch.round((pos_sel[:, 0] - ox) / step).long().clamp_(0, grid_cols - 1)
    row = torch.round((pos_sel[:, 1] - oy) / step).long().clamp_(0, grid_rows - 1)
    current_cells = row * grid_cols + col

    teacher["target_age"] += 1
    invalid_target = teacher["target_idx"] < 0
    target_idx_safe = teacher["target_idx"].clone()
    target_idx_safe[invalid_target] = int(teacher["start_cell_idx"])
    target_centers = teacher["cell_centers"][target_idx_safe]
    reached = torch.linalg.vector_norm(target_centers - pos_sel, dim=1) <= max(0.10, 0.20 * step)
    stale = teacher["target_age"] >= int(max(8, teacher_max_target_steps))
    refresh = reached | stale | invalid_target
    _maze_teacher_choose_targets(
        teacher,
        current_cells,
        refresh,
        beacon_prob=float(teacher_beacon_prob),
        frontier_prob=float(teacher_frontier_prob),
    )

    next_cells = teacher["next_hop"][current_cells, teacher["target_idx"]]
    invalid = next_cells < 0
    if bool(torch.any(invalid).item()):
        next_cells = next_cells.clone()
        next_cells[invalid] = current_cells[invalid]

    next_centers = teacher["cell_centers"][next_cells]
    target_centers = teacher["cell_centers"][teacher["target_idx"]]
    delta_next = next_centers - pos_sel
    delta_target = target_centers - pos_sel
    near_next = torch.linalg.vector_norm(delta_next, dim=1) <= max(0.08, 0.16 * step)
    nav_delta = torch.where(near_next.unsqueeze(1), delta_target, delta_next)

    nav_world = torch.zeros((env_idx.numel(), 3), device=pos.device, dtype=torch.float32)
    nav_world[:, :2] = nav_delta
    nav_body = world_to_body_vec(quat_sel, nav_world)[:, :2]
    yaw_err = torch.atan2(nav_body[:, 1], nav_body[:, 0].clamp_min(1e-4))
    dist = torch.linalg.vector_norm(nav_delta, dim=1)

    forward = (0.60 * dist).clamp_(0.0, 0.45)
    lateral = (0.65 * nav_body[:, 1]).clamp_(-0.18, 0.18)
    yaw_rate = (1.75 * yaw_err).clamp_(-1.1, 1.1)

    large_turn = yaw_err.abs() > 0.85
    medium_turn = yaw_err.abs() > 0.40
    forward = torch.where(large_turn, 0.08 * forward, forward)
    forward = torch.where(medium_turn, 0.55 * forward, forward)
    lateral = torch.where(large_turn, torch.zeros_like(lateral), lateral)

    cmds = torch.stack([forward, lateral, yaw_rate], dim=-1)
    if float(teacher_noise_scale) > 0.0:
        noise = torch.randn(cmds.shape, device=cmds.device, generator=teacher["rng"]) * float(teacher_noise_scale)
        noise[:, 0] *= 0.6
        cmds = cmds + noise

    cmds[:, 0] = cmds[:, 0].clamp_(-0.15, 0.50)
    cmds[:, 1] = cmds[:, 1].clamp_(-0.20, 0.20)
    cmds[:, 2] = cmds[:, 2].clamp_(-1.2, 1.2)
    return cmds, teacher["target_kind"].clone()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect trajectory data from frozen PPO policy with mazes/beacons."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to PPO checkpoint.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Timesteps per chunk.")
    parser.add_argument("--chunks", type=int, default=5,
                        help="Number of data chunks to collect.")
    parser.add_argument("--n_envs", type=int, default=2048,
                        help="Number of parallel environments.")
    parser.add_argument("--out_dir", type=str, default="jepa_raw_data",
                        help="Output directory for .npz chunks.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--sim_backend", type=str, default="auto",
                        help="Genesis backend: auto | gpu | cuda | vulkan | metal | cpu.")
    parser.add_argument("--scene_batch_envs", type=int, default=DEFAULT_SCENE_BATCH_ENVS,
                        help="Max envs per Genesis scene build.")
    # New options
    parser.add_argument("--use_mazes", action="store_true", default=True,
                        help="Enable maze topology generation.")
    parser.add_argument(
        "--scene_distribution",
        type=str,
        default="mixed",
        choices=["legacy", "mixed", "composite", "enclosed", "free"],
        help="Scene family to sample per chunk. 'mixed' injects enclosed mazes "
             "so training data better matches deployment topology.",
    )
    parser.add_argument("--n_beacons", type=int, default=2,
                        help="Number of beacons per scene.")
    parser.add_argument("--n_distractors", type=int, default=1,
                        help="Number of distractor patches per scene.")
    parser.add_argument("--n_free_obstacles", type=int, default=2,
                        help="Number of additional free obstacles in composite scenes.")
    parser.add_argument("--arena_half", type=float, default=3.0,
                        help="Arena half-width used for composite scene generation.")
    parser.add_argument("--enclosed_grid_rows_min", type=int, default=4)
    parser.add_argument("--enclosed_grid_rows_max", type=int, default=7)
    parser.add_argument("--enclosed_grid_cols_min", type=int, default=4)
    parser.add_argument("--enclosed_grid_cols_max", type=int, default=7)
    parser.add_argument("--enclosed_cell_size_min", type=float, default=0.50)
    parser.add_argument("--enclosed_cell_size_max", type=float, default=0.70)
    parser.add_argument("--enclosed_wall_thickness_min", type=float, default=0.18)
    parser.add_argument("--enclosed_wall_thickness_max", type=float, default=0.24)
    parser.add_argument("--ou_fraction", type=float, default=0.5,
                        help="Fraction of envs using OU commands (rest use structured patterns).")
    parser.add_argument(
        "--command_policy",
        type=str,
        default="mixed",
        choices=["open_loop", "maze_teacher", "mixed"],
        help="How to generate locomotion commands. 'maze_teacher' uses a privileged "
             "closed-loop teacher on enclosed mazes; 'mixed' blends it with the "
             "older open-loop curriculum.",
    )
    parser.add_argument("--maze_teacher_fraction", type=float, default=0.35,
                        help="Fraction of envs controlled by the closed-loop maze teacher when it is enabled.")
    parser.add_argument("--maze_teacher_max_target_steps", type=int, default=120,
                        help="Retarget teacher-controlled envs after this many steps if they have not reached the current cell target.")
    parser.add_argument("--maze_teacher_beacon_prob", type=float, default=0.60,
                        help="When retargeting, probability of selecting a beacon cell as the next teacher target.")
    parser.add_argument("--maze_teacher_frontier_prob", type=float, default=0.30,
                        help="When retargeting without selecting a beacon, probability of selecting a dead-end/frontier cell.")
    parser.add_argument("--maze_teacher_noise_scale", type=float, default=0.03,
                        help="Small command noise added to the teacher for trajectory diversity.")
    parser.add_argument("--soft_collision_prob", type=float, default=0.3,
                        help="Probability of NOT terminating on collision (allows recovery).")
    parser.add_argument("--min_z", type=float, default=SimConfig.min_z,
                        help="Base-height threshold below which an env is marked fallen.")
    parser.add_argument(
        "--respawn_strategy",
        type=str,
        default="auto",
        choices=["auto", "uniform_safe", "scene_start", "scene_start_jitter"],
        help="How to place initial episodes and resets. 'auto' uses scene-start "
             "jitter for enclosed mazes and uniform safe sampling otherwise.",
    )
    parser.add_argument("--start_jitter_radius", type=float, default=0.18,
                        help="XY jitter radius (m) around the designated scene start when using scene_start_jitter.")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        n_envs=args.n_envs,
        soft_collision_prob=args.soft_collision_prob,
        min_z=args.min_z,
    )
    scene_batch_envs = cfg.n_envs if args.scene_batch_envs <= 0 else min(args.scene_batch_envs, cfg.n_envs)
    n_scene_batches = math.ceil(cfg.n_envs / scene_batch_envs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nPhysics rollout configuration:")
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Envs        : {cfg.n_envs}")
    print(f"  Scene batch : {scene_batch_envs} ({n_scene_batches} scene(s)/chunk)")
    print(f"  Steps/chunk : {args.steps}")
    print(f"  Chunks      : {args.chunks}")
    print(f"  Mazes       : {args.use_mazes}")
    print(f"  Scene dist  : {args.scene_distribution}")
    print(f"  Beacons     : {args.n_beacons}")
    print(f"  OU fraction : {args.ou_fraction}")
    print(f"  Cmd policy  : {args.command_policy}")
    print(f"  Teacher frac: {args.maze_teacher_fraction}")
    print(f"  Soft collide: {cfg.soft_collision_prob}")
    print(f"  Min z       : {cfg.min_z}")
    print(f"  Respawn     : {args.respawn_strategy}")
    print(f"  Output      : {out_dir.resolve()}")
    print()

    # -------------------------------------------------------------------- #
    # Chunk loop
    # -------------------------------------------------------------------- #
    total_steps_all = args.chunks * args.steps
    global_pbar = tqdm(
        total=total_steps_all,
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for chunk_idx in range(args.chunks):
        chunk_seed = args.seed + chunk_idx
        print(f"--- Chunk {chunk_idx + 1}/{args.chunks} (seed={chunk_seed}) ---")

        # Generate scene
        obstacle_layout, beacon_layout, scene_meta = generate_scene(
            seed=chunk_seed,
            scene_distribution=args.scene_distribution,
            use_maze=args.use_mazes,
            n_beacons=args.n_beacons,
            n_distractors=args.n_distractors,
            n_free_obstacles=args.n_free_obstacles,
            arena_half=args.arena_half,
            enclosed_grid_rows_min=args.enclosed_grid_rows_min,
            enclosed_grid_rows_max=args.enclosed_grid_rows_max,
            enclosed_grid_cols_min=args.enclosed_grid_cols_min,
            enclosed_grid_cols_max=args.enclosed_grid_cols_max,
            enclosed_cell_size_min=args.enclosed_cell_size_min,
            enclosed_cell_size_max=args.enclosed_cell_size_max,
            enclosed_wall_thickness_min=args.enclosed_wall_thickness_min,
            enclosed_wall_thickness_max=args.enclosed_wall_thickness_max,
        )
        print(f"  Scene: {scene_meta['scene_label']} ({scene_meta['scene_type']}) | {len(obstacle_layout.obstacles)} obstacles "
              f"| {len(beacon_layout.beacons)} beacons | {len(beacon_layout.distractors)} distractors")

        # Pre-generate the open-loop command baseline. Closed-loop teacher
        # commands, when enabled, overwrite a subset of envs online.
        env_cmds_np, cmd_pattern_np = generate_env_commands(
            cfg.n_envs, args.steps, seed=chunk_seed + 1000,
            ou_fraction=args.ou_fraction,
        )

        # ---- Per-chunk storage ------------------------------------------- #
        d_proprio    = torch.zeros((cfg.n_envs, args.steps, 47), dtype=torch.float32, device="cpu")
        d_cmds       = torch.zeros((cfg.n_envs, args.steps, 3),  dtype=torch.float32, device="cpu")
        d_dones      = torch.zeros((cfg.n_envs, args.steps),      dtype=torch.bool,    device="cpu")
        d_base_pos   = torch.zeros((cfg.n_envs, args.steps, 3),  dtype=torch.float32, device="cpu")
        d_base_quat  = torch.zeros((cfg.n_envs, args.steps, 4),  dtype=torch.float32, device="cpu")
        d_joint_pos  = torch.zeros((cfg.n_envs, args.steps, 12), dtype=torch.float32, device="cpu")
        d_collisions = torch.zeros((cfg.n_envs, args.steps),      dtype=torch.bool,    device="cpu")

        t0 = time.time()
        total_resets = 0
        model = None
        try:
            init_genesis_once(args.sim_backend)
            model = load_frozen_policy(args.ckpt)

            for batch_idx, env_start in enumerate(range(0, cfg.n_envs, scene_batch_envs)):
                env_end = min(env_start + scene_batch_envs, cfg.n_envs)
                batch_n_envs = env_end - env_start
                batch_slice = slice(env_start, env_end)

                print(
                    f"  Scene batch {batch_idx + 1}/{n_scene_batches}: "
                    f"envs {env_start}-{env_end - 1} ({batch_n_envs})"
                )

                scene = None
                try:
                    scene = gs.Scene(show_viewer=False)
                    scene.add_entity(gs.morphs.Plane())
                    robot = scene.add_entity(
                        gs.morphs.URDF(file=URDF_PATH, pos=ROBOT_SPAWN, fixed=False)
                    )
                    add_obstacles_to_scene(scene, obstacle_layout)

                    # Add beacon panels and distractors as box entities
                    for obs_spec in beacon_layout.all_obstacles():
                        scene.add_entity(
                            gs.morphs.Box(pos=obs_spec.pos, size=obs_spec.size, fixed=True),
                            surface=gs.surfaces.Rough(color=obs_spec.color),
                        )

                    scene.build(n_envs=batch_n_envs)

                    # ---- Joint indexing ----------------------------------- #
                    name_to_joint = {j.name: j for j in robot.joints}
                    missing = [jn for jn in JOINTS_ACTUATED if jn not in name_to_joint]
                    if missing:
                        print(f"ERROR: missing joints in URDF: {missing}")
                        sys.exit(1)

                    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0]
                               for jn in JOINTS_ACTUATED]
                    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)
                    q0 = torch.tensor(Q0_VALUES, device=gs.device, dtype=torch.float32)

                    robot.set_dofs_kp(torch.ones(12, device=gs.device) * cfg.kp, act_dofs)
                    robot.set_dofs_kv(torch.ones(12, device=gs.device) * cfg.kv, act_dofs)

                    teacher_fraction = 0.0
                    if args.command_policy == "maze_teacher":
                        teacher_fraction = 1.0
                    elif args.command_policy == "mixed":
                        teacher_fraction = float(args.maze_teacher_fraction)
                    maze_teacher = build_maze_teacher(
                        scene_meta,
                        n_envs=batch_n_envs,
                        teacher_fraction=teacher_fraction,
                        seed=chunk_seed + 7000 + batch_idx,
                        device=gs.device,
                    )

                    # ---- Per-scene state --------------------------------- #
                    ou_noise = OUNoiseBatched(batch_n_envs, 3, gs.device)
                    latency_buffer = torch.zeros((2, batch_n_envs, 3), device=gs.device)
                    prev_a = torch.zeros((batch_n_envs, 12), device=gs.device)

                    # Pre-computed commands for this batch (moved to GPU)
                    batch_cmds_pre = torch.from_numpy(
                        env_cmds_np[env_start:env_end]
                    ).to(gs.device)

                    # Soft-collision random draws (per env, per step)
                    soft_rng = np.random.RandomState(chunk_seed + batch_idx)
                    soft_survive = torch.from_numpy(
                        soft_rng.rand(batch_n_envs, args.steps).astype(np.float32)
                    ).to(gs.device) < cfg.soft_collision_prob

                    initial_xy = sample_episode_spawn_positions(
                        batch_n_envs,
                        obstacle_layout,
                        scene_meta,
                        clearance=cfg.safe_clearance,
                        strategy=args.respawn_strategy,
                        start_jitter_radius=args.start_jitter_radius,
                        device=gs.device,
                    )
                    initial_pos = torch.zeros((batch_n_envs, 3), device=gs.device)
                    initial_pos[:, 0] = initial_xy[:, 0]
                    initial_pos[:, 1] = initial_xy[:, 1]
                    initial_pos[:, 2] = ROBOT_SPAWN[2]
                    initial_yaw = torch.rand(batch_n_envs, device=gs.device) * 2 * math.pi
                    initial_quat = torch.zeros((batch_n_envs, 4), device=gs.device)
                    initial_quat[:, 0] = torch.cos(initial_yaw * 0.5)
                    initial_quat[:, 3] = torch.sin(initial_yaw * 0.5)
                    robot.set_dofs_position(
                        q0.unsqueeze(0).expand(batch_n_envs, -1), act_dofs,
                    )
                    robot.set_dofs_velocity(
                        torch.zeros((batch_n_envs, 12), device=gs.device), act_dofs,
                    )
                    robot.set_pos(initial_pos, zero_velocity=True)
                    robot.set_quat(initial_quat, zero_velocity=False)

                    for step in range(args.steps):
                        # -- Read robot state ------------------------------- #
                        pos = robot.get_pos()
                        quat = robot.get_quat()
                        vel_b = world_to_body_vec(quat, robot.get_vel())
                        ang_b = world_to_body_vec(quat, robot.get_ang())
                        q = robot.get_dofs_position(act_dofs)
                        dq = robot.get_dofs_velocity(act_dofs)
                        q_rel = q - q0.unsqueeze(0)

                        proprio = torch.cat(
                            [pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_a], dim=1
                        )

                        # -- Synthetic sensor noise ------------------------- #
                        noise = torch.randn_like(proprio) * 0.01
                        noise[:, 1:5] *= 2.0
                        noise[:, 5:11] *= 5.0
                        proprio_noisy = proprio + noise

                        # -- Command generation ----------------------------- #
                        scaled_cmds = batch_cmds_pre[:, step, :].clone()
                        if maze_teacher is not None:
                            teacher_cmds, teacher_pattern = compute_maze_teacher_commands(
                                maze_teacher,
                                pos,
                                quat,
                                teacher_max_target_steps=int(args.maze_teacher_max_target_steps),
                                teacher_beacon_prob=float(args.maze_teacher_beacon_prob),
                                teacher_frontier_prob=float(args.maze_teacher_frontier_prob),
                                teacher_noise_scale=float(args.maze_teacher_noise_scale),
                            )
                            if teacher_cmds is not None:
                                teacher_env_idx = maze_teacher["env_indices"]
                                scaled_cmds[teacher_env_idx] = teacher_cmds
                                pattern_step = cmd_pattern_np[env_start:env_end, step]
                                pattern_step[
                                    teacher_env_idx.detach().cpu().numpy()
                                ] = teacher_pattern.detach().cpu().numpy()

                        # Two-step command latency buffer
                        latency_buffer = torch.roll(latency_buffer, shifts=-1, dims=0)
                        latency_buffer[-1] = scaled_cmds
                        active_cmds = latency_buffer[0]

                        # -- Forward pass through frozen policy ------------- #
                        obs = torch.cat([proprio_noisy, active_cmds], dim=1)
                        actions = model.act_deterministic(obs)
                        prev_a = actions.clone()

                        # -- Compute joint targets and step physics --------- #
                        q_tgt = q0.unsqueeze(0) + cfg.action_scale * actions
                        q_tgt[:, 0:4] = torch.clamp(q_tgt[:, 0:4], -0.8, 0.8)
                        q_tgt[:, 4:8] = torch.clamp(q_tgt[:, 4:8], -1.5, 1.5)
                        q_tgt[:, 8:12] = torch.clamp(q_tgt[:, 8:12], -2.5, -0.5)

                        robot.control_dofs_position(q_tgt, act_dofs)
                        for _ in range(cfg.decimation):
                            scene.step()

                        # -- Termination: falls and collisions -------------- #
                        fallen = pos[:, 2] < cfg.min_z
                        colliding = detect_collisions(
                            pos[:, :2], obstacle_layout, margin=cfg.collision_margin
                        )

                        # Soft collision: some colliding envs survive (for recovery data)
                        hard_collision = colliding & ~soft_survive[:, step]
                        done = fallen | hard_collision

                        # -- Reset environments that are done --------------- #
                        done_ids = torch.nonzero(done).squeeze(-1)
                        if done_ids.numel() > 0:
                            total_resets += done_ids.numel()
                            ou_noise.reset(done_ids)
                            prev_a[done_ids] = 0.0
                            if maze_teacher is not None:
                                teacher_slots = maze_teacher["teacher_slot_by_env"][done_ids]
                                teacher_slots = teacher_slots[teacher_slots >= 0]
                                if teacher_slots.numel() > 0:
                                    maze_teacher["target_idx"][teacher_slots] = -1
                                    maze_teacher["target_age"][teacher_slots] = int(args.maze_teacher_max_target_steps)

                            n_reset = done_ids.numel()
                            robot.set_dofs_position(
                                q0.unsqueeze(0).expand(n_reset, -1), act_dofs, envs_idx=done_ids
                            )
                            robot.set_dofs_velocity(
                                torch.zeros((n_reset, 12), device=gs.device), act_dofs, envs_idx=done_ids
                            )

                            safe_xy = sample_episode_spawn_positions(
                                n_reset,
                                obstacle_layout,
                                scene_meta,
                                clearance=cfg.safe_clearance,
                                strategy=args.respawn_strategy,
                                start_jitter_radius=args.start_jitter_radius,
                                device=gs.device,
                            )

                            respawn_pos = torch.zeros((n_reset, 3), device=gs.device)
                            respawn_pos[:, 0] = safe_xy[:, 0]
                            respawn_pos[:, 1] = safe_xy[:, 1]
                            respawn_pos[:, 2] = ROBOT_SPAWN[2]

                            yaw_angles = torch.rand(n_reset, device=gs.device) * 2 * math.pi
                            respawn_quat = torch.zeros((n_reset, 4), device=gs.device)
                            respawn_quat[:, 0] = torch.cos(yaw_angles * 0.5)
                            respawn_quat[:, 3] = torch.sin(yaw_angles * 0.5)

                            robot.set_pos(respawn_pos, envs_idx=done_ids, zero_velocity=True)
                            robot.set_quat(respawn_quat, envs_idx=done_ids, zero_velocity=False)

                        # -- Record data ------------------------------------ #
                        d_proprio[batch_slice, step] = proprio_noisy.cpu()
                        d_cmds[batch_slice, step] = scaled_cmds.cpu()
                        d_dones[batch_slice, step] = done.cpu()
                        d_base_pos[batch_slice, step] = pos.cpu()
                        d_base_quat[batch_slice, step] = quat.cpu()
                        d_joint_pos[batch_slice, step] = q.cpu()
                        d_collisions[batch_slice, step] = colliding.cpu()

                        global_pbar.update(1)
                        global_pbar.set_description(
                            f"Chunk {chunk_idx + 1}/{args.chunks} | "
                            f"step {step + 1}/{args.steps} | "
                            f"resets {total_resets}"
                        )
                finally:
                    if scene is not None:
                        scene.destroy()
                        del scene
                        gc.collect()
        finally:
            if model is not None:
                del model
            if getattr(gs, "_initialized", False):
                gs.destroy()
            gc.collect()

        # -- Compute labels post-hoc (CPU) --------------------------------- #
        print("  Computing labels...")
        base_pos_np = d_base_pos.numpy()
        base_quat_np = d_base_quat.numpy()

        # Extract yaw from quaternion (w,x,y,z) → yaw
        # yaw = atan2(2*(wz + xy), 1 - 2*(yy + zz))
        w, x, y, z = base_quat_np[..., 0], base_quat_np[..., 1], base_quat_np[..., 2], base_quat_np[..., 3]
        yaw_np = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Compute labels per env
        clearance_np = np.zeros((cfg.n_envs, args.steps), dtype=np.float32)
        near_miss_np = np.zeros((cfg.n_envs, args.steps), dtype=bool)
        traversability_np = np.zeros((cfg.n_envs, args.steps), dtype=np.int32)
        beacon_visible_np = np.zeros((cfg.n_envs, args.steps), dtype=bool)
        beacon_identity_np = np.full((cfg.n_envs, args.steps), -1, dtype=np.int32)
        beacon_bearing_np = np.zeros((cfg.n_envs, args.steps), dtype=np.float32)
        beacon_range_np = np.full((cfg.n_envs, args.steps), float("inf"), dtype=np.float32)

        for env_i in range(cfg.n_envs):
            labels = compute_episode_labels(
                robot_xy=base_pos_np[env_i, :, :2],
                robot_yaw=yaw_np[env_i],
                obstacle_layout=obstacle_layout,
                beacon_layout=beacon_layout if len(beacon_layout.beacons) > 0 else None,
                near_miss_threshold=cfg.near_miss_threshold,
            )
            clearance_np[env_i] = labels["clearance"]
            near_miss_np[env_i] = labels["near_miss"]
            traversability_np[env_i] = labels["traversability"]
            beacon_visible_np[env_i] = labels["beacon_visible"]
            beacon_identity_np[env_i] = labels["beacon_identity"]
            beacon_bearing_np[env_i] = labels["beacon_bearing"]
            beacon_range_np[env_i] = labels["beacon_range"]

        # -- Save chunk ---------------------------------------------------- #
        chunk_path = out_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(
            str(chunk_path),
            proprio=d_proprio.numpy(),
            cmds=d_cmds.numpy(),
            dones=d_dones.numpy(),
            base_pos=d_base_pos.numpy(),
            base_quat=d_base_quat.numpy(),
            joint_pos=d_joint_pos.numpy(),
            collisions=d_collisions.numpy(),
            clearance=clearance_np,
            near_miss=near_miss_np,
            traversability=traversability_np,
            beacon_visible=beacon_visible_np,
            beacon_identity=beacon_identity_np,
            beacon_bearing=beacon_bearing_np,
            beacon_range=beacon_range_np,
            cmd_pattern=cmd_pattern_np,
            obstacle_layout=np.array(obstacle_layout.to_json()),
            beacon_layout=np.array(beacon_layout.to_json()),
            scene_seed=np.array(int(scene_meta["scene_seed"]), dtype=np.int64),
            scene_type=np.array(str(scene_meta["scene_type"])),
            scene_meta=np.array(json.dumps(scene_meta)),
        )

        chunk_elapsed = time.time() - t0
        chunk_fps = (cfg.n_envs * args.steps) / chunk_elapsed
        size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {chunk_path} ({size_mb:.1f} MB)  |  "
              f"Chunk FPS: {chunk_fps:,.0f}  |  Total resets: {total_resets}\n")

    global_pbar.close()
    print("Physics rollout complete.")


if __name__ == "__main__":
    main()
