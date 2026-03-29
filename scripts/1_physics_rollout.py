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
from lewm.maze_utils import generate_maze, generate_composite_scene, MAZE_STYLES
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
CMD_PATTERN_NAMES = list(COMMAND_PATTERNS.keys())
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
    min_z: float = 0.05
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
    max_attempts: int = 200,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample n (x, y) positions that are >= clearance from every obstacle."""
    positions = torch.zeros((n, 2), device=device)
    filled = 0
    attempts = 0

    while filled < n and attempts < max_attempts * n:
        batch_size = min(n - filled, 256)
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
    use_maze: bool = True,
    n_beacons: int = 2,
    n_distractors: int = 1,
    n_free_obstacles: int = 2,
    arena_half: float = 3.0,
) -> tuple:
    """Generate a scene (obstacles + beacons) for one chunk.

    Returns:
        (obstacle_layout, beacon_layout, maze_style)
    """
    rng = np.random.RandomState(seed)

    if use_maze and rng.rand() < 0.7:
        # 70% of scenes use a maze layout
        maze_style = rng.choice(MAZE_STYLES)
        obstacle_layout, beacon_layout = generate_composite_scene(
            seed=seed,
            maze_style=maze_style,
            n_free_obstacles=n_free_obstacles,
            n_beacons=n_beacons,
            n_distractors=n_distractors,
            arena_half=arena_half,
        )
        return obstacle_layout, beacon_layout, maze_style
    else:
        # 30% use the original free-obstacle layout
        obstacle_layout = generate_random_layout(seed=seed)
        beacon_layout = BeaconLayout()  # empty
        return obstacle_layout, beacon_layout, "free_obstacles"


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
    parser.add_argument("--n_beacons", type=int, default=2,
                        help="Number of beacons per scene.")
    parser.add_argument("--n_distractors", type=int, default=1,
                        help="Number of distractor patches per scene.")
    parser.add_argument("--ou_fraction", type=float, default=0.5,
                        help="Fraction of envs using OU commands (rest use structured patterns).")
    parser.add_argument("--soft_collision_prob", type=float, default=0.3,
                        help="Probability of NOT terminating on collision (allows recovery).")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(n_envs=args.n_envs, soft_collision_prob=args.soft_collision_prob)
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
    print(f"  Beacons     : {args.n_beacons}")
    print(f"  OU fraction : {args.ou_fraction}")
    print(f"  Soft collide: {cfg.soft_collision_prob}")
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
        obstacle_layout, beacon_layout, maze_style = generate_scene(
            seed=chunk_seed,
            use_maze=args.use_mazes,
            n_beacons=args.n_beacons,
            n_distractors=args.n_distractors,
        )
        print(f"  Scene: {maze_style} | {len(obstacle_layout.obstacles)} obstacles "
              f"| {len(beacon_layout.beacons)} beacons | {len(beacon_layout.distractors)} distractors")

        # Pre-generate command sequences
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

                    for step in range(args.steps):
                        # -- Command: use pre-generated mixed commands ------ #
                        scaled_cmds = batch_cmds_pre[:, step, :]

                        # Two-step command latency buffer
                        latency_buffer = torch.roll(latency_buffer, shifts=-1, dims=0)
                        latency_buffer[-1] = scaled_cmds
                        active_cmds = latency_buffer[0]

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

                            n_reset = done_ids.numel()
                            robot.set_dofs_position(
                                q0.unsqueeze(0).expand(n_reset, -1), act_dofs, envs_idx=done_ids
                            )
                            robot.set_dofs_velocity(
                                torch.zeros((n_reset, 12), device=gs.device), act_dofs, envs_idx=done_ids
                            )

                            safe_xy = sample_safe_positions(
                                n_reset, obstacle_layout,
                                clearance=cfg.safe_clearance, device=gs.device,
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
