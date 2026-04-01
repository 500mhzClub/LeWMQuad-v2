#!/usr/bin/env python3
"""Perception-only maze exploration with CEM planning.

Pure perception — no oracle labels in the planning loop.

Pipeline:
  1. Generate an enclosed grid maze with hidden beacons.
  2. Render one front-face view of the target beacon → encode as breadcrumb.
  3. Spawn robot at the maze start cell.
  4. Main loop: render ego frame → encode → CEM plan → PPO execute → step.
  5. Post-loop: oracle evaluation for metrics, export video + trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from PIL import Image, ImageDraw

from lewm.beacon_utils import BEACON_FAMILIES, BeaconLayout
from lewm.camera_utils import (
    EgoCameraConfig,
    add_egocentric_camera_args,
    camera_rotation_matrix,
    camera_safety_metrics,
    ego_camera_config_from_args,
    egocentric_camera_pose,
    retract_camera_to_safe,
)
from lewm.checkpoint_utils import clean_state_dict, load_ppo_checkpoint
from lewm.genesis_utils import init_genesis_once, to_numpy
from lewm.math_utils import quat_to_yaw, world_to_body_vec, yaw_to_quat
from lewm.maze_utils import generate_enclosed_maze
from lewm.models import (
    ActorCritic,
    ExplorationBonus,
    GoalEnergyHead,
    LatentEnergyHead,
    LeWorldModel,
    TrajectoryScorer,
)
from lewm.obstacle_utils import add_obstacles_to_scene, detect_collisions

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ---- Constants ----------------------------------------------------------- #

JOINTS_ACTUATED = [
    "lf_hip_joint",  "lh_hip_joint",  "rf_hip_joint",  "rh_hip_joint",
    "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]
Q0_VALUES = [
    0.06, 0.06, -0.06, -0.06,
    0.85, 0.85, 0.85, 0.85,
    -1.75, -1.75, -1.75, -1.75,
]
PHYSICS_URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
EGO_RENDER_URDF_PATH = "assets/mini_pupper/mini_pupper_render.urdf"
THIRD_PERSON_URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"  # full body visible
ROBOT_SPAWN_Z = 0.12
BEACON_IDENTITY_NAMES = list(BEACON_FAMILIES.keys())


# ---- Dataclasses --------------------------------------------------------- #

@dataclass(frozen=True)
class RobotSimConfig:
    kp: float = 5.0
    kv: float = 0.5
    action_scale: float = 0.30
    decimation: int = 4
    collision_margin: float = 0.15
    min_z: float = 0.04


@dataclass
class PlanningStats:
    best_cost: float
    mean_cost: float
    std_cost: float
    elite_cost: float


# ---- CEM planner --------------------------------------------------------- #

class CEMPlanner:
    """CEM over velocity-command sequences scored in latent space."""

    def __init__(
        self,
        world_model: LeWorldModel,
        scorer: TrajectoryScorer,
        horizon: int,
        n_candidates: int,
        cem_iters: int,
        elite_frac: float,
        cmd_low: torch.Tensor,
        cmd_high: torch.Tensor,
        init_std: torch.Tensor,
        min_std: torch.Tensor,
        forward_reward_weight: float,
        device: torch.device,
    ):
        self.world_model = world_model
        self.scorer = scorer
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.cem_iters = int(cem_iters)
        self.n_elite = max(1, int(round(self.n_candidates * elite_frac)))
        self.cmd_low = cmd_low.to(device=device, dtype=torch.float32)
        self.cmd_high = cmd_high.to(device=device, dtype=torch.float32)
        self.init_std = init_std.to(device=device, dtype=torch.float32)
        self.min_std = min_std.to(device=device, dtype=torch.float32)
        self.forward_reward_weight = float(forward_reward_weight)
        self.device = device
        self._warm_start: torch.Tensor | None = None

    def reset(self) -> None:
        self._warm_start = None

    def _initial_mean(self, last_cmd: torch.Tensor | None) -> torch.Tensor:
        if self._warm_start is not None:
            return self._warm_start.clone()
        if last_cmd is None:
            seed = 0.5 * (self.cmd_low + self.cmd_high)
        else:
            seed = last_cmd.to(self.device, dtype=torch.float32)
        return seed.unsqueeze(0).repeat(self.horizon, 1)

    @torch.no_grad()
    def plan(
        self,
        z_start_raw: torch.Tensor,
        z_goal_proj: torch.Tensor | None = None,
        last_cmd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PlanningStats]:
        mean = self._initial_mean(last_cmd)
        std = self.init_std.unsqueeze(0).repeat(self.horizon, 1)
        best_seq = mean.clone()
        best_cost = float("inf")
        best_costs = None

        z0 = z_start_raw.to(self.device, dtype=torch.float32)
        if z0.ndim != 2 or z0.shape[0] != 1:
            raise ValueError(f"Expected z_start_raw shape (1, D), got {tuple(z0.shape)}")
        z0_batch = z0.repeat(self.n_candidates, 1)

        z_goal_batch = None
        if z_goal_proj is not None:
            z_goal_batch = z_goal_proj.to(self.device, dtype=torch.float32).repeat(self.n_candidates, 1)

        for _ in range(self.cem_iters):
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_candidates, self.horizon, 3, device=self.device,
            )
            samples = samples.clamp(self.cmd_low.view(1, 1, 3), self.cmd_high.view(1, 1, 3))
            samples[0] = mean  # always evaluate the current mean

            z_rollouts = self.world_model.plan_rollout(z0_batch, samples)

            # Full TrajectoryScorer: safety + goal + exploration
            costs = self.scorer.score(z_rollouts, z_goal=z_goal_batch if z_goal_batch is not None else None)

            # Forward reward with soft safety gating: encourages forward
            # motion while moderately backing off near walls.
            if self.forward_reward_weight > 0.0:
                safety_cost = self.scorer.safety_head.score_trajectory(z_rollouts)
                safety_mean = safety_cost / float(max(1, self.horizon))
                # Soft gate: sigmoid-like, only suppresses at very high safety
                bonus_gate = torch.exp(-0.3 * safety_mean.detach())
                forward_bonus = samples[:, :, 0].clamp_min(0.0).sum(dim=-1)
                costs = costs - self.forward_reward_weight * bonus_gate * forward_bonus

            # Yaw oscillation penalty: discourages indecisive spinning
            yaw_penalty_weight = 0.15
            yaw_rates = samples[:, :, 2]  # (N, H)
            yaw_abs = yaw_rates.abs().sum(dim=-1)
            costs = costs + yaw_penalty_weight * yaw_abs

            min_cost, min_idx = torch.min(costs, dim=0)
            if float(min_cost.item()) < best_cost:
                best_cost = float(min_cost.item())
                best_seq = samples[int(min_idx.item())].detach().clone()
                best_costs = costs.detach()

            elite_idx = torch.topk(costs, k=self.n_elite, largest=False).indices
            elite = samples[elite_idx]
            elite_costs = costs[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(self.min_std)
            elite_mean_cost = float(elite_costs.mean().item())

        # Warm-start: shift sequence forward by one step
        shifted = torch.cat([best_seq[1:], best_seq[-1:].clone()], dim=0)
        self._warm_start = shifted.detach()

        if best_costs is None:
            best_costs = torch.tensor([best_cost], device=self.device, dtype=torch.float32)
        stats = PlanningStats(
            best_cost=float(best_cost),
            mean_cost=float(best_costs.mean().item()),
            std_cost=float(best_costs.std(unbiased=False).item()),
            elite_cost=float(elite_mean_cost),
        )
        return best_seq, stats


# ---- Argument parsing ---------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perception-only maze exploration with CEM planning.")
    # Checkpoints
    parser.add_argument("--ppo_ckpt", type=str, required=True)
    parser.add_argument("--wm_ckpt", type=str, required=True)
    parser.add_argument("--scorer_ckpt", type=str, required=True)
    # Maze generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--cell_size", type=float, default=0.70)
    parser.add_argument("--wall_thickness", type=float, default=0.20)
    parser.add_argument("--n_beacons", type=int, default=2)
    parser.add_argument("--n_distractors", type=int, default=0)
    parser.add_argument("--target_beacon", type=str, default=None,
                        help="Beacon colour to seek (default: furthest from start)")
    # Simulation
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim_backend", type=str, default="auto")
    parser.add_argument("--show_viewer", action="store_true")
    # CEM planner
    parser.add_argument("--plan_horizon", type=int, default=8)
    parser.add_argument("--n_candidates", type=int, default=256)
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--elite_frac", type=float, default=0.15)
    parser.add_argument("--cmd_low", type=float, nargs=3, default=[-0.4, -0.3, -1.0])
    parser.add_argument("--cmd_high", type=float, nargs=3, default=[0.8, 0.3, 1.0])
    parser.add_argument("--cem_init_std", type=float, nargs=3, default=[0.3, 0.15, 0.25])
    parser.add_argument("--cem_min_std", type=float, nargs=3, default=[0.05, 0.03, 0.08])
    parser.add_argument("--forward_reward_weight", type=float, default=2.0,
                        help="Safety-gated forward velocity bonus (prevents backing up)")
    # PPO noise
    parser.add_argument("--ppo_obs_noise_std", type=float, default=0.0)
    # Success / termination
    parser.add_argument("--success_range", type=float, default=0.4)
    parser.add_argument("--terminate_on_collision", action="store_true")
    # Output
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_gif", action="store_true")
    parser.add_argument("--no_topdown", action="store_true")
    parser.add_argument("--video_format", type=str, default="auto", choices=["auto", "mp4", "gif", "both"])
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--gif_stride", type=int, default=2)
    # Camera
    add_egocentric_camera_args(parser)
    # Third-person camera
    parser.add_argument("--third_person_res", type=int, default=480)
    parser.add_argument("--third_person_fov", type=float, default=60.0)
    parser.add_argument("--chase_dist", type=float, default=0.6)
    parser.add_argument("--chase_height", type=float, default=0.45)
    parser.add_argument("--side_offset", type=float, default=0.15)
    parser.add_argument("--lookahead", type=float, default=0.3)
    # Breadcrumb view
    parser.add_argument("--breadcrumb_view_dist", type=float, default=0.5)

    return parser.parse_args()


# ---- Model loading ------------------------------------------------------- #

def clean_load_state(module: torch.nn.Module, state_dict: dict[str, Any], *, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(clean_state_dict(state_dict), strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch. missing={missing}, unexpected={unexpected}")


def infer_world_model_kwargs(model_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    pos_embed = model_state["encoder.vis_enc.pos_embed"]
    patch_weight = model_state["encoder.vis_enc.patch_embed.weight"]
    pred_pos_embed = model_state["predictor.pos_embed"]
    latent_dim = int(pos_embed.shape[-1])
    patch_size = int(patch_weight.shape[-1])
    n_tokens = int(pos_embed.shape[1] - 1)
    grid = int(round(math.sqrt(n_tokens)))
    image_size = grid * patch_size
    max_seq_len = int(pred_pos_embed.shape[1])
    use_proprio = any(k.startswith("encoder.prop_enc.") for k in model_state)
    return {
        "latent_dim": latent_dim,
        "image_size": image_size,
        "patch_size": patch_size,
        "max_seq_len": max_seq_len,
        "use_proprio": use_proprio,
    }


def load_world_model(ckpt_path: str, device: torch.device) -> tuple[LeWorldModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = clean_state_dict(ckpt["model_state_dict"])
    kwargs = infer_world_model_kwargs(state)
    model = LeWorldModel(
        latent_dim=kwargs["latent_dim"],
        image_size=kwargs["image_size"],
        patch_size=kwargs["patch_size"],
        max_seq_len=kwargs["max_seq_len"],
        use_proprio=kwargs["use_proprio"],
    ).to(device)
    clean_load_state(model, state)
    model.eval()
    return model, kwargs


def load_trajectory_scorer(ckpt_path: str, device: torch.device) -> tuple[TrajectoryScorer, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = int(ckpt.get("latent_dim", 192))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout = float(ckpt.get("dropout", 0.0))
    exploration_dim = int(ckpt.get("exploration_feature_dim", 128))

    safety = LatentEnergyHead(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    clean_load_state(safety, ckpt["safety_head"])
    safety.eval()

    goal = None
    if ckpt.get("goal_head") is not None:
        goal = GoalEnergyHead(latent_dim=latent_dim, dropout=dropout).to(device)
        clean_load_state(goal, ckpt["goal_head"])
        goal.eval()

    exploration = None
    if ckpt.get("exploration") is not None:
        exploration = ExplorationBonus(latent_dim=latent_dim, feature_dim=exploration_dim).to(device)
        clean_load_state(exploration, ckpt["exploration"])
        exploration.eval()

    scorer = TrajectoryScorer(
        safety_head=safety,
        goal_head=goal,
        exploration=exploration,
        goal_weight=float(ckpt.get("goal_weight", 1.0)),
        exploration_weight=float(ckpt.get("exploration_weight", 0.1)),
    ).to(device)
    scorer.eval()
    return scorer, ckpt


def load_frozen_policy(ckpt_path: str, gs) -> ActorCritic:
    model = ActorCritic(obs_dim=50, act_dim=12).to(gs.device)
    ppo_sd = load_ppo_checkpoint(ckpt_path, device=gs.device)
    model.load_state_dict(ppo_sd, strict=False)
    model.eval()
    return model


# ---- Scene building ------------------------------------------------------ #

def build_physics_scene(gs, torch_mod, args, obstacle_layout, beacon_layout):
    cfg = RobotSimConfig()
    scene = gs.Scene(show_viewer=bool(args.show_viewer))
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=PHYSICS_URDF_PATH, pos=(0.0, 0.0, ROBOT_SPAWN_Z), fixed=False),
    )
    add_obstacles_to_scene(scene, obstacle_layout)
    for obs in beacon_layout.all_obstacles():
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )
    scene.build(n_envs=1)

    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[name].dofs_idx_local)[0] for name in JOINTS_ACTUATED]
    act_dofs = torch_mod.tensor(dof_idx, device=gs.device, dtype=torch_mod.int64)
    q0 = torch_mod.tensor(Q0_VALUES, device=gs.device, dtype=torch_mod.float32)

    robot.set_dofs_kp(torch_mod.ones(12, device=gs.device) * cfg.kp, act_dofs)
    robot.set_dofs_kv(torch_mod.ones(12, device=gs.device) * cfg.kv, act_dofs)
    robot.set_dofs_position(q0.unsqueeze(0), act_dofs)
    robot.set_dofs_velocity(torch_mod.zeros((1, 12), device=gs.device), act_dofs)

    return scene, robot, act_dofs, q0, cfg


def build_render_scene(gs, torch_mod, urdf_path, obstacle_layout, beacon_layout, img_res, fov, near):
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    add_obstacles_to_scene(scene, obstacle_layout)
    for obs in beacon_layout.all_obstacles():
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )
    robot = scene.add_entity(
        gs.morphs.URDF(file=urdf_path, fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(img_res, img_res), fov=fov, near=near, GUI=False)
    scene.build(n_envs=1)

    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[name].dofs_idx_local)[0] for name in JOINTS_ACTUATED]
    act_dofs = torch_mod.tensor(dof_idx, device=gs.device, dtype=torch_mod.int64)
    return scene, robot, cam, act_dofs


# ---- Robot helpers ------------------------------------------------------- #

def reset_robot(robot, act_dofs, q0, spawn_xy: np.ndarray, yaw_rad: float, gs, torch_mod) -> None:
    pos = torch_mod.tensor(
        [[float(spawn_xy[0]), float(spawn_xy[1]), ROBOT_SPAWN_Z]],
        device=gs.device, dtype=torch_mod.float32,
    )
    quat = torch_mod.tensor(yaw_to_quat(yaw_rad), device=gs.device, dtype=torch_mod.float32).unsqueeze(0)
    robot.set_pos(pos, zero_velocity=True)
    robot.set_quat(quat, zero_velocity=False)
    robot.set_dofs_position(q0.unsqueeze(0), act_dofs)
    robot.set_dofs_velocity(torch_mod.zeros((1, 12), device=gs.device), act_dofs)


def collect_proprio(robot, act_dofs, q0, prev_action: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    pos = robot.get_pos()
    quat = robot.get_quat()
    vel_b = world_to_body_vec(quat, robot.get_vel())
    ang_b = world_to_body_vec(quat, robot.get_ang())
    q = robot.get_dofs_position(act_dofs)
    dq = robot.get_dofs_velocity(act_dofs)
    q_rel = q - q0.unsqueeze(0)
    proprio = torch.cat([pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_action], dim=1)
    return proprio, to_numpy(pos[0]), to_numpy(quat[0])


def sync_render_robot(src_robot, src_act_dofs, dst_robot, dst_act_dofs) -> tuple[np.ndarray, np.ndarray]:
    pos = src_robot.get_pos()
    quat = src_robot.get_quat()
    q = src_robot.get_dofs_position(src_act_dofs)
    dst_robot.set_pos(pos)
    dst_robot.set_quat(quat)
    dst_robot.set_dofs_position(q, dst_act_dofs)
    return to_numpy(pos[0]), to_numpy(quat[0])


# ---- Rendering ----------------------------------------------------------- #

def render_egocentric_frame(
    physics_robot, physics_act_dofs,
    ego_robot, ego_act_dofs, ego_cam,
    obstacle_layout, camera_cfg: EgoCameraConfig,
    fallback_frame_hwc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Render an egocentric frame with camera safety retraction."""
    pos_np, quat_np = sync_render_robot(physics_robot, physics_act_dofs, ego_robot, ego_act_dofs)
    cam_pos, cam_lookat, cam_up, cam_forward = egocentric_camera_pose(pos_np, quat_np, camera_cfg)
    cam_rot = camera_rotation_matrix(quat_np, camera_cfg.pitch_rad)

    safety = camera_safety_metrics(cam_pos, cam_forward, obstacle_layout, camera_cfg, cam_rot=cam_rot)
    if bool(safety["unsafe"]):
        cam_pos, cam_lookat, cam_up, cam_forward, retract_dist = retract_camera_to_safe(
            cam_pos, cam_forward, cam_up, cam_rot, obstacle_layout, camera_cfg,
        )
        if retract_dist > 0.0:
            safety = camera_safety_metrics(cam_pos, cam_forward, obstacle_layout, camera_cfg, cam_rot=cam_rot)
        if bool(safety["unsafe"]):
            if fallback_frame_hwc is not None:
                return np.ascontiguousarray(fallback_frame_hwc.copy()), pos_np, quat_np, True
            raise RuntimeError(
                f"Camera unsafe after retraction with no fallback: "
                f"inside_wall={bool(safety['inside_wall'])}, clearance={float(safety['clearance']):.3f}"
            )

    ego_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
    render_out = ego_cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), pos_np, quat_np, False


def render_third_person_frame(
    physics_robot, physics_act_dofs,
    render_robot, render_act_dofs, cam,
    chase_dist: float, chase_height: float,
    side_offset: float, lookahead: float,
) -> np.ndarray:
    """Chase-cam third-person view with the full robot body visible."""
    pos_np, quat_np = sync_render_robot(physics_robot, physics_act_dofs, render_robot, render_act_dofs)

    w, x, y, z = [float(v) for v in quat_np]
    fw = np.array([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        0.0,
    ], dtype=np.float32)
    fw_norm = float(np.linalg.norm(fw[:2]))
    if fw_norm < 1e-6:
        fw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        fw /= fw_norm
    side = np.array([-fw[1], fw[0], 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    cam_pos = pos_np - chase_dist * fw + chase_height * up + side_offset * side
    cam_lookat = pos_np + lookahead * fw + 0.18 * up
    cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=up)

    render_out = cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))


def resize_frame(frame_hwc: np.ndarray, target_res: int) -> np.ndarray:
    if frame_hwc.shape[0] == target_res and frame_hwc.shape[1] == target_res:
        return frame_hwc
    return np.asarray(
        Image.fromarray(frame_hwc).resize((target_res, target_res), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def build_side_by_side_frame(first_person_hwc: np.ndarray, third_person_hwc: np.ndarray) -> np.ndarray:
    ego = resize_frame(first_person_hwc, third_person_hwc.shape[0])
    divider = np.full((third_person_hwc.shape[0], 8, 3), 12, dtype=np.uint8)
    return np.concatenate([ego, divider, third_person_hwc], axis=1)


# ---- Observation (perception-only, no oracle labels) --------------------- #

@torch.no_grad()
def observe(
    physics_robot, physics_act_dofs,
    ego_robot, ego_act_dofs, ego_cam,
    obstacle_layout, camera_cfg: EgoCameraConfig,
    world_model: LeWorldModel, planning_device: torch.device,
    q0, prev_action: torch.Tensor,
    fallback_frame_hwc: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render + encode. Returns a dict with frame, latents, proprioception, pose."""
    frame_hwc, pos_np, quat_np, frame_substituted = render_egocentric_frame(
        physics_robot, physics_act_dofs,
        ego_robot, ego_act_dofs, ego_cam,
        obstacle_layout, camera_cfg, fallback_frame_hwc,
    )
    proprio, _, _ = collect_proprio(physics_robot, physics_act_dofs, q0, prev_action)
    yaw_rad = float(quat_to_yaw(quat_np))

    frame_chw = np.ascontiguousarray(np.transpose(frame_hwc, (2, 0, 1)))
    vision = torch.from_numpy(frame_chw).unsqueeze(0).to(planning_device).float().div_(255.0)
    proprio_enc = proprio.to(planning_device) if world_model.encoder.use_proprio else None
    z_raw, z_proj = world_model.encode(vision, proprio_enc)

    return {
        "frame_hwc": frame_hwc,
        "frame_substituted": frame_substituted,
        "z_raw": z_raw.detach(),
        "z_proj": z_proj.detach(),
        "proprio": proprio.detach(),
        "pos_np": pos_np,
        "quat_np": quat_np,
        "yaw_rad": yaw_rad,
    }


# ---- Breadcrumb encoding (single front-face view) ----------------------- #

@torch.no_grad()
def encode_breadcrumb(
    world_model: LeWorldModel,
    render_scene, render_robot, render_act_dofs, cam,
    beacon, view_dist: float,
    planning_device: torch.device,
    q0, gs, torch_mod,
) -> torch.Tensor:
    """Render one front-face view of a beacon and encode it as a breadcrumb latent.

    This is the task specification — equivalent to showing a photo of the target
    to the robot before deployment. Not an oracle cheat.
    """
    # Park the robot out of view
    render_robot.set_pos(
        torch_mod.tensor([[999.0, 999.0, -10.0]], device=gs.device, dtype=torch_mod.float32),
        zero_velocity=True,
    )
    render_robot.set_quat(
        torch_mod.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=torch_mod.float32),
        zero_velocity=True,
    )
    render_robot.set_dofs_position(q0.unsqueeze(0), render_act_dofs)
    render_robot.set_dofs_velocity(torch_mod.zeros((1, 12), device=gs.device), render_act_dofs)
    render_scene.step()

    bx, by, bz = [float(v) for v in beacon.pos]
    nx, ny = [float(v) for v in beacon.normal]
    # Camera positioned directly in front of the beacon face
    angle = math.atan2(ny, nx) + math.pi
    cam_pos = np.array([
        bx + view_dist * math.cos(angle),
        by + view_dist * math.sin(angle),
        bz + 0.05,
    ], dtype=np.float32)
    cam.set_pose(
        pos=cam_pos,
        lookat=np.array([bx, by, bz], dtype=np.float32),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    render_scene.step()

    render_out = cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    rgb_chw = np.ascontiguousarray(np.transpose(rgb[:, :, :3], (2, 0, 1)))
    vis_t = torch.from_numpy(rgb_chw).unsqueeze(0).to(planning_device).float().div_(255.0)
    _, z_proj = world_model.encode(vis_t, None)
    return z_proj.squeeze(0).detach()


# ---- PPO execution ------------------------------------------------------- #

def execute_command(
    scene, robot, policy: ActorCritic,
    act_dofs, q0, prev_action: torch.Tensor,
    nominal_cmd: torch.Tensor,
    sim_cfg: RobotSimConfig,
    obs_noise_std: float, gs, torch_mod,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one planning step through the PPO low-level controller."""
    proprio, _, _ = collect_proprio(robot, act_dofs, q0, prev_action)
    ppo_proprio = proprio
    if obs_noise_std > 0.0:
        noise = torch_mod.randn_like(ppo_proprio) * obs_noise_std
        noise[:, 1:5] *= 2.0
        noise[:, 5:11] *= 5.0
        ppo_proprio = ppo_proprio + noise

    cmd = nominal_cmd.to(device=gs.device, dtype=torch_mod.float32).view(1, 3)
    obs_tensor = torch.cat([ppo_proprio, cmd], dim=1)
    actions = policy.act_deterministic(obs_tensor)

    q_tgt = q0.unsqueeze(0) + sim_cfg.action_scale * actions
    q_tgt[:, 0:4] = torch_mod.clamp(q_tgt[:, 0:4], -0.8, 0.8)
    q_tgt[:, 4:8] = torch_mod.clamp(q_tgt[:, 4:8], -1.5, 1.5)
    q_tgt[:, 8:12] = torch_mod.clamp(q_tgt[:, 8:12], -2.5, -0.5)

    robot.control_dofs_position(q_tgt, act_dofs)
    for _ in range(sim_cfg.decimation):
        scene.step()

    return cmd.detach().clone(), actions.detach().clone()


# ---- Visualization ------------------------------------------------------ #

def color255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in rgb)


def compute_plot_bounds(obstacle_layout, beacon_layout, path_xy: list[list[float]]) -> float:
    max_extent = 0.5
    for obs in obstacle_layout.obstacles:
        max_extent = max(
            max_extent,
            abs(float(obs.pos[0])) + 0.5 * float(obs.size[0]),
            abs(float(obs.pos[1])) + 0.5 * float(obs.size[1]),
        )
    for beacon in beacon_layout.beacons:
        max_extent = max(
            max_extent,
            abs(float(beacon.pos[0])) + 0.5 * float(beacon.size[0]),
            abs(float(beacon.pos[1])) + 0.5 * float(beacon.size[1]),
        )
    for xy in path_xy:
        max_extent = max(max_extent, abs(float(xy[0])), abs(float(xy[1])))
    return max_extent + 0.3


def world_to_canvas(x: float, y: float, half_extent: float, size: int) -> tuple[float, float]:
    scale = (size - 1) / (2.0 * half_extent)
    return (x + half_extent) * scale, (half_extent - y) * scale


def draw_topdown_trajectory(
    out_path: str,
    obstacle_layout, beacon_layout: BeaconLayout,
    path_xy: list[list[float]],
    breadcrumb_xy: list[float] | None,
) -> None:
    size = 900
    half_extent = compute_plot_bounds(obstacle_layout, beacon_layout, path_xy)
    img = Image.new("RGB", (size, size), color=(250, 248, 242))
    draw = ImageDraw.Draw(img)

    for obs in obstacle_layout.obstacles:
        hx = 0.5 * float(obs.size[0])
        hy = 0.5 * float(obs.size[1])
        x0, y0 = world_to_canvas(float(obs.pos[0]) - hx, float(obs.pos[1]) + hy, half_extent, size)
        x1, y1 = world_to_canvas(float(obs.pos[0]) + hx, float(obs.pos[1]) - hy, half_extent, size)
        draw.rectangle((x0, y0, x1, y1), fill=(88, 92, 101), outline=(58, 62, 70))

    for distractor in beacon_layout.distractors:
        hx = 0.5 * float(distractor.size[0])
        hy = 0.5 * float(distractor.size[1])
        x0, y0 = world_to_canvas(float(distractor.pos[0]) - hx, float(distractor.pos[1]) + hy, half_extent, size)
        x1, y1 = world_to_canvas(float(distractor.pos[0]) + hx, float(distractor.pos[1]) - hy, half_extent, size)
        draw.rectangle((x0, y0, x1, y1), fill=color255(distractor.color), outline=(40, 40, 40))

    for beacon in beacon_layout.beacons:
        hx = 0.5 * float(beacon.size[0])
        hy = 0.5 * float(beacon.size[1])
        x0, y0 = world_to_canvas(float(beacon.pos[0]) - hx, float(beacon.pos[1]) + hy, half_extent, size)
        x1, y1 = world_to_canvas(float(beacon.pos[0]) + hx, float(beacon.pos[1]) - hy, half_extent, size)
        draw.rectangle((x0, y0, x1, y1), fill=color255(beacon.color), outline=(0, 0, 0), width=2)

    if len(path_xy) >= 2:
        path_points = [world_to_canvas(float(x), float(y), half_extent, size) for x, y in path_xy]
        draw.line(path_points, fill=(30, 110, 210), width=5)

    if path_xy:
        sx, sy = world_to_canvas(float(path_xy[0][0]), float(path_xy[0][1]), half_extent, size)
        ex, ey = world_to_canvas(float(path_xy[-1][0]), float(path_xy[-1][1]), half_extent, size)
        r = 8
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(40, 170, 90), outline=(0, 0, 0))
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(220, 50, 60), outline=(0, 0, 0))

    if breadcrumb_xy is not None:
        bx, by = world_to_canvas(float(breadcrumb_xy[0]), float(breadcrumb_xy[1]), half_extent, size)
        r = 10
        draw.ellipse((bx - r, by - r, bx + r, by + r), fill=(255, 210, 40), outline=(0, 0, 0), width=2)

    img.save(out_path)


# ---- Video export -------------------------------------------------------- #

def encode_mp4(out_path: str, frames_hwc: list[np.ndarray], fps: int) -> None:
    if not frames_hwc:
        return
    height, width = frames_hwc[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames_hwc:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        ret = proc.wait()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}:\n{stderr}")


def resolve_video_formats(requested: str) -> list[str]:
    has_ffmpeg = shutil.which("ffmpeg") is not None
    requested = requested.strip().lower()
    if requested == "auto":
        return ["mp4"] if has_ffmpeg else ["gif"]
    if requested == "mp4":
        if not has_ffmpeg:
            raise RuntimeError("ffmpeg not found; use --video_format gif")
        return ["mp4"]
    if requested == "gif":
        return ["gif"]
    if requested == "both":
        if not has_ffmpeg:
            raise RuntimeError("ffmpeg not found; cannot use --video_format both")
        return ["mp4", "gif"]
    raise ValueError(f"Unsupported video format: {requested}")


def save_gif(out_path: str, frames_hwc: list[np.ndarray], stride: int, fps: int) -> None:
    keep = max(1, int(stride))
    frames = [Image.fromarray(frame) for frame in frames_hwc[::keep]]
    if not frames:
        return
    duration_ms = max(1, round(1000 * keep / max(1, fps)))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def export_video(out_stem: str, frames_hwc: list[np.ndarray], stride: int, fps: int, formats: list[str]) -> None:
    if not frames_hwc:
        return
    kept = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames_hwc[:: max(1, int(stride))]]
    if "mp4" in formats:
        encode_mp4(f"{out_stem}.mp4", kept, max(1, round(fps / max(1, int(stride)))))
    if "gif" in formats:
        save_gif(f"{out_stem}.gif", frames_hwc, stride, fps)


# ---- Helpers ------------------------------------------------------------- #

def beacon_name_to_id(identity: str) -> int:
    try:
        return BEACON_IDENTITY_NAMES.index(identity)
    except ValueError:
        return -1


def pretty_beacon(beacon_id: int) -> str:
    if 0 <= beacon_id < len(BEACON_IDENTITY_NAMES):
        return f"{beacon_id}:{BEACON_IDENTITY_NAMES[beacon_id]}"
    return "none"


# ---- Main ---------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("inference_runs", f"maze_seed_{args.seed:04d}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_name, ckpt_path in [("PPO", args.ppo_ckpt), ("WM", args.wm_ckpt), ("Scorer", args.scorer_ckpt)]:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"{ckpt_name} checkpoint not found: {ckpt_path}")

    planning_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1. Load models ----
    world_model, wm_meta = load_world_model(args.wm_ckpt, planning_device)
    scorer, scorer_meta = load_trajectory_scorer(args.scorer_ckpt, planning_device)
    camera_cfg = ego_camera_config_from_args(args)
    video_formats = resolve_video_formats(args.video_format)

    # ---- 2. Generate enclosed maze ----
    obstacle_layout, beacon_layout, start_cell = generate_enclosed_maze(
        seed=args.seed,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        cell_size=args.cell_size,
        wall_thickness=args.wall_thickness,
        n_beacons=args.n_beacons,
        n_distractors=args.n_distractors,
    )
    maze_step = args.cell_size + args.wall_thickness
    grid_ox = -(args.grid_cols - 1) * maze_step / 2.0
    grid_oy = -(args.grid_rows - 1) * maze_step / 2.0
    spawn_xy = np.array([
        grid_ox + start_cell[1] * maze_step,
        grid_oy + start_cell[0] * maze_step,
    ], dtype=np.float32)

    # Select target beacon (furthest from spawn by default)
    target_beacon = None
    if args.target_beacon is not None:
        for b in beacon_layout.beacons:
            if b.identity == args.target_beacon:
                target_beacon = b
                break
    if target_beacon is None and beacon_layout.beacons:
        # Pick the beacon furthest from spawn
        best_dist = -1.0
        for b in beacon_layout.beacons:
            bxy = np.array(b.pos[:2], dtype=np.float32)
            d = float(np.linalg.norm(bxy - spawn_xy))
            if d > best_dist:
                best_dist = d
                target_beacon = b

    # Choose spawn heading: face into the first open corridor, not toward the
    # beacon through walls.  Probe each cardinal direction from the start cell
    # and pick the one with the most clearance.
    spawn_yaw = 0.0
    best_clearance = -1.0
    from lewm.obstacle_utils import detect_collisions as _dc
    for probe_yaw, label in [(0.0, "+X"), (math.pi / 2, "+Y"), (math.pi, "-X"), (-math.pi / 2, "-Y")]:
        probe_xy = spawn_xy + 0.3 * np.array([math.cos(probe_yaw), math.sin(probe_yaw)], dtype=np.float32)
        probe_t = torch.from_numpy(probe_xy.reshape(1, 2))
        blocked = bool(_dc(probe_t, obstacle_layout, margin=0.12)[0].item())
        if not blocked:
            # Among open directions, prefer the one pointing toward the target
            dir_score = 1.0
            if target_beacon is not None:
                dx = float(target_beacon.pos[0]) - float(spawn_xy[0])
                dy = float(target_beacon.pos[1]) - float(spawn_xy[1])
                dir_score = math.cos(probe_yaw - math.atan2(dy, dx))
            if dir_score > best_clearance:
                best_clearance = dir_score
                spawn_yaw = probe_yaw

    target_beacon_id = beacon_name_to_id(target_beacon.identity) if target_beacon else None
    breadcrumb_xy = [float(target_beacon.pos[0]), float(target_beacon.pos[1])] if target_beacon else None

    print(
        f"Maze: {args.grid_rows}x{args.grid_cols} grid, seed={args.seed}, "
        f"obstacles={len(obstacle_layout.obstacles)}, beacons={len(beacon_layout.beacons)}"
    )
    print(
        f"Spawn: cell=({start_cell[0]},{start_cell[1]}) "
        f"pos=({spawn_xy[0]:+.2f}, {spawn_xy[1]:+.2f}) yaw={math.degrees(spawn_yaw):+.1f}deg"
    )
    if target_beacon:
        print(
            f"Target beacon: {target_beacon.identity} at "
            f"({target_beacon.pos[0]:.2f}, {target_beacon.pos[1]:.2f})"
        )

    # ---- 3. Build Genesis scenes ----
    cmd_low_t = torch.tensor(args.cmd_low, dtype=torch.float32, device=planning_device)
    cmd_high_t = torch.tensor(args.cmd_high, dtype=torch.float32, device=planning_device)

    ego_frames_hwc: list[np.ndarray] = []
    third_person_frames_hwc: list[np.ndarray] = []
    combined_frames_hwc: list[np.ndarray] = []
    path_xy: list[list[float]] = []
    plan_costs: list[float] = []
    nominal_cmds: list[list[float]] = []
    last_clean_ego_frame: np.ndarray | None = None
    ego_frame_substitutions = 0
    collision_count = 0
    terminate_reason = "max_steps"

    t0 = time.time()
    physics_scene = None
    ego_scene = None
    third_person_scene = None
    try:
        import genesis as gs

        init_genesis_once(args.sim_backend)
        policy = load_frozen_policy(args.ppo_ckpt, gs)

        print(f"Planning device: {planning_device}")
        print(f"Genesis device:  {gs.device}")
        print(
            f"World model: latent_dim={wm_meta['latent_dim']} image_size={wm_meta['image_size']} "
            f"use_proprio={wm_meta['use_proprio']}"
        )
        print(
            f"Scorer: goal_weight={scorer.goal_weight:.3f} "
            f"exploration_weight={scorer.exploration_weight:.3f} "
            f"forward_reward={args.forward_reward_weight:.3f}"
        )

        physics_scene, physics_robot, physics_act_dofs, q0, sim_cfg = build_physics_scene(
            gs, torch, args, obstacle_layout, beacon_layout,
        )
        ego_scene, ego_robot, ego_cam, ego_act_dofs = build_render_scene(
            gs=gs, torch_mod=torch,
            urdf_path=EGO_RENDER_URDF_PATH,
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            img_res=wm_meta["image_size"],
            fov=camera_cfg.fov_deg,
            near=camera_cfg.near_plane,
        )
        # Third-person scene: full robot URDF (body visible)
        third_person_scene, tp_robot, tp_cam, tp_act_dofs = build_render_scene(
            gs=gs, torch_mod=torch,
            urdf_path=THIRD_PERSON_URDF_PATH,
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            img_res=args.third_person_res,
            fov=args.third_person_fov,
            near=0.01,
        )

        # ---- 4. Encode breadcrumb ----
        z_breadcrumb: torch.Tensor | None = None
        if target_beacon is not None:
            z_breadcrumb = encode_breadcrumb(
                world_model=world_model,
                render_scene=ego_scene,
                render_robot=ego_robot,
                render_act_dofs=ego_act_dofs,
                cam=ego_cam,
                beacon=target_beacon,
                view_dist=args.breadcrumb_view_dist,
                planning_device=planning_device,
                q0=q0, gs=gs, torch_mod=torch,
            )
            print(f"Breadcrumb encoded: z_norm={float(z_breadcrumb.norm()):.3f}")

        # ---- 5. Spawn robot ----
        reset_robot(physics_robot, physics_act_dofs, q0, spawn_xy, spawn_yaw, gs, torch)

        planner = CEMPlanner(
            world_model=world_model,
            scorer=scorer,
            horizon=args.plan_horizon,
            n_candidates=args.n_candidates,
            cem_iters=args.cem_iters,
            elite_frac=args.elite_frac,
            cmd_low=cmd_low_t,
            cmd_high=cmd_high_t,
            init_std=torch.tensor(args.cem_init_std, dtype=torch.float32),
            min_std=torch.tensor(args.cem_min_std, dtype=torch.float32),
            forward_reward_weight=args.forward_reward_weight,
            device=planning_device,
        )

        prev_action = torch.zeros((1, 12), device=gs.device, dtype=torch.float32)
        last_nominal_cmd = torch.zeros(3, device=planning_device, dtype=torch.float32)

        # ---- 6. Initial observation ----
        obs = observe(
            physics_robot, physics_act_dofs,
            ego_robot, ego_act_dofs, ego_cam,
            obstacle_layout, camera_cfg,
            world_model, planning_device,
            q0, prev_action,
        )
        last_clean_ego_frame = obs["frame_hwc"].copy()
        ego_frames_hwc.append(obs["frame_hwc"])
        tp_frame = render_third_person_frame(
            physics_robot, physics_act_dofs,
            tp_robot, tp_act_dofs, tp_cam,
            args.chase_dist, args.chase_height, args.side_offset, args.lookahead,
        )
        third_person_frames_hwc.append(tp_frame)
        combined_frames_hwc.append(build_side_by_side_frame(obs["frame_hwc"], tp_frame))
        path_xy.append([float(obs["pos_np"][0]), float(obs["pos_np"][1])])

        # ---- 7. Main loop ----
        for step in range(args.steps):
            # Plan
            plan_seq, plan_stats = planner.plan(
                z_start_raw=obs["z_raw"],
                z_goal_proj=z_breadcrumb,
                last_cmd=last_nominal_cmd,
            )
            nominal_cmd = plan_seq[0]
            last_nominal_cmd = nominal_cmd.detach().clone()
            cmd_vals = [float(v) for v in nominal_cmd.detach().cpu().tolist()]

            # Execute
            active_cmd, actions = execute_command(
                scene=physics_scene,
                robot=physics_robot,
                policy=policy,
                act_dofs=physics_act_dofs,
                q0=q0,
                prev_action=prev_action,
                nominal_cmd=nominal_cmd.to(gs.device),
                sim_cfg=sim_cfg,
                obs_noise_std=args.ppo_obs_noise_std,
                gs=gs, torch_mod=torch,
            )
            prev_action = actions.detach().clone()

            # Observe
            obs = observe(
                physics_robot, physics_act_dofs,
                ego_robot, ego_act_dofs, ego_cam,
                obstacle_layout, camera_cfg,
                world_model, planning_device,
                q0, prev_action, last_clean_ego_frame,
            )
            if obs["frame_substituted"]:
                ego_frame_substitutions += 1
            else:
                last_clean_ego_frame = obs["frame_hwc"].copy()

            # Third-person render
            tp_frame = render_third_person_frame(
                physics_robot, physics_act_dofs,
                tp_robot, tp_act_dofs, tp_cam,
                args.chase_dist, args.chase_height, args.side_offset, args.lookahead,
            )

            # Log
            ego_frames_hwc.append(obs["frame_hwc"])
            third_person_frames_hwc.append(tp_frame)
            combined_frames_hwc.append(build_side_by_side_frame(obs["frame_hwc"], tp_frame))
            path_xy.append([float(obs["pos_np"][0]), float(obs["pos_np"][1])])
            nominal_cmds.append(cmd_vals)
            plan_costs.append(float(plan_stats.best_cost))

            # Collision / termination checks
            pos_xy = torch.from_numpy(np.asarray(obs["pos_np"][:2], dtype=np.float32)).unsqueeze(0)
            collided = bool(detect_collisions(pos_xy, obstacle_layout, margin=sim_cfg.collision_margin)[0].item())
            fallen = bool(obs["pos_np"][2] < sim_cfg.min_z)

            # Oracle success check (post-hoc metric, NOT used for planning)
            reached = False
            if target_beacon is not None:
                target_xy = np.array(target_beacon.pos[:2], dtype=np.float32)
                dist_to_target = float(np.linalg.norm(obs["pos_np"][:2] - target_xy))
                if dist_to_target <= args.success_range:
                    reached = True
                    terminate_reason = "goal_reached"
                    print(f"Step {step:03d} | REACHED beacon {target_beacon.identity} at dist={dist_to_target:.3f}m")
                    break

            print(
                f"Step {step:03d} | pos=({obs['pos_np'][0]:+.2f}, {obs['pos_np'][1]:+.2f}) "
                f"yaw={math.degrees(obs['yaw_rad']):+6.1f}deg "
                f"cmd=[{cmd_vals[0]:+.2f}, {cmd_vals[1]:+.2f}, {cmd_vals[2]:+.2f}] "
                f"cost={plan_stats.best_cost:.3f}"
            )

            if fallen:
                terminate_reason = "fallen"
                print(f"Step {step:03d} | terminating: robot fell")
                break
            if collided:
                collision_count += 1
                if args.terminate_on_collision:
                    terminate_reason = "collision"
                    print(f"Step {step:03d} | terminating: collision")
                    break
                planner.reset()
                last_nominal_cmd.zero_()

    finally:
        try:
            import genesis as gs
            if third_person_scene is not None:
                third_person_scene.destroy()
            if ego_scene is not None:
                ego_scene.destroy()
            if physics_scene is not None:
                physics_scene.destroy()
            if getattr(gs, "_initialized", False):
                gs.destroy()
        except Exception:
            pass

    elapsed = time.time() - t0

    # ---- 8. Export ----
    summary = {
        "seed": args.seed,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "n_obstacles": len(obstacle_layout.obstacles),
        "n_beacons": len(beacon_layout.beacons),
        "target_beacon": target_beacon.identity if target_beacon else None,
        "terminate_reason": terminate_reason,
        "elapsed_sec": elapsed,
        "steps_executed": len(nominal_cmds),
        "collision_count": collision_count,
        "ego_frame_substitutions": ego_frame_substitutions,
        "forward_reward_weight": args.forward_reward_weight,
        "spawn_xy": [float(spawn_xy[0]), float(spawn_xy[1])],
        "spawn_yaw_rad": float(spawn_yaw),
        "breadcrumb_xy": breadcrumb_xy,
        "path_xy": path_xy,
        "nominal_cmds": nominal_cmds,
        "plan_costs": plan_costs,
        "beacons": [
            {"identity": b.identity, "pos": [float(v) for v in b.pos], "normal": [float(v) for v in b.normal]}
            for b in beacon_layout.beacons
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_gif and ego_frames_hwc:
        export_video(str(out_dir / "ego_rollout"), ego_frames_hwc, args.gif_stride, args.video_fps, video_formats)
    if not args.no_gif and third_person_frames_hwc:
        export_video(str(out_dir / "third_person_rollout"), third_person_frames_hwc, args.gif_stride, args.video_fps, video_formats)
    if not args.no_gif and combined_frames_hwc:
        export_video(str(out_dir / "side_by_side_rollout"), combined_frames_hwc, args.gif_stride, args.video_fps, video_formats)

    if not args.no_topdown and path_xy:
        draw_topdown_trajectory(
            out_path=str(out_dir / "trajectory_topdown.png"),
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            path_xy=path_xy,
            breadcrumb_xy=breadcrumb_xy,
        )

    print(f"Finished in {elapsed:.1f}s | {terminate_reason} | collisions={collision_count}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
