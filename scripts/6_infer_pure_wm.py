#!/usr/bin/env python3
"""Pure world-model maze inference — Aligned Latents and Smoothness Prior.

Implements the planning approach from Section 3.2 of the LeWM paper, utilizing
strictly aligned z_raw latents for planning, memory, and goal tracking.
Exploration is driven by Sequence Novelty, balanced by an L2 Action Penalty
to encourage smooth, deliberate quadrupedal movement without safety heuristics.

python3 scripts/6_infer_pure_wm.py \
    --ppo_ckpt models/ppo/ckpt_20000.pt \
    --wm_ckpt lewm_checkpoints/epoch_20.pt \
    --seed 42 --steps 4000 
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
from typing import Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
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
from lewm.models import ActorCritic, LeWorldModel
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
THIRD_PERSON_URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
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


# ---- Pure CEM planner (Aligned Latents + Action Smoothness) -------------- #

class PureCEMPlanner:
    """CEM planner operating strictly in z_raw space with an L2 action prior."""

    def __init__(
        self,
        world_model: LeWorldModel,
        horizon: int,
        n_candidates: int,
        cem_iters: int,
        elite_frac: float,
        cmd_low: torch.Tensor,
        cmd_high: torch.Tensor,
        init_std: torch.Tensor,
        min_std: torch.Tensor,
        device: torch.device,
        novelty_weight: float = 10.0,
        action_penalty_weight: float = 0.001,
    ):
        self.world_model = world_model
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.cem_iters = int(cem_iters)
        self.n_elite = max(1, int(round(self.n_candidates * elite_frac)))
        self.cmd_low = cmd_low.to(device=device, dtype=torch.float32)
        self.cmd_high = cmd_high.to(device=device, dtype=torch.float32)
        self.init_std = init_std.to(device=device, dtype=torch.float32)
        self.min_std = min_std.to(device=device, dtype=torch.float32)
        self.device = device
        self.novelty_weight = novelty_weight
        self.action_penalty_weight = action_penalty_weight
        self._warm_start: torch.Tensor | None = None

    def reset(self) -> None:
        self._warm_start = None

    @torch.no_grad()
    def plan(
        self,
        z_start_raw: torch.Tensor,
        z_goal_raw: torch.Tensor | None = None,
        history_windows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        z0 = z_start_raw.to(self.device, dtype=torch.float32)
        z0_batch = z0.expand(self.n_candidates, -1)
        
        z_goal_batch = None
        if z_goal_raw is not None:
            if z_goal_raw.ndim == 1:
                z_goal_raw = z_goal_raw.unsqueeze(0)
            z_goal_batch = z_goal_raw.to(self.device, dtype=torch.float32).expand(
                self.n_candidates, -1,
            )

        if self._warm_start is not None:
            mean = self._warm_start.clone()
        else:
            mean = 0.5 * (self.cmd_low + self.cmd_high)
            mean = mean.unsqueeze(0).expand(self.horizon, -1).clone()
        std = self.init_std.unsqueeze(0).expand(self.horizon, -1).clone()

        best_seq = mean.clone()
        best_cost = float("inf")

        for _ in range(self.cem_iters):
            noise = torch.randn(
                self.n_candidates, self.horizon, 3, device=self.device,
            )
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            samples = samples.clamp(
                self.cmd_low.view(1, 1, 3),
                self.cmd_high.view(1, 1, 3),
            )
            samples[0] = mean

            # z_rollouts are raw latents predicted by the world model
            z_rollouts = self.world_model.plan_rollout(z0_batch, samples)
            costs = torch.zeros(self.n_candidates, device=self.device)

            # 1. Terminal Goal Cost (Aligned z_raw space)
            if z_goal_batch is not None:
                z_terminal = z_rollouts[:, -1, :]
                cos_sim = F.cosine_similarity(z_terminal, z_goal_batch, dim=-1)
                costs += (1.0 - cos_sim)

            # 2. Sequence Novelty (Aligned z_raw space)
            if history_windows is not None and self.novelty_weight > 0.0:
                B, H, D = z_rollouts.shape
                M = history_windows.shape[0]
                if M > 0:
                    z_flat = z_rollouts.reshape(B, H * D)
                    h_flat = history_windows.reshape(M, H * D)
                    
                    z_norm = F.normalize(z_flat, p=2, dim=-1)
                    h_norm = F.normalize(h_flat, p=2, dim=-1)
                    
                    sim_matrix = torch.mm(z_norm, h_norm.transpose(0, 1))
                    max_sim = sim_matrix.max(dim=-1).values
                    
                    seq_novelty_dist = 1.0 - max_sim
                    costs -= self.novelty_weight * seq_novelty_dist

            # 3. Action Smoothness Penalty (L2)
            if self.action_penalty_weight > 0.0:
                act_penalty = samples.square().sum(dim=(1, 2))
                costs += self.action_penalty_weight * act_penalty

            min_cost, min_idx = costs.min(dim=0)
            if min_cost.item() < best_cost:
                best_cost = min_cost.item()
                best_seq = samples[min_idx.item()].detach().clone()

            elite_idx = torch.topk(costs, k=self.n_elite, largest=False).indices
            elite = samples[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(self.min_std)

        self._warm_start = torch.cat(
            [best_seq[1:], best_seq[-1:].clone()], dim=0,
        ).detach()

        return best_seq, best_cost


# ---- Argument parsing ---------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure world-model maze inference (Aligned z_raw latents).",
    )
    p.add_argument("--ppo_ckpt", type=str, required=True)
    p.add_argument("--wm_ckpt", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid_rows", type=int, default=4)
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--cell_size", type=float, default=0.70)
    p.add_argument("--wall_thickness", type=float, default=0.20)
    p.add_argument("--n_beacons", type=int, default=2)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--target_beacon", type=str, default=None)
    
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sim_backend", type=str, default="auto")
    p.add_argument("--show_viewer", action="store_true")
    
    p.add_argument("--plan_horizon", type=int, default=5)
    p.add_argument("--n_candidates", type=int, default=300)
    p.add_argument("--cem_iters", type=int, default=30)
    p.add_argument("--elite_frac", type=float, default=0.10)
    p.add_argument("--mpc_execute", type=int, default=1)
    p.add_argument("--cmd_low", type=float, nargs=3, default=[-0.4, -0.3, -1.0])
    p.add_argument("--cmd_high", type=float, nargs=3, default=[0.8, 0.3, 1.0])
    p.add_argument("--cem_init_std", type=float, nargs=3, default=[0.3, 0.15, 0.4])
    p.add_argument("--cem_min_std", type=float, nargs=3, default=[0.05, 0.03, 0.08])
    
    p.add_argument("--novelty_weight", type=float, default=10.0)
    p.add_argument("--action_penalty_weight", type=float, default=0.001)
    p.add_argument("--history_len", type=int, default=100)
    p.add_argument("--success_range", type=float, default=0.4)
    
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--video_format", type=str, default="auto", choices=["auto", "mp4", "gif", "both"])
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--gif_stride", type=int, default=2)
    
    add_egocentric_camera_args(p)
    p.add_argument("--third_person_res", type=int, default=480)
    p.add_argument("--third_person_fov", type=float, default=60.0)
    p.add_argument("--chase_dist", type=float, default=0.6)
    p.add_argument("--chase_height", type=float, default=0.45)
    p.add_argument("--side_offset", type=float, default=0.15)
    p.add_argument("--lookahead", type=float, default=0.3)
    p.add_argument("--breadcrumb_view_dist", type=float, default=0.5)

    return p.parse_args()


# ---- Model loading ------------------------------------------------------- #

def clean_load_state(module: torch.nn.Module, sd: dict, *, strict: bool = True):
    missing, unexpected = module.load_state_dict(clean_state_dict(sd), strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: {missing=}, {unexpected=}")

def load_world_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = clean_state_dict(ckpt["model_state_dict"])
    pos_embed = state["encoder.vis_enc.pos_embed"]
    patch_w = state["encoder.vis_enc.patch_embed.weight"]
    pred_pos = state["predictor.pos_embed"]
    latent_dim = int(pos_embed.shape[-1])
    patch_size = int(patch_w.shape[-1])
    n_tokens = int(pos_embed.shape[1] - 1)
    grid = int(round(math.sqrt(n_tokens)))
    image_size = grid * patch_size
    max_seq_len = int(pred_pos.shape[1])
    use_proprio = any(k.startswith("encoder.prop_enc.") for k in state)

    model = LeWorldModel(
        latent_dim=latent_dim,
        image_size=image_size,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        use_proprio=use_proprio,
    ).to(device)
    clean_load_state(model, state)
    model.eval()
    return model, {
        "latent_dim": latent_dim, "image_size": image_size,
        "patch_size": patch_size, "use_proprio": use_proprio,
    }

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

def build_render_scene(gs, torch_mod, urdf_path, obstacle_layout, beacon_layout,
                       img_res, fov, near):
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

def reset_robot(robot, act_dofs, q0, spawn_xy, yaw_rad, gs, torch_mod):
    pos = torch_mod.tensor(
        [[float(spawn_xy[0]), float(spawn_xy[1]), ROBOT_SPAWN_Z]],
        device=gs.device, dtype=torch_mod.float32,
    )
    quat = torch_mod.tensor(
        yaw_to_quat(yaw_rad), device=gs.device, dtype=torch_mod.float32,
    ).unsqueeze(0)
    robot.set_pos(pos, zero_velocity=True)
    robot.set_quat(quat, zero_velocity=False)
    robot.set_dofs_position(q0.unsqueeze(0), act_dofs)
    robot.set_dofs_velocity(torch_mod.zeros((1, 12), device=gs.device), act_dofs)

def collect_proprio(robot, act_dofs, q0, prev_action):
    pos = robot.get_pos()
    quat = robot.get_quat()
    vel_b = world_to_body_vec(quat, robot.get_vel())
    ang_b = world_to_body_vec(quat, robot.get_ang())
    q = robot.get_dofs_position(act_dofs)
    dq = robot.get_dofs_velocity(act_dofs)
    q_rel = q - q0.unsqueeze(0)
    proprio = torch.cat([pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_action], dim=1)
    return proprio, to_numpy(pos[0]), to_numpy(quat[0])

def sync_render_robot(src_robot, src_act_dofs, dst_robot, dst_act_dofs):
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
    pos_np, quat_np = sync_render_robot(
        physics_robot, physics_act_dofs, ego_robot, ego_act_dofs,
    )
    cam_pos, cam_lookat, cam_up, cam_forward = egocentric_camera_pose(
        pos_np, quat_np, camera_cfg,
    )
    cam_rot = camera_rotation_matrix(quat_np, camera_cfg.pitch_rad)

    safety = camera_safety_metrics(
        cam_pos, cam_forward, obstacle_layout, camera_cfg, cam_rot=cam_rot,
    )
    if bool(safety["unsafe"]):
        cam_pos, cam_lookat, cam_up, cam_forward, _ = retract_camera_to_safe(
            cam_pos, cam_forward, cam_up, cam_rot, obstacle_layout, camera_cfg,
        )
        safety = camera_safety_metrics(
            cam_pos, cam_forward, obstacle_layout, camera_cfg, cam_rot=cam_rot,
        )
        if bool(safety["unsafe"]) and fallback_frame_hwc is not None:
            return fallback_frame_hwc.copy(), pos_np, quat_np, True

    ego_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
    render_out = ego_cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), pos_np, quat_np, False

def render_third_person_frame(
    physics_robot, physics_act_dofs,
    render_robot, render_act_dofs, cam,
    chase_dist, chase_height, side_offset, lookahead,
) -> np.ndarray:
    pos_np, quat_np = sync_render_robot(
        physics_robot, physics_act_dofs, render_robot, render_act_dofs,
    )
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


# ---- Observation (perception-only) --------------------------------------- #

@torch.no_grad()
def observe(
    physics_robot, physics_act_dofs,
    ego_robot, ego_act_dofs, ego_cam,
    obstacle_layout, camera_cfg,
    world_model, planning_device,
    q0, prev_action,
    fallback_frame_hwc=None,
) -> dict[str, Any]:
    frame_hwc, pos_np, quat_np, substituted = render_egocentric_frame(
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
        "frame_substituted": substituted,
        "z_raw": z_raw.detach(),
        "z_proj": z_proj.detach(),
        "proprio": proprio.detach(),
        "pos_np": pos_np,
        "quat_np": quat_np,
        "yaw_rad": yaw_rad,
    }


# ---- Breadcrumb encoding ------------------------------------------------- #

@torch.no_grad()
def encode_breadcrumb(
    world_model, render_scene, render_robot, render_act_dofs, cam,
    beacon, view_dist, planning_device, q0, gs, torch_mod,
) -> torch.Tensor:
    """Return the RAW latent z_raw of the goal view for accurate planning."""
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
    
    z_raw, _ = world_model.encode(vis_t, None)
    return z_raw.squeeze(0).detach()


# ---- PPO execution ------------------------------------------------------- #

def execute_command(
    scene, robot, policy, act_dofs, q0, prev_action,
    nominal_cmd, sim_cfg, gs, torch_mod,
) -> tuple[torch.Tensor, torch.Tensor]:
    proprio, _, _ = collect_proprio(robot, act_dofs, q0, prev_action)
    cmd = nominal_cmd.to(device=gs.device, dtype=torch_mod.float32).view(1, 3)
    obs_tensor = torch.cat([proprio, cmd], dim=1)
    actions = policy.act_deterministic(obs_tensor)

    q_tgt = q0.unsqueeze(0) + sim_cfg.action_scale * actions
    q_tgt[:, 0:4] = torch_mod.clamp(q_tgt[:, 0:4], -0.8, 0.8)
    q_tgt[:, 4:8] = torch_mod.clamp(q_tgt[:, 4:8], -1.5, 1.5)
    q_tgt[:, 8:12] = torch_mod.clamp(q_tgt[:, 8:12], -2.5, -0.5)

    robot.control_dofs_position(q_tgt, act_dofs)
    for _ in range(sim_cfg.decimation):
        scene.step()

    return cmd.detach().clone(), actions.detach().clone()


# ---- Visualization ------------------------------------------------------- #

def build_side_by_side_frame(fp_hwc, tp_hwc):
    from PIL import Image as PILImage
    target_h = tp_hwc.shape[0]
    if fp_hwc.shape[0] != target_h:
        fp_hwc = np.asarray(
            PILImage.fromarray(fp_hwc).resize(
                (target_h, target_h), PILImage.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    divider = np.full((target_h, 8, 3), 12, dtype=np.uint8)
    return np.concatenate([fp_hwc, divider, tp_hwc], axis=1)

def encode_mp4(out_path, frames_hwc, fps):
    if not frames_hwc:
        return
    h, w = frames_hwc[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_hwc:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()

def export_video(stem, frames, stride, fps):
    if not frames:
        return
    kept = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames[::max(1, stride)]]
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        encode_mp4(f"{stem}.mp4", kept, max(1, round(fps / max(1, stride))))
    else:
        pil_frames = [Image.fromarray(f) for f in kept]
        duration = max(1, round(1000 * stride / max(1, fps)))
        pil_frames[0].save(
            f"{stem}.gif", save_all=True,
            append_images=pil_frames[1:], duration=duration, loop=0,
        )

def draw_topdown_trajectory(out_path, obstacle_layout, beacon_layout, path_xy, breadcrumb_xy):
    size = 900
    max_ext = 0.5
    for obs in obstacle_layout.obstacles:
        max_ext = max(max_ext,
                      abs(float(obs.pos[0])) + 0.5 * float(obs.size[0]),
                      abs(float(obs.pos[1])) + 0.5 * float(obs.size[1]))
    for xy in path_xy:
        max_ext = max(max_ext, abs(float(xy[0])), abs(float(xy[1])))
    half = max_ext + 0.3

    def w2c(x, y):
        s = (size - 1) / (2.0 * half)
        return (x + half) * s, (half - y) * s

    img = Image.new("RGB", (size, size), color=(250, 248, 242))
    draw = ImageDraw.Draw(img)

    for obs in obstacle_layout.obstacles:
        hx, hy = 0.5 * float(obs.size[0]), 0.5 * float(obs.size[1])
        x0, y0 = w2c(float(obs.pos[0]) - hx, float(obs.pos[1]) + hy)
        x1, y1 = w2c(float(obs.pos[0]) + hx, float(obs.pos[1]) - hy)
        draw.rectangle((x0, y0, x1, y1), fill=(88, 92, 101), outline=(58, 62, 70))

    for b in beacon_layout.beacons:
        hx, hy = 0.5 * float(b.size[0]), 0.5 * float(b.size[1])
        x0, y0 = w2c(float(b.pos[0]) - hx, float(b.pos[1]) + hy)
        x1, y1 = w2c(float(b.pos[0]) + hx, float(b.pos[1]) - hy)
        c = tuple(int(max(0, min(1, v)) * 255) for v in b.color)
        draw.rectangle((x0, y0, x1, y1), fill=c, outline=(0, 0, 0), width=2)

    if len(path_xy) >= 2:
        pts = [w2c(float(x), float(y)) for x, y in path_xy]
        draw.line(pts, fill=(30, 110, 210), width=5)

    if path_xy:
        sx, sy = w2c(float(path_xy[0][0]), float(path_xy[0][1]))
        ex, ey = w2c(float(path_xy[-1][0]), float(path_xy[-1][1]))
        r = 8
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(40, 170, 90), outline=(0, 0, 0))
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(220, 50, 60), outline=(0, 0, 0))

    if breadcrumb_xy:
        bx, by = w2c(float(breadcrumb_xy[0]), float(breadcrumb_xy[1]))
        r = 10
        draw.ellipse((bx - r, by - r, bx + r, by + r), fill=(255, 210, 40), outline=(0, 0, 0), width=2)

    img.save(out_path)

def beacon_claim_xy(beacon, wall_thickness, stand_off=0.03):
    normal = np.array(beacon.normal[:2], dtype=np.float32)
    wall_center_xy = np.array(beacon.pos[:2], dtype=np.float32) - 0.035 * normal
    return wall_center_xy + normal * (0.5 * float(wall_thickness) + stand_off)

def has_line_of_sight(a_xy, b_xy, obstacle_layout, step_size=0.03, margin=0.05):
    diff = b_xy - a_xy
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return True
    direction = diff / dist
    n_steps = max(1, int(dist / step_size))
    for i in range(1, n_steps + 1):
        t = min(float(i * step_size), dist)
        pt = torch.from_numpy(
            (a_xy + direction * t).reshape(1, 2).astype(np.float32),
        )
        if detect_collisions(pt, obstacle_layout, margin=margin)[0].item():
            return False
    return True


# ---- Main ---------------------------------------------------------------- #

def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("inference_runs", f"pure_wm_seed_{args.seed:04d}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planning_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    world_model, wm_meta = load_world_model(args.wm_ckpt, planning_device)
    camera_cfg = ego_camera_config_from_args(args)

    print(f"Loaded world model: latent_dim={wm_meta['latent_dim']} "
          f"image_size={wm_meta['image_size']}")

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

    target_beacon = None
    if args.target_beacon is not None:
        for b in beacon_layout.beacons:
            if b.identity == args.target_beacon:
                target_beacon = b
                break
    if target_beacon is None and beacon_layout.beacons:
        best_dist = -1.0
        for b in beacon_layout.beacons:
            d = float(np.linalg.norm(np.array(b.pos[:2]) - spawn_xy))
            if d > best_dist:
                best_dist = d
                target_beacon = b

    spawn_yaw = 0.0
    best_score = -float("inf")
    for probe_yaw in [0.0, math.pi / 2, math.pi, -math.pi / 2]:
        probe_xy = spawn_xy + 0.3 * np.array(
            [math.cos(probe_yaw), math.sin(probe_yaw)], dtype=np.float32,
        )
        blocked = bool(detect_collisions(
            torch.from_numpy(probe_xy.reshape(1, 2)),
            obstacle_layout, margin=0.12,
        )[0].item())
        if not blocked:
            score = 1.0
            if target_beacon is not None:
                dx = float(target_beacon.pos[0]) - float(spawn_xy[0])
                dy = float(target_beacon.pos[1]) - float(spawn_xy[1])
                score = math.cos(probe_yaw - math.atan2(dy, dx))
            if score > best_score:
                best_score = score
                spawn_yaw = probe_yaw

    target_claim_xy = beacon_claim_xy(target_beacon, args.wall_thickness) if target_beacon else None
    breadcrumb_xy = [float(target_beacon.pos[0]), float(target_beacon.pos[1])] if target_beacon else None

    print(f"Maze: {args.grid_rows}x{args.grid_cols}, spawn=({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f})")
    if target_beacon:
        print(f"Target: {target_beacon.identity} at ({target_beacon.pos[0]:.2f}, {target_beacon.pos[1]:.2f})")
    print(f"Planning: H={args.plan_horizon}, N={args.n_candidates}, "
          f"iters={args.cem_iters}, K={args.mpc_execute}, "
          f"Novelty={args.novelty_weight}, ActionPen={args.action_penalty_weight}, "
          f"HistLen={args.history_len}")

    t0 = time.time()
    physics_scene = ego_scene = third_person_scene = None

    ego_frames_hwc: List[np.ndarray] = []
    tp_frames_hwc: List[np.ndarray] = []
    combined_frames: List[np.ndarray] = []
    path_xy: List[List[float]] = []
    costs_log: List[float] = []
    cmds_log: List[List[float]] = []
    latent_history: List[torch.Tensor] = []
    terminate_reason = "max_steps"
    collision_count = 0

    try:
        import genesis as gs

        init_genesis_once(args.sim_backend)
        policy = load_frozen_policy(args.ppo_ckpt, gs)

        physics_scene, physics_robot, physics_act_dofs, q0, sim_cfg = build_physics_scene(
            gs, torch, args, obstacle_layout, beacon_layout,
        )
        ego_scene, ego_robot, ego_cam, ego_act_dofs = build_render_scene(
            gs, torch, EGO_RENDER_URDF_PATH, obstacle_layout, beacon_layout,
            wm_meta["image_size"], camera_cfg.fov_deg, camera_cfg.near_plane,
        )
        third_person_scene, tp_robot, tp_cam, tp_act_dofs = build_render_scene(
            gs, torch, THIRD_PERSON_URDF_PATH, obstacle_layout, beacon_layout,
            args.third_person_res, args.third_person_fov, 0.01,
        )

        z_breadcrumb = None
        if target_beacon is not None:
            z_breadcrumb = encode_breadcrumb(
                world_model, ego_scene, ego_robot, ego_act_dofs, ego_cam,
                target_beacon, args.breadcrumb_view_dist,
                planning_device, q0, gs, torch,
            )
            print(f"Goal latent encoded (z_raw): ||z_g||={float(z_breadcrumb.norm()):.3f}")

        reset_robot(physics_robot, physics_act_dofs, q0, spawn_xy, spawn_yaw, gs, torch)

        planner = PureCEMPlanner(
            world_model=world_model,
            horizon=args.plan_horizon,
            n_candidates=args.n_candidates,
            cem_iters=args.cem_iters,
            elite_frac=args.elite_frac,
            cmd_low=torch.tensor(args.cmd_low, dtype=torch.float32),
            cmd_high=torch.tensor(args.cmd_high, dtype=torch.float32),
            init_std=torch.tensor(args.cem_init_std, dtype=torch.float32),
            min_std=torch.tensor(args.cem_min_std, dtype=torch.float32),
            device=planning_device,
            novelty_weight=args.novelty_weight,
            action_penalty_weight=args.action_penalty_weight,
        )

        prev_action = torch.zeros((1, 12), device=gs.device, dtype=torch.float32)
        last_clean_frame: np.ndarray | None = None
        plan_seq: torch.Tensor | None = None
        plan_step_idx = 0 

        obs = observe(
            physics_robot, physics_act_dofs,
            ego_robot, ego_act_dofs, ego_cam,
            obstacle_layout, camera_cfg,
            world_model, planning_device,
            q0, prev_action,
        )
        last_clean_frame = obs["frame_hwc"].copy()
        ego_frames_hwc.append(obs["frame_hwc"])
        tp_frame = render_third_person_frame(
            physics_robot, physics_act_dofs,
            tp_robot, tp_act_dofs, tp_cam,
            args.chase_dist, args.chase_height, args.side_offset, args.lookahead,
        )
        tp_frames_hwc.append(tp_frame)
        combined_frames.append(build_side_by_side_frame(obs["frame_hwc"], tp_frame))
        path_xy.append([float(obs["pos_np"][0]), float(obs["pos_np"][1])])
        
        latent_history.append(obs["z_raw"].squeeze(0).detach())

        for step in range(args.steps):
            need_replan = (plan_seq is None or plan_step_idx >= args.mpc_execute)

            if need_replan:
                history_windows = None
                if args.novelty_weight > 0.0 and len(latent_history) >= args.plan_horizon:
                    H = args.plan_horizon
                    windows = []
                    for i in range(len(latent_history) - H + 1):
                        windows.append(torch.stack(latent_history[i : i + H]))
                    history_windows = torch.stack(windows).to(planning_device)

                plan_seq, cost = planner.plan(obs["z_raw"], z_breadcrumb, history_windows)
                plan_step_idx = 0
                costs_log.append(cost)

            nominal_cmd = plan_seq[plan_step_idx]
            plan_step_idx += 1
            cmd_vals = [float(v) for v in nominal_cmd.cpu().tolist()]
            cmds_log.append(cmd_vals)

            _, actions = execute_command(
                physics_scene, physics_robot, policy,
                physics_act_dofs, q0, prev_action,
                nominal_cmd.to(gs.device), sim_cfg, gs, torch,
            )
            prev_action = actions.detach().clone()

            obs = observe(
                physics_robot, physics_act_dofs,
                ego_robot, ego_act_dofs, ego_cam,
                obstacle_layout, camera_cfg,
                world_model, planning_device,
                q0, prev_action, last_clean_frame,
            )
            if not obs["frame_substituted"]:
                last_clean_frame = obs["frame_hwc"].copy()
                latent_history.append(obs["z_raw"].squeeze(0).detach())
                if len(latent_history) > args.history_len:
                    latent_history.pop(0)

            tp_frame = render_third_person_frame(
                physics_robot, physics_act_dofs,
                tp_robot, tp_act_dofs, tp_cam,
                args.chase_dist, args.chase_height, args.side_offset, args.lookahead,
            )

            ego_frames_hwc.append(obs["frame_hwc"])
            tp_frames_hwc.append(tp_frame)
            combined_frames.append(build_side_by_side_frame(obs["frame_hwc"], tp_frame))
            cur_xy = [float(obs["pos_np"][0]), float(obs["pos_np"][1])]
            path_xy.append(cur_xy)

            pos_xy_t = torch.from_numpy(
                np.asarray(obs["pos_np"][:2], dtype=np.float32),
            ).unsqueeze(0)
            collided = bool(detect_collisions(
                pos_xy_t, obstacle_layout, margin=sim_cfg.collision_margin,
            )[0].item())
            
            if collided:
                collision_count += 1
                planner.reset()
                plan_seq = None

            if obs["pos_np"][2] < sim_cfg.min_z:
                terminate_reason = "fallen"
                print(f"Step {step:03d} | fallen")
                break

            reached = False
            if target_beacon is not None and target_claim_xy is not None:
                dist_claim = float(np.linalg.norm(
                    np.asarray(cur_xy, dtype=np.float32) - target_claim_xy,
                ))
                frontness = float(np.dot(
                    np.asarray(cur_xy, dtype=np.float32) - np.array(target_beacon.pos[:2], dtype=np.float32),
                    np.array(target_beacon.normal[:2], dtype=np.float32),
                ))
                los = has_line_of_sight(
                    np.asarray(cur_xy, dtype=np.float32),
                    target_claim_xy, obstacle_layout,
                    step_size=0.02, margin=0.01,
                )
                if dist_claim <= args.success_range and los and frontness > 0:
                    reached = True

                if step % 20 == 0 or reached:
                    print(
                        f"Step {step:03d} | pos=({cur_xy[0]:+.2f}, {cur_xy[1]:+.2f}) "
                        f"cmd=[{cmd_vals[0]:+.2f}, {cmd_vals[1]:+.2f}, {cmd_vals[2]:+.2f}] "
                        f"cost={costs_log[-1]:.3f} d_goal={dist_claim:.2f}m "
                        f"coll={collision_count}"
                    )

            if reached:
                terminate_reason = "goal_reached"
                print(f"Step {step:03d} | REACHED {target_beacon.identity}")
                break

    finally:
        try:
            import genesis as gs
            for s in [third_person_scene, ego_scene, physics_scene]:
                if s is not None:
                    s.destroy()
            if getattr(gs, "_initialized", False):
                gs.destroy()
        except Exception:
            pass

    elapsed = time.time() - t0

    summary = {
        "approach": "pure_world_model_terminal_cost",
        "paper_section": "3.2",
        "seed": args.seed,
        "grid": f"{args.grid_rows}x{args.grid_cols}",
        "target": target_beacon.identity if target_beacon else None,
        "result": terminate_reason,
        "steps": len(cmds_log),
        "collisions": collision_count,
        "elapsed_sec": elapsed,
        "planner": {
            "horizon": args.plan_horizon,
            "candidates": args.n_candidates,
            "cem_iters": args.cem_iters,
            "elite_frac": args.elite_frac,
            "mpc_execute_k": args.mpc_execute,
            "novelty_weight": args.novelty_weight,
            "action_penalty_weight": args.action_penalty_weight,
            "history_len": args.history_len,
        },
        "path_xy": path_xy,
        "commands": cmds_log,
        "costs": costs_log,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if ego_frames_hwc:
        export_video(str(out_dir / "ego"), ego_frames_hwc, args.gif_stride, args.video_fps)
    if tp_frames_hwc:
        export_video(str(out_dir / "third_person"), tp_frames_hwc, args.gif_stride, args.video_fps)
    if combined_frames:
        export_video(str(out_dir / "side_by_side"), combined_frames, args.gif_stride, args.video_fps)
    if path_xy:
        draw_topdown_trajectory(
            str(out_dir / "trajectory.png"),
            obstacle_layout, beacon_layout, path_xy, breadcrumb_xy,
        )

    print("=" * 60)
    print(f"RESULT: {terminate_reason} | steps={len(cmds_log)} | "
          f"collisions={collision_count} | time={elapsed:.1f}s")
    print(f"Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
