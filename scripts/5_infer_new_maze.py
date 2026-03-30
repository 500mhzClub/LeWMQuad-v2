#!/usr/bin/env python3
"""Run latent-space planning on a brand-new maze with CEM.

This script:
  1. Generates a novel maze with beacon panels.
  2. Builds a Genesis physics scene plus two render scenes:
     - egocentric render scene with the base shell hidden, matching training.
     - third-person render scene with the full robot visible.
  3. Loads the frozen PPO low-level controller, LeWorldModel encoder/predictor,
     and the trained TrajectoryScorer heads.
  4. Replans every step with Cross-Entropy Method over 3D velocity commands
     ``[vx, vy, yaw_rate]``.
  5. Captures a breadcrumb latent the first time a beacon enters the camera FOV.
  6. Saves egocentric, third-person, side-by-side GIFs, and a top-down trajectory plot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    add_egocentric_camera_args,
    camera_rotation_matrix,
    camera_safety_metrics,
    ego_camera_config_from_args,
    egocentric_camera_pose,
    retract_camera_to_safe,
)
from lewm.checkpoint_utils import clean_state_dict, load_ppo_checkpoint
from lewm.genesis_utils import init_genesis_once, to_numpy
from lewm.label_utils import compute_beacon_labels
from lewm.math_utils import quat_to_yaw, world_to_body_vec, yaw_to_quat
from lewm.maze_utils import MAZE_STYLES, generate_composite_scene
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
MAZE_STYLE_CHOICES = ["random", *MAZE_STYLES]


@dataclass(frozen=True)
class RobotSimConfig:
    kp: float = 5.0
    kv: float = 0.5
    action_scale: float = 0.30
    decimation: int = 4
    collision_margin: float = 0.15
    safe_clearance: float = 0.40
    min_z: float = 0.04


@dataclass
class RuntimeState:
    prev_action: torch.Tensor
    latency_buffer: torch.Tensor


@dataclass
class PlanningStats:
    best_cost: float
    mean_cost: float
    std_cost: float
    elite_cost: float


@dataclass
class ObservationBundle:
    frame_hwc: np.ndarray
    z_raw: torch.Tensor
    z_proj: torch.Tensor
    proprio: torch.Tensor
    pos_np: np.ndarray
    quat_np: np.ndarray
    yaw_rad: float
    beacon_visible: bool
    beacon_identity: int
    beacon_range: float
    beacon_bearing: float


class CEMPlanner:
    """CEM over command sequences in latent space."""

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
            samples[0] = mean

            z_rollouts = self.world_model.plan_rollout(z0_batch, samples)
            costs = self.scorer.score(z_rollouts, z_goal_batch)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CEM planning on a new maze.")
    parser.add_argument("--ppo_ckpt", type=str, required=True, help="Frozen PPO checkpoint path.")
    parser.add_argument("--wm_ckpt", type=str, required=True, help="Frozen LeWorldModel checkpoint path.")
    parser.add_argument("--scorer_ckpt", type=str, required=True, help="TrajectoryScorer checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Planning device.")
    parser.add_argument("--sim_backend", type=str, default="auto", help="Genesis backend.")
    parser.add_argument("--seed", type=int, default=0, help="Scene and planner RNG seed.")
    parser.add_argument(
        "--maze_style",
        type=str,
        default="random",
        choices=MAZE_STYLE_CHOICES,
        help="Maze style or 'random'.",
    )
    parser.add_argument("--n_beacons", type=int, default=1, help="Number of maze beacons.")
    parser.add_argument("--n_distractors", type=int, default=0, help="Number of distractor patches.")
    parser.add_argument("--n_free_obstacles", type=int, default=0, help="Additional free obstacles.")
    parser.add_argument("--arena_half", type=float, default=3.0, help="Arena half-extent.")
    parser.add_argument("--steps", type=int, default=120, help="Maximum executed planning steps.")
    parser.add_argument("--success_range", type=float, default=0.40, help="Stop when target beacon is this close.")
    parser.add_argument("--show_viewer", action="store_true", help="Open the Genesis viewer.")
    parser.add_argument("--ppo_obs_noise_std", type=float, default=0.0, help="Optional noise on PPO proprio.")
    parser.add_argument("--plan_horizon", type=int, default=12, help="Number of commands per CEM rollout.")
    parser.add_argument("--n_candidates", type=int, default=512, help="CEM candidate sequences per iteration.")
    parser.add_argument("--cem_iters", type=int, default=4, help="CEM refinement iterations.")
    parser.add_argument("--elite_frac", type=float, default=0.125, help="Elite fraction for CEM.")
    parser.add_argument(
        "--cmd_low",
        type=float,
        nargs=3,
        default=(-0.40, -0.25, -1.20),
        metavar=("VX", "VY", "WZ"),
        help="Lower command bounds.",
    )
    parser.add_argument(
        "--cmd_high",
        type=float,
        nargs=3,
        default=(0.60, 0.25, 1.20),
        metavar=("VX", "VY", "WZ"),
        help="Upper command bounds.",
    )
    parser.add_argument(
        "--cem_init_std",
        type=float,
        nargs=3,
        default=(0.25, 0.12, 0.50),
        metavar=("VX", "VY", "WZ"),
        help="Initial CEM sampling std.",
    )
    parser.add_argument(
        "--cem_min_std",
        type=float,
        nargs=3,
        default=(0.05, 0.04, 0.10),
        metavar=("VX", "VY", "WZ"),
        help="Minimum per-dimension CEM std.",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for visuals and logs.")
    parser.add_argument("--gif_stride", type=int, default=1, help="Keep every Nth frame in the GIF.")
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF exports.")
    parser.add_argument("--no_topdown", action="store_true", help="Skip top-down trajectory export.")
    parser.add_argument("--third_person_res", type=int, default=384, help="Third-person render resolution.")
    parser.add_argument("--third_person_fov", type=float, default=60.0, help="Third-person camera field of view.")
    parser.add_argument("--chase_dist", type=float, default=1.0, help="Third-person chase distance.")
    parser.add_argument("--chase_height", type=float, default=0.55, help="Third-person camera height.")
    parser.add_argument("--side_offset", type=float, default=0.25, help="Third-person lateral offset.")
    parser.add_argument("--lookahead", type=float, default=0.20, help="Third-person look-ahead distance.")
    add_egocentric_camera_args(parser)
    return parser.parse_args()


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
    if grid * grid != n_tokens:
        raise ValueError(f"Cannot infer image size from {n_tokens} patches")
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


def sample_spawn_xy(
    obstacle_layout,
    safe_clearance: float,
    rng: np.random.RandomState,
    attempts: int = 2048,
    span: float = 1.5,
) -> np.ndarray:
    origin = torch.zeros((1, 2), dtype=torch.float32)
    if not bool(detect_collisions(origin, obstacle_layout, margin=safe_clearance)[0].item()):
        return np.zeros(2, dtype=np.float32)

    candidates = rng.uniform(-span, span, size=(attempts, 2)).astype(np.float32)
    cand_t = torch.from_numpy(candidates)
    safe_mask = ~detect_collisions(cand_t, obstacle_layout, margin=safe_clearance)
    safe_ids = torch.nonzero(safe_mask).squeeze(-1).cpu().numpy()
    if safe_ids.size == 0:
        return np.zeros(2, dtype=np.float32)

    dists = np.linalg.norm(candidates[safe_ids], axis=1)
    return candidates[safe_ids[int(np.argmin(dists))]]


def spawn_heading(spawn_xy: np.ndarray, obstacle_layout, beacon_layout: BeaconLayout) -> float:
    if beacon_layout.beacons:
        target = np.array(beacon_layout.beacons[0].pos[:2], dtype=np.float32)
    elif obstacle_layout.obstacles:
        target = np.mean(
            np.array([obs.pos[:2] for obs in obstacle_layout.obstacles], dtype=np.float32),
            axis=0,
        )
    else:
        target = np.array([1.0, 0.0], dtype=np.float32)
    delta = target - np.asarray(spawn_xy, dtype=np.float32)
    return float(math.atan2(float(delta[1]), float(delta[0])))


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
    missing = [name for name in JOINTS_ACTUATED if name not in name_to_joint]
    if missing:
        raise RuntimeError(f"Missing joints in URDF: {missing}")

    dof_idx = [list(name_to_joint[name].dofs_idx_local)[0] for name in JOINTS_ACTUATED]
    act_dofs = torch_mod.tensor(dof_idx, device=gs.device, dtype=torch_mod.int64)
    q0 = torch_mod.tensor(Q0_VALUES, device=gs.device, dtype=torch_mod.float32)

    robot.set_dofs_kp(torch_mod.ones(12, device=gs.device) * cfg.kp, act_dofs)
    robot.set_dofs_kv(torch_mod.ones(12, device=gs.device) * cfg.kv, act_dofs)
    robot.set_dofs_position(q0.unsqueeze(0), act_dofs)
    robot.set_dofs_velocity(torch_mod.zeros((1, 12), device=gs.device), act_dofs)

    return scene, robot, act_dofs, q0, cfg


def build_render_scene(
    gs,
    torch_mod,
    urdf_path: str,
    obstacle_layout,
    beacon_layout,
    img_res: int,
    fov: float,
    near: float,
):
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
    missing = [name for name in JOINTS_ACTUATED if name not in name_to_joint]
    if missing:
        raise RuntimeError(f"Missing joints in render URDF {urdf_path}: {missing}")

    dof_idx = [list(name_to_joint[name].dofs_idx_local)[0] for name in JOINTS_ACTUATED]
    act_dofs = torch_mod.tensor(dof_idx, device=gs.device, dtype=torch_mod.int64)
    return scene, robot, cam, act_dofs


def reset_robot(robot, act_dofs, q0, spawn_xy: np.ndarray, yaw_rad: float, gs, torch_mod) -> None:
    pos = torch_mod.tensor(
        [[float(spawn_xy[0]), float(spawn_xy[1]), ROBOT_SPAWN_Z]],
        device=gs.device,
        dtype=torch_mod.float32,
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


def sync_render_robot(
    src_robot,
    src_act_dofs,
    dst_robot,
    dst_act_dofs,
) -> tuple[np.ndarray, np.ndarray]:
    pos = src_robot.get_pos()
    quat = src_robot.get_quat()
    q = src_robot.get_dofs_position(src_act_dofs)
    dst_robot.set_pos(pos)
    dst_robot.set_quat(quat)
    dst_robot.set_dofs_position(q, dst_act_dofs)
    return to_numpy(pos[0]), to_numpy(quat[0])


def render_egocentric_frame(
    physics_robot,
    physics_act_dofs,
    ego_robot,
    ego_act_dofs,
    ego_cam,
    obstacle_layout,
    camera_cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            raise RuntimeError(
                "Camera remained unsafe after retraction. "
                f"clearance={float(safety['clearance']):.3f}, "
                f"frustum_min_hit={float(safety['frustum_min_hit']):.3f}"
            )

    ego_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
    render_out = ego_cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.asarray(rgb, dtype=np.uint8), pos_np, quat_np


def render_third_person_frame(
    physics_robot,
    physics_act_dofs,
    render_robot,
    render_act_dofs,
    cam,
    chase_dist: float,
    chase_height: float,
    side_offset: float,
    lookahead: float,
) -> np.ndarray:
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
    return np.asarray(rgb, dtype=np.uint8)


@torch.no_grad()
def observe(
    physics_robot,
    physics_act_dofs,
    ego_robot,
    ego_act_dofs,
    ego_cam,
    obstacle_layout,
    beacon_layout: BeaconLayout,
    camera_cfg,
    world_model: LeWorldModel,
    planning_device: torch.device,
    q0,
    prev_action: torch.Tensor,
) -> ObservationBundle:
    frame_hwc, pos_np, quat_np = render_egocentric_frame(
        physics_robot=physics_robot,
        physics_act_dofs=physics_act_dofs,
        ego_robot=ego_robot,
        ego_act_dofs=ego_act_dofs,
        ego_cam=ego_cam,
        obstacle_layout=obstacle_layout,
        camera_cfg=camera_cfg,
    )
    proprio, _pos_np_dup, _quat_np_dup = collect_proprio(physics_robot, physics_act_dofs, q0, prev_action)
    yaw_rad = float(quat_to_yaw(quat_np))

    beacon_labels = compute_beacon_labels(
        robot_xy=np.asarray([pos_np[:2]], dtype=np.float32),
        robot_yaw=np.asarray([yaw_rad], dtype=np.float32),
        beacon_layout=beacon_layout,
        fov_deg=camera_cfg.fov_deg,
    )

    frame_chw = np.transpose(frame_hwc, (2, 0, 1))
    vision = torch.from_numpy(frame_chw).unsqueeze(0).to(planning_device).float().div_(255.0)
    proprio_enc = None
    if world_model.encoder.use_proprio:
        proprio_enc = proprio.to(planning_device)

    z_raw, z_proj = world_model.encode(vision, proprio_enc)

    return ObservationBundle(
        frame_hwc=frame_hwc,
        z_raw=z_raw.detach(),
        z_proj=z_proj.detach(),
        proprio=proprio.detach(),
        pos_np=pos_np,
        quat_np=quat_np,
        yaw_rad=yaw_rad,
        beacon_visible=bool(beacon_labels["beacon_visible"][0]),
        beacon_identity=int(beacon_labels["beacon_identity"][0]),
        beacon_range=float(beacon_labels["beacon_range"][0]),
        beacon_bearing=float(beacon_labels["beacon_bearing"][0]),
    )


def execute_nominal_command(
    scene,
    robot,
    policy: ActorCritic,
    act_dofs,
    q0,
    runtime: RuntimeState,
    nominal_cmd: torch.Tensor,
    sim_cfg: RobotSimConfig,
    obs_noise_std: float,
    gs,
    torch_mod,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime.latency_buffer = torch.roll(runtime.latency_buffer, shifts=-1, dims=0)
    runtime.latency_buffer[-1] = nominal_cmd.to(device=gs.device, dtype=torch_mod.float32).view(1, 3)
    active_cmd = runtime.latency_buffer[0]

    proprio, _pos_np, _quat_np = collect_proprio(robot, act_dofs, q0, runtime.prev_action)
    ppo_proprio = proprio
    if obs_noise_std > 0.0:
        noise = torch_mod.randn_like(ppo_proprio) * obs_noise_std
        noise[:, 1:5] *= 2.0
        noise[:, 5:11] *= 5.0
        ppo_proprio = ppo_proprio + noise

    obs = torch.cat([ppo_proprio, active_cmd], dim=1)
    actions = policy.act_deterministic(obs)
    runtime.prev_action = actions.detach().clone()

    q_tgt = q0.unsqueeze(0) + sim_cfg.action_scale * actions
    q_tgt[:, 0:4] = torch_mod.clamp(q_tgt[:, 0:4], -0.8, 0.8)
    q_tgt[:, 4:8] = torch_mod.clamp(q_tgt[:, 4:8], -1.5, 1.5)
    q_tgt[:, 8:12] = torch_mod.clamp(q_tgt[:, 8:12], -2.5, -0.5)

    robot.control_dofs_position(q_tgt, act_dofs)
    for _ in range(sim_cfg.decimation):
        scene.step()

    return active_cmd.detach().clone(), actions.detach().clone()


def color255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in rgb)


def compute_plot_bounds(obstacle_layout, beacon_layout, path_xy: list[list[float]], arena_half: float) -> float:
    max_extent = float(arena_half)
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
    px = (x + half_extent) * scale
    py = (half_extent - y) * scale
    return float(px), float(py)


def draw_topdown_trajectory(
    out_path: str,
    obstacle_layout,
    beacon_layout: BeaconLayout,
    path_xy: list[list[float]],
    breadcrumb_xy: list[float] | None,
    arena_half: float,
) -> None:
    size = 900
    half_extent = compute_plot_bounds(obstacle_layout, beacon_layout, path_xy, arena_half)
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


def save_gif(out_path: str, frames_hwc: list[np.ndarray], stride: int) -> None:
    keep = max(1, int(stride))
    frames = [Image.fromarray(frame) for frame in frames_hwc[::keep]]
    if not frames:
        return
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )


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


def pretty_beacon(beacon_id: int) -> str:
    if beacon_id < 0:
        return "none"
    names = list(BEACON_FAMILIES.keys())
    if beacon_id >= len(names):
        return str(beacon_id)
    return f"{beacon_id}:{names[beacon_id]}"


def main() -> None:
    args = parse_args()
    if args.plan_horizon < 1:
        raise ValueError("--plan_horizon must be >= 1")
    if args.n_candidates < 1:
        raise ValueError("--n_candidates must be >= 1")
    if args.cem_iters < 1:
        raise ValueError("--cem_iters must be >= 1")
    if not (0.0 < args.elite_frac <= 1.0):
        raise ValueError("--elite_frac must be in (0, 1]")
    if args.out_dir is None:
        args.out_dir = os.path.join("inference_runs", f"maze_seed_{args.seed:04d}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.ppo_ckpt):
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_ckpt}")
    if not os.path.isfile(args.wm_ckpt):
        raise FileNotFoundError(f"World-model checkpoint not found: {args.wm_ckpt}")
    if not os.path.isfile(args.scorer_ckpt):
        raise FileNotFoundError(f"TrajectoryScorer checkpoint not found: {args.scorer_ckpt}")

    planning_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    world_model, wm_meta = load_world_model(args.wm_ckpt, planning_device)
    scorer, scorer_meta = load_trajectory_scorer(args.scorer_ckpt, planning_device)
    if int(scorer_meta.get("latent_dim", wm_meta["latent_dim"])) != int(wm_meta["latent_dim"]):
        raise RuntimeError("Latent-dimension mismatch between world model and trajectory scorer")

    camera_cfg = ego_camera_config_from_args(args)
    maze_style = None if args.maze_style == "random" else args.maze_style
    obstacle_layout, beacon_layout = generate_composite_scene(
        seed=args.seed,
        maze_style=maze_style,
        n_free_obstacles=args.n_free_obstacles,
        n_beacons=args.n_beacons,
        n_distractors=args.n_distractors,
        arena_half=args.arena_half,
    )

    planner = CEMPlanner(
        world_model=world_model,
        scorer=scorer,
        horizon=args.plan_horizon,
        n_candidates=args.n_candidates,
        cem_iters=args.cem_iters,
        elite_frac=args.elite_frac,
        cmd_low=torch.tensor(args.cmd_low, dtype=torch.float32),
        cmd_high=torch.tensor(args.cmd_high, dtype=torch.float32),
        init_std=torch.tensor(args.cem_init_std, dtype=torch.float32),
        min_std=torch.tensor(args.cem_min_std, dtype=torch.float32),
        device=planning_device,
    )

    ego_frames_hwc: list[np.ndarray] = []
    third_person_frames_hwc: list[np.ndarray] = []
    combined_frames_hwc: list[np.ndarray] = []
    path_xy: list[list[float]] = []
    nominal_cmds: list[list[float]] = []
    active_cmds: list[list[float]] = []
    plan_costs: list[float] = []
    breadcrumb_xy: list[float] | None = None
    breadcrumb_step: int | None = None
    target_beacon_id: int | None = None
    z_breadcrumb: torch.Tensor | None = None
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
            f"patch={wm_meta['patch_size']} seq={wm_meta['max_seq_len']} use_proprio={wm_meta['use_proprio']}"
        )
        print(
            f"Scorer weights: goal={scorer.goal_weight:.3f} exploration={scorer.exploration_weight:.3f}"
        )
        print(
            f"Scene: maze_style={args.maze_style} obstacles={len(obstacle_layout.obstacles)} "
            f"beacons={len(beacon_layout.beacons)} distractors={len(beacon_layout.distractors)}"
        )
        print(
            f"Egocentric render URDF: {EGO_RENDER_URDF_PATH} "
            f"(matches training: base shell hidden)"
        )
        print(f"Third-person render URDF: {THIRD_PERSON_URDF_PATH}")

        physics_scene, physics_robot, physics_act_dofs, q0, sim_cfg = build_physics_scene(
            gs, torch, args, obstacle_layout, beacon_layout,
        )
        ego_scene, ego_robot, ego_cam, ego_act_dofs = build_render_scene(
            gs=gs,
            torch_mod=torch,
            urdf_path=EGO_RENDER_URDF_PATH,
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            img_res=wm_meta["image_size"],
            fov=camera_cfg.fov_deg,
            near=camera_cfg.near_plane,
        )
        third_person_scene, third_person_robot, third_person_cam, third_person_act_dofs = build_render_scene(
            gs=gs,
            torch_mod=torch,
            urdf_path=THIRD_PERSON_URDF_PATH,
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            img_res=args.third_person_res,
            fov=args.third_person_fov,
            near=0.01,
        )
        spawn_xy = sample_spawn_xy(obstacle_layout, sim_cfg.safe_clearance, rng)
        spawn_yaw = spawn_heading(spawn_xy, obstacle_layout, beacon_layout)
        reset_robot(physics_robot, physics_act_dofs, q0, spawn_xy, spawn_yaw, gs, torch)

        runtime = RuntimeState(
            prev_action=torch.zeros((1, 12), device=gs.device, dtype=torch.float32),
            latency_buffer=torch.zeros((2, 1, 3), device=gs.device, dtype=torch.float32),
        )

        obs = observe(
            physics_robot=physics_robot,
            physics_act_dofs=physics_act_dofs,
            ego_robot=ego_robot,
            ego_act_dofs=ego_act_dofs,
            ego_cam=ego_cam,
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            camera_cfg=camera_cfg,
            world_model=world_model,
            planning_device=planning_device,
            q0=q0,
            prev_action=runtime.prev_action,
        )
        third_person_frame = render_third_person_frame(
            physics_robot=physics_robot,
            physics_act_dofs=physics_act_dofs,
            render_robot=third_person_robot,
            render_act_dofs=third_person_act_dofs,
            cam=third_person_cam,
            chase_dist=args.chase_dist,
            chase_height=args.chase_height,
            side_offset=args.side_offset,
            lookahead=args.lookahead,
        )
        ego_frames_hwc.append(obs.frame_hwc)
        third_person_frames_hwc.append(third_person_frame)
        combined_frames_hwc.append(build_side_by_side_frame(obs.frame_hwc, third_person_frame))
        path_xy.append([float(obs.pos_np[0]), float(obs.pos_np[1])])

        last_nominal_cmd = torch.zeros(3, device=planning_device, dtype=torch.float32)

        for step in range(args.steps):
            if z_breadcrumb is None and obs.beacon_visible:
                z_breadcrumb = obs.z_proj.detach().clone()
                target_beacon_id = obs.beacon_identity
                breadcrumb_xy = [float(obs.pos_np[0]), float(obs.pos_np[1])]
                breadcrumb_step = step
                print(
                    f"Step {step:03d} | captured breadcrumb for beacon {pretty_beacon(target_beacon_id)} "
                    f"at range={obs.beacon_range:.3f}m"
                )

            if (
                target_beacon_id is not None
                and obs.beacon_visible
                and obs.beacon_identity == target_beacon_id
                and obs.beacon_range <= args.success_range
            ):
                terminate_reason = "goal_reached"
                print(
                    f"Step {step:03d} | reached beacon {pretty_beacon(target_beacon_id)} "
                    f"at range={obs.beacon_range:.3f}m"
                )
                break

            plan_seq, plan_stats = planner.plan(
                z_start_raw=obs.z_raw,
                z_goal_proj=z_breadcrumb,
                last_cmd=last_nominal_cmd,
            )
            nominal_cmd = plan_seq[0]
            last_nominal_cmd = nominal_cmd.detach().clone()
            nominal_cmd_vals = [float(v) for v in nominal_cmd.detach().cpu().tolist()]

            active_cmd, _actions = execute_nominal_command(
                scene=physics_scene,
                robot=physics_robot,
                policy=policy,
                act_dofs=physics_act_dofs,
                q0=q0,
                runtime=runtime,
                nominal_cmd=nominal_cmd.to(gs.device),
                sim_cfg=sim_cfg,
                obs_noise_std=args.ppo_obs_noise_std,
                gs=gs,
                torch_mod=torch,
            )

            obs = observe(
                physics_robot=physics_robot,
                physics_act_dofs=physics_act_dofs,
                ego_robot=ego_robot,
                ego_act_dofs=ego_act_dofs,
                ego_cam=ego_cam,
                obstacle_layout=obstacle_layout,
                beacon_layout=beacon_layout,
                camera_cfg=camera_cfg,
                world_model=world_model,
                planning_device=planning_device,
                q0=q0,
                prev_action=runtime.prev_action,
            )
            third_person_frame = render_third_person_frame(
                physics_robot=physics_robot,
                physics_act_dofs=physics_act_dofs,
                render_robot=third_person_robot,
                render_act_dofs=third_person_act_dofs,
                cam=third_person_cam,
                chase_dist=args.chase_dist,
                chase_height=args.chase_height,
                side_offset=args.side_offset,
                lookahead=args.lookahead,
            )

            pos_xy = torch.tensor([obs.pos_np[:2]], dtype=torch.float32)
            collided = bool(detect_collisions(pos_xy, obstacle_layout, margin=sim_cfg.collision_margin)[0].item())
            fallen = bool(obs.pos_np[2] < sim_cfg.min_z)

            ego_frames_hwc.append(obs.frame_hwc)
            third_person_frames_hwc.append(third_person_frame)
            combined_frames_hwc.append(build_side_by_side_frame(obs.frame_hwc, third_person_frame))
            path_xy.append([float(obs.pos_np[0]), float(obs.pos_np[1])])
            nominal_cmds.append(nominal_cmd_vals)
            active_cmds.append([float(v) for v in active_cmd[0].detach().cpu().tolist()])
            plan_costs.append(float(plan_stats.best_cost))

            print(
                f"Step {step:03d} | pos=({obs.pos_np[0]:+.2f}, {obs.pos_np[1]:+.2f}) "
                f"yaw={math.degrees(obs.yaw_rad):+6.1f}deg "
                f"cmd=[{nominal_cmd_vals[0]:+.2f}, {nominal_cmd_vals[1]:+.2f}, {nominal_cmd_vals[2]:+.2f}] "
                f"cost={plan_stats.best_cost:.3f} "
                f"beacon={pretty_beacon(obs.beacon_identity)} "
                f"range={'inf' if not obs.beacon_visible else f'{obs.beacon_range:.2f}'} "
                f"crumb={'yes' if z_breadcrumb is not None else 'no'}"
            )

            if fallen:
                terminate_reason = "fallen"
                print(f"Step {step:03d} | terminating after fall")
                break
            if collided:
                terminate_reason = "collision"
                print(f"Step {step:03d} | terminating after collision")
                break
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

    summary = {
        "seed": int(args.seed),
        "maze_style": args.maze_style,
        "n_obstacles": int(len(obstacle_layout.obstacles)),
        "n_beacons": int(len(beacon_layout.beacons)),
        "n_distractors": int(len(beacon_layout.distractors)),
        "terminate_reason": terminate_reason,
        "elapsed_sec": elapsed,
        "steps_executed": len(nominal_cmds),
        "breadcrumb_step": breadcrumb_step,
        "target_beacon_id": target_beacon_id,
        "target_beacon_name": pretty_beacon(target_beacon_id if target_beacon_id is not None else -1),
        "ego_render_urdf": EGO_RENDER_URDF_PATH,
        "third_person_render_urdf": THIRD_PERSON_URDF_PATH,
        "path_xy": path_xy,
        "nominal_cmds": nominal_cmds,
        "active_cmds": active_cmds,
        "plan_costs": plan_costs,
        "beacons": [
            {
                "identity": beacon.identity,
                "pos": [float(v) for v in beacon.pos],
                "normal": [float(v) for v in beacon.normal],
            }
            for beacon in beacon_layout.beacons
        ],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_gif and ego_frames_hwc:
        save_gif(str(out_dir / "ego_rollout.gif"), ego_frames_hwc, args.gif_stride)
    if not args.no_gif and third_person_frames_hwc:
        save_gif(str(out_dir / "third_person_rollout.gif"), third_person_frames_hwc, args.gif_stride)
    if not args.no_gif and combined_frames_hwc:
        save_gif(str(out_dir / "side_by_side_rollout.gif"), combined_frames_hwc, args.gif_stride)

    if not args.no_topdown and path_xy:
        draw_topdown_trajectory(
            out_path=str(out_dir / "trajectory_topdown.png"),
            obstacle_layout=obstacle_layout,
            beacon_layout=beacon_layout,
            path_xy=path_xy,
            breadcrumb_xy=breadcrumb_xy,
            arena_half=args.arena_half,
        )

    print(f"Finished in {elapsed:.1f}s")
    print(f"Outputs: {out_dir}")
    print(f"Termination: {terminate_reason}")


if __name__ == "__main__":
    main()
