#!/usr/bin/env python3
"""Pure-perception maze inference with a minimal learned planner.

The active control policy is intentionally narrow:
  - Explore with learned safety + learned novelty.
  - Treat beacon similarity as an evaluation signal only, not a planner cost.
  - Keep simulator geometry for offline metrics only; never use it to steer.

This file still contains some legacy helpers from earlier routing experiments,
but the main loop no longer uses prototype banks, keyframe graphs, dead-reckoned
route following, learned progress shaping, or goal-directed mode switching.

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
from lewm.models import (
    ActorCritic,
    ExplorationBonus,
    GoalEnergyHead,
    LatentEnergyHead,
    LeWorldModel,
    PlaceSnippetHead,
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


@dataclass(frozen=True)
class PrototypeBankUpdate:
    n_seen: int
    action: str
    nearest_sim: float | None = None
    filled_now: bool = False
    replace_idx: int | None = None
    prototype_idx: int | None = None


@dataclass
class KeyframeNode:
    idx: int
    score_latent: torch.Tensor
    proj_latent: torch.Tensor
    step: int
    last_seen_step: int
    visit_count: int
    odom_xy: tuple[float, float]
    yaw_rad: float


@dataclass
class PlannerHeads:
    safety_head: LatentEnergyHead | None = None
    goal_head: GoalEnergyHead | None = None
    exploration: ExplorationBonus | None = None
    place_head: PlaceSnippetHead | None = None
    goal_weight: float = 0.0
    exploration_weight: float = 0.0
    safety_weight: float = 1.0


def summarize_command_sequence(seq: torch.Tensor) -> dict[str, list[Any]]:
    seq_cpu = seq.detach().to(dtype=torch.float32, device="cpu")
    return {
        "command_sequence": seq_cpu.tolist(),
        "command_first": seq_cpu[0].tolist(),
        "command_last": seq_cpu[-1].tolist(),
        "command_mean": seq_cpu.mean(dim=0).tolist(),
    }


# ---- Pure CEM planner ---------------------------------------------------- #

class PureCEMPlanner:
    """CEM planner over learned safety and novelty."""

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
        if self.cmd_low.ndim != 1:
            raise ValueError(f"cmd_low must be 1D, got shape {tuple(self.cmd_low.shape)}")
        self.cmd_dim = int(self.cmd_low.numel())
        for name, tensor in (
            ("cmd_high", self.cmd_high),
            ("init_std", self.init_std),
            ("min_std", self.min_std),
        ):
            if tensor.shape != self.cmd_low.shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match cmd_low {tuple(self.cmd_low.shape)}"
                )
        self.device = device
        self.action_penalty_weight = action_penalty_weight
        self._warm_start: torch.Tensor | None = None

    def reset(self) -> None:
        self._warm_start = None

    @torch.no_grad()
    def plan(
        self,
        z_start_raw: torch.Tensor,
        z_start_proj: torch.Tensor | None = None,
        heads: PlannerHeads | None = None,
        exploration_bonus_mode: str = "terminal",
        terminal_displacement_weight: float = 0.0,
        exploration_safety_gate_threshold: float | None = None,
        exploration_safety_gate_sharpness: float = 5.0,
        visited_rollout_bank: list[torch.Tensor] | None = None,
        visited_rollout_knn_k: int = 8,
        visited_rollout_margin: float = 0.0,
        visited_rollout_tail_steps: int = 1,
        visited_revisit_penalty_weight: float = 0.0,
        return_diagnostics: bool = False,
        diagnostics_topk: int = 5,
    ) -> tuple[torch.Tensor, float, dict[str, float], dict[str, Any] | None]:
        z0 = z_start_raw.to(self.device, dtype=torch.float32)
        if z0.ndim != 2 or z0.shape[0] != 1:
            raise ValueError(f"Expected z_start_raw shape (1, D), got {tuple(z0.shape)}")
        z0_batch = z0.expand(self.n_candidates, -1)
        z0_proj = None
        if z_start_proj is not None:
            z0_proj = z_start_proj.to(self.device, dtype=torch.float32)
            if z0_proj.ndim != 2 or z0_proj.shape[0] != 1:
                raise ValueError(f"Expected z_start_proj shape (1, D), got {tuple(z0_proj.shape)}")

        if self._warm_start is not None:
            mean = self._warm_start.clone()
        else:
            mean = 0.5 * (self.cmd_low + self.cmd_high)
            mean = mean.unsqueeze(0).expand(self.horizon, -1).clone()
        std = self.init_std.unsqueeze(0).expand(self.horizon, -1).clone()

        best_seq = mean.clone()
        best_cost = float("inf")
        best_metrics: dict[str, float] = {}
        diagnostics: dict[str, Any] | None = None
        prepared_visited_snippets = None
        prepared_visited_snippet_emb = None
        if (
            exploration_bonus_mode == "visited_nn"
            and visited_rollout_bank
            and int(max(1, visited_rollout_tail_steps)) > 1
        ):
            prepared_visited_snippets = prepare_visited_rollout_snippet_bank(
                visited_rollout_bank,
                int(max(1, visited_rollout_tail_steps)),
                device=self.device,
                dtype=torch.float32,
            )
            if (
                prepared_visited_snippets is not None
                and heads is not None
                and heads.place_head is not None
                and int(max(1, visited_rollout_tail_steps)) == int(heads.place_head.snippet_len)
            ):
                prepared_visited_snippet_emb = heads.place_head(prepared_visited_snippets)

        for _ in range(self.cem_iters):
            noise = torch.randn(
                self.n_candidates, self.horizon, self.cmd_dim, device=self.device,
            )
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            samples = samples.clamp(
                self.cmd_low.view(1, 1, -1),
                self.cmd_high.view(1, 1, -1),
            )
            samples[0] = mean

            z_rollouts_proj = self.world_model.plan_rollout(z0_batch, samples)
            costs = torch.zeros(self.n_candidates, device=self.device)
            metrics: dict[str, torch.Tensor] = {}

            if heads is not None and heads.safety_head is not None:
                safety_cost = heads.safety_weight * heads.safety_head.score_trajectory(z_rollouts_proj)
                costs += safety_cost
                metrics["safety_cost"] = safety_cost
            else:
                safety_cost = torch.zeros(self.n_candidates, device=self.device)

            if (
                heads is not None
                and heads.exploration_weight > 0.0
                and (
                    heads.exploration is not None
                    or exploration_bonus_mode == "visited_nn"
                )
            ):
                n_cand, horizon, latent_dim = z_rollouts_proj.shape
                if exploration_bonus_mode == "visited_nn":
                    tail_steps = min(max(1, int(visited_rollout_tail_steps)), horizon)
                    tail_z = z_rollouts_proj[:, -tail_steps:, :]
                    has_visited_reference = False
                    if tail_steps > 1:
                        if prepared_visited_snippets is not None:
                            has_visited_reference = True
                            tail_dist = visited_rollout_snippet_novelty(
                                tail_z,
                                prepared_visited_snippets,
                                k=visited_rollout_knn_k,
                                place_head=heads.place_head if heads is not None else None,
                                visited_bank_emb=prepared_visited_snippet_emb,
                            )
                        else:
                            tail_dist = torch.zeros(self.n_candidates, device=self.device)
                    else:
                        if visited_rollout_bank:
                            has_visited_reference = True
                            tail_dist = visited_rollout_bank_novelty(
                                tail_z[:, -1, :],
                                visited_rollout_bank,
                                k=visited_rollout_knn_k,
                            )
                        else:
                            tail_dist = torch.zeros(self.n_candidates, device=self.device)
                    if has_visited_reference:
                        tail_margin = tail_dist - float(visited_rollout_margin)
                        bonus = tail_margin.clamp_min(0.0)
                    else:
                        tail_margin = torch.zeros_like(tail_dist)
                        bonus = torch.zeros_like(tail_dist)
                    metrics["visited_nn_distance"] = tail_dist
                    if visited_revisit_penalty_weight > 0.0 and has_visited_reference:
                        revisit_penalty = (-tail_margin).clamp_min(0.0)
                        costs += visited_revisit_penalty_weight * revisit_penalty
                        metrics["revisit_penalty"] = revisit_penalty
                elif exploration_bonus_mode == "sum":
                    bonus = heads.exploration(
                        z_rollouts_proj.reshape(n_cand * horizon, latent_dim),
                    ).reshape(n_cand, horizon).sum(dim=-1)
                else:
                    terminal_bonus = heads.exploration(z_rollouts_proj[:, -1, :])
                    if exploration_bonus_mode == "terminal":
                        bonus = terminal_bonus
                    elif exploration_bonus_mode == "gain":
                        # Compare rollout endpoints in the same predicted-latent
                        # space. Using the encoder-view current observation as
                        # the gain anchor reintroduces an enc-vs-pred mismatch.
                        first_bonus = heads.exploration(z_rollouts_proj[:, 0, :])
                        bonus = (terminal_bonus - first_bonus).clamp_min(0.0)
                    else:
                        raise ValueError(
                            f"Unsupported exploration_bonus_mode={exploration_bonus_mode!r}",
                        )
                effective_bonus = bonus
                if exploration_safety_gate_threshold is not None:
                    safety_mean = safety_cost / float(self.horizon)
                    exploration_gate = torch.sigmoid(
                        exploration_safety_gate_sharpness
                        * (exploration_safety_gate_threshold - safety_mean),
                    )
                    effective_bonus = bonus * exploration_gate
                    metrics["exploration_bonus_raw"] = bonus
                    metrics["exploration_safety_gate"] = exploration_gate
                costs -= heads.exploration_weight * effective_bonus
                metrics["exploration_bonus"] = effective_bonus

            if terminal_displacement_weight > 0.0:
                if z0_proj is None:
                    raise ValueError(
                        "terminal_displacement_weight > 0 requires z_start_proj.",
                    )
                final_proj = F.normalize(z_rollouts_proj[:, -1, :], dim=-1)
                start_proj = F.normalize(z0_proj, dim=-1).expand(self.n_candidates, -1)
                terminal_displacement = (1.0 - (final_proj * start_proj).sum(dim=-1)).clamp_min(0.0)
                costs -= terminal_displacement_weight * terminal_displacement
                metrics["terminal_displacement_bonus"] = terminal_displacement

            if self.action_penalty_weight > 0.0:
                act_penalty = samples.square().sum(dim=(1, 2))
                costs += self.action_penalty_weight * act_penalty
                metrics["action_penalty"] = act_penalty

            min_cost, min_idx = costs.min(dim=0)
            if min_cost.item() < best_cost:
                best_cost = min_cost.item()
                best_seq = samples[min_idx.item()].detach().clone()
                best_metrics = {
                    name: float(values[min_idx.item()].item())
                    for name, values in metrics.items()
                }

            if return_diagnostics:
                topk = min(max(1, int(diagnostics_topk)), self.n_candidates)
                top_idx = torch.topk(costs, k=topk, largest=False).indices.tolist()
                final_iteration_topk = []
                for rank, idx in enumerate(top_idx, start=1):
                    seq = samples[idx]
                    final_iteration_topk.append({
                        "rank": rank,
                        "candidate_index": int(idx),
                        "cost": float(costs[idx].item()),
                        "metrics": {
                            name: float(values[idx].item())
                            for name, values in metrics.items()
                        },
                        **summarize_command_sequence(seq),
                    })
                diagnostics = {
                    "final_iteration_topk": final_iteration_topk,
                    "final_iteration_best_cost": float(costs[top_idx[0]].item()),
                    "final_iteration_best_metrics": {
                        name: float(values[top_idx[0]].item())
                        for name, values in metrics.items()
                    },
                }

            elite_idx = torch.topk(costs, k=self.n_elite, largest=False).indices
            elite = samples[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(self.min_std)

        self._warm_start = torch.cat(
            [best_seq[1:], best_seq[-1:].clone()], dim=0,
        ).detach()

        if return_diagnostics:
            diagnostics = dict(diagnostics or {})
            diagnostics.update({
                "selected_plan": {
                    "source": "best_overall",
                    "cost": float(best_cost),
                    "metrics": dict(best_metrics),
                    **summarize_command_sequence(best_seq),
                },
            })

        return best_seq, best_cost, best_metrics, diagnostics


# ---- Argument parsing ---------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure-perception maze inference with learned safety and novelty.",
    )
    p.add_argument("--ppo_ckpt", type=str, required=True)
    p.add_argument("--wm_ckpt", type=str, required=True)
    p.add_argument("--scorer_ckpt", type=str, default=None,
                   help="Optional trajectory scorer checkpoint with safety / exploration heads. Goal head is ignored by the active planner.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid_rows", type=int, default=4)
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--cell_size", type=float, default=0.70)
    p.add_argument("--wall_thickness", type=float, default=0.20)
    p.add_argument("--n_beacons", type=int, default=2)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--target_beacon", type=str, default=None)
    
    p.add_argument("--steps", type=int, default=480,
                   help="Number of planner steps (each may repeat the low-level controller).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sim_backend", type=str, default="auto")
    p.add_argument("--show_viewer", action="store_true")
    p.add_argument("--allow_mixed_latent_wm", action="store_true",
                   help="Override the pure-perception guard and allow a world-model checkpoint that uses proprio.")
    
    p.add_argument("--plan_horizon", type=int, default=5)
    p.add_argument("--n_candidates", type=int, default=300)
    p.add_argument("--cem_iters", type=int, default=30)
    p.add_argument("--elite_frac", type=float, default=0.10)
    p.add_argument("--mpc_execute", type=int, default=1)
    p.add_argument("--macro_action_repeat", type=int, default=1,
                   help="Low-level control repeats per world-model planner step.")
    p.add_argument("--score_space", type=str, default="mixed",
                   choices=["mixed", "raw", "proj"],
                   help="Deprecated legacy flag. Planner now always scores in projected latent space.")
    p.add_argument("--cmd_low", type=float, nargs=3, default=[-0.4, -0.3, -1.0])
    p.add_argument("--cmd_high", type=float, nargs=3, default=[0.8, 0.3, 1.0])
    p.add_argument("--cem_init_std", type=float, nargs=3, default=[0.3, 0.15, 0.4])
    p.add_argument("--cem_min_std", type=float, nargs=3, default=[0.05, 0.03, 0.08])
    
    p.add_argument("--novelty_weight", type=float, default=10.0,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--exploration_weight", type=float, default=None,
                   help="Override learned RND exploration weight from the scorer checkpoint.")
    p.add_argument("--exploration_bonus_mode", type=str, default="terminal",
                   choices=["sum", "terminal", "gain", "visited_nn"],
                   help="How to score novelty over predicted rollout latents.")
    p.add_argument("--terminal_displacement_weight", type=float, default=0.0,
                   help="Optional bonus on predicted terminal latent displacement from the current observation.")
    p.add_argument("--exploration_safety_gate_threshold", type=float, default=None,
                   help="Optional mean predicted safety-cost threshold for gating the learned novelty bonus.")
    p.add_argument("--exploration_safety_gate_sharpness", type=float, default=5.0,
                   help="Sigmoid sharpness for the optional safety gate on the learned novelty bonus.")
    p.add_argument("--rnd_online_lr", type=float, default=1e-3,
                   help="Online adaptation rate for the learned exploration bonus. Set to 0 to disable.")
    p.add_argument("--visited_nn_k", type=int, default=8,
                   help="Average the k nearest rollout-bank distances when exploration_bonus_mode=visited_nn.")
    p.add_argument("--visited_nn_margin", type=float, default=0.0,
                   help="Only visited_nn distance above this margin counts as novelty.")
    p.add_argument("--visited_nn_tail_steps", type=int, default=1,
                   help="Require novelty to persist over the last this many rollout steps when exploration_bonus_mode=visited_nn.")
    p.add_argument("--recent_latent_window", type=int, default=128,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--revisit_penalty_weight", type=float, default=0.0,
                   help="Additional penalty for rollout-tail states that remain inside the visited_nn margin.")
    p.add_argument("--forward_bonus_weight", type=float, default=0.30,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--forward_bonus_safety_threshold", type=float, default=0.12,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--current_safety_brake_threshold", type=float, default=0.20,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--unsafe_macro_action_repeat", type=int, default=1,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--frontier_knn", type=int, default=8,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--goal_progress_weight", type=float, default=8.0,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--route_progress_weight", type=float, default=7.0,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--recover_displacement_weight", type=float, default=2.5,
                   help="Legacy routing-era flag. Ignored by the active planner.")
    p.add_argument("--action_penalty_weight", type=float, default=0.001)
    p.add_argument("--visited_bank_size", type=int, default=512,
                   help="Capacity of the executed rollout-latent bank used by visited_nn novelty.")
    p.add_argument("--audit_plan", action="store_true",
                   help="Log top-k planner candidates at each audited replan together with the actual outcome after executing the chosen first macro step.")
    p.add_argument("--audit_every", type=int, default=1,
                   help="Audit every nth replan event when --audit_plan is enabled.")
    p.add_argument("--audit_topk", type=int, default=5,
                   help="Number of lowest-cost candidates to store per audited replan.")
    p.add_argument("--prototype_sim_threshold", type=float, default=0.995,
                   help="Add a new prototype only when cosine similarity to the nearest stored prototype is below this threshold.")
    p.add_argument("--keyframe_sim_threshold", type=float, default=0.985,
                   help="Match the current observation to an existing keyframe node above this similarity.")
    p.add_argument("--keyframe_match_radius_m", type=float, default=1.5,
                   help="Maximum dead-reckoned distance allowed for keyframe reuse.")
    p.add_argument("--keyframe_min_step_gap", type=int, default=8,
                   help="Minimum temporal separation before matching the current node back onto an older keyframe.")
    p.add_argument("--keyframe_add_interval", type=int, default=24,
                   help="Add a fresh keyframe if this many steps pass without a new node.")
    p.add_argument("--stall_plateau_steps", type=int, default=200,
                   help="Trigger routing after this many steps without a novel prototype or route/goal progress.")
    p.add_argument("--subgoal_budget_steps", type=int, default=120,
                   help="Maximum number of planner steps to spend following graph waypoints during a route episode.")
    p.add_argument("--subgoal_min_age_steps", type=int, default=120,
                   help="A keyframe must not have been visited for at least this many steps to qualify as a route target.")
    p.add_argument("--subgoal_frontier_window_steps", type=int, default=800,
                   help="Prefer keyframes inserted within this recent window when choosing a frontier route target.")
    p.add_argument("--subgoal_cooldown_steps", type=int, default=120,
                   help="Minimum wait before another plateau-triggered route can activate.")
    p.add_argument("--route_min_hops", type=int, default=3,
                   help="Minimum keyframe-graph distance required for a route target.")
    p.add_argument("--goal_direct_sim_threshold", type=float, default=0.72,
                   help="Legacy goal-mode flag. Ignored by the active planner.")
    p.add_argument("--goal_activation_hold_steps", type=int, default=3,
                   help="Legacy goal-mode flag. Ignored by the active planner.")
    p.add_argument("--goal_route_improve_margin", type=float, default=0.03,
                   help="Require a route target to improve breadcrumb similarity by at least this margin before treating it as goal-directed.")
    p.add_argument("--success_goal_sim_threshold", type=float, default=0.90,
                   help="Projected-latent success threshold on current breadcrumb similarity.")
    p.add_argument("--success_goal_raw_sim_threshold", type=float, default=0.95,
                   help="Raw-latent success threshold on current breadcrumb similarity.")
    p.add_argument("--success_hold_steps", type=int, default=6,
                   help="Require this many consecutive high-similarity frames before declaring perceptual goal detection.")
    p.add_argument("--stuck_window_steps", type=int, default=12,
                   help="Window for stuck detection using odometry and latent displacement.")
    p.add_argument("--stuck_cmd_threshold", type=float, default=0.25,
                   help="Minimum commanded magnitude for a step to count toward stuck detection.")
    p.add_argument("--stuck_odom_threshold", type=float, default=0.015,
                   help="Maximum mean dead-reckoned XY motion per step while still considered stuck.")
    p.add_argument("--stuck_latent_threshold", type=float, default=0.010,
                   help="Maximum mean latent displacement per step while still considered stuck.")
    p.add_argument("--recover_budget_steps", type=int, default=28,
                   help="Duration of recover mode after a stuck event.")
    p.add_argument("--recover_cooldown_steps", type=int, default=24,
                   help="Minimum wait before another recover episode can trigger.")
    p.add_argument("--history_len", type=int, default=None,
                   help="Deprecated alias for --visited_bank_size.")
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

    args = p.parse_args()
    if args.history_len is not None:
        args.visited_bank_size = args.history_len
    args.auto_scaled_defaults = {}
    return args


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
    cmd_w = state["predictor.action_embed.patch_embed.weight"]
    latent_dim = int(pos_embed.shape[-1])
    patch_size = int(patch_w.shape[-1])
    n_tokens = int(pos_embed.shape[1] - 1)
    grid = int(round(math.sqrt(n_tokens)))
    image_size = grid * patch_size
    max_seq_len = int(pred_pos.shape[1])
    cmd_dim = int(cmd_w.shape[1])
    use_proprio = any(k.startswith("encoder.prop_enc.") for k in state)
    command_representation = ckpt.get(
        "command_representation",
        "mean_scaled" if cmd_dim == 3 else "active_block",
    )
    command_latency = int(ckpt.get("command_latency", 2))
    if command_representation == "active_block":
        if cmd_dim % 3 != 0:
            raise ValueError(
                f"Active-block checkpoint has cmd_dim={cmd_dim}, which is not divisible by 3."
            )
        command_block_size = int(ckpt.get("action_block_size", cmd_dim // 3))
        inferred_block_size = cmd_dim // 3
        if int(command_block_size) != int(inferred_block_size):
            raise ValueError(
                f"Checkpoint action_block_size={command_block_size} does not match "
                f"inferred block size {inferred_block_size} from cmd_dim={cmd_dim}."
            )
    else:
        command_block_size = int(ckpt.get("action_block_size", 1))

    model = LeWorldModel(
        latent_dim=latent_dim,
        cmd_dim=cmd_dim,
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
        "max_seq_len": max_seq_len,
        "cmd_dim": cmd_dim,
        "command_representation": command_representation,
        "command_latency": command_latency,
        "command_block_size": command_block_size,
    }


def build_command_sampling_config(
    args: argparse.Namespace,
    wm_meta: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cmd_low = torch.tensor(args.cmd_low, dtype=torch.float32)
    cmd_high = torch.tensor(args.cmd_high, dtype=torch.float32)
    init_std = torch.tensor(args.cem_init_std, dtype=torch.float32)
    min_std = torch.tensor(args.cem_min_std, dtype=torch.float32)
    cmd_repr = wm_meta.get("command_representation", "mean_scaled")
    cmd_dim = int(wm_meta.get("cmd_dim", 3))

    if cmd_repr != "active_block":
        if cmd_dim != 3:
            raise ValueError(f"Expected cmd_dim=3 for {cmd_repr}, got {cmd_dim}.")
        return cmd_low, cmd_high, init_std, min_std

    block_size = int(wm_meta.get("command_block_size", 0))
    if block_size <= 0:
        raise ValueError(f"Invalid command_block_size={block_size} for active_block model.")
    if cmd_dim != block_size * 3:
        raise ValueError(
            f"Active-block model has cmd_dim={cmd_dim}, expected {block_size * 3}."
        )
    return (
        cmd_low.repeat(block_size),
        cmd_high.repeat(block_size),
        init_std.repeat(block_size),
        min_std.repeat(block_size),
    )


def format_command_for_log(
    cmd_values: list[float],
    wm_meta: dict[str, Any],
) -> str:
    if wm_meta.get("command_representation") != "active_block":
        return f"[{cmd_values[0]:+.2f}, {cmd_values[1]:+.2f}, {cmd_values[2]:+.2f}]"

    block = np.asarray(cmd_values, dtype=np.float32).reshape(-1, 3)
    mean_cmd = block.mean(axis=0)
    first_cmd = block[0]
    last_cmd = block[-1]
    return (
        f"mean=[{mean_cmd[0]:+.2f}, {mean_cmd[1]:+.2f}, {mean_cmd[2]:+.2f}] "
        f"first=[{first_cmd[0]:+.2f}, {first_cmd[1]:+.2f}, {first_cmd[2]:+.2f}] "
        f"last=[{last_cmd[0]:+.2f}, {last_cmd[1]:+.2f}, {last_cmd[2]:+.2f}]"
    )


def load_planner_heads(
    ckpt_path: str | None,
    device: torch.device,
    wm_meta: dict[str, Any],
    macro_action_repeat: int,
) -> tuple[PlannerHeads, dict[str, Any]]:
    heads = PlannerHeads()
    meta: dict[str, Any] = {}
    if ckpt_path is None:
        return heads, meta
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"scorer checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "latent_dim" in ckpt and int(ckpt["latent_dim"]) != int(wm_meta["latent_dim"]):
        raise ValueError(
            f"Scorer latent_dim={ckpt['latent_dim']} does not match world-model latent_dim={wm_meta['latent_dim']}"
        )
    if "image_size" in ckpt and int(ckpt["image_size"]) != int(wm_meta["image_size"]):
        raise ValueError(
            f"Scorer image_size={ckpt['image_size']} does not match world-model image_size={wm_meta['image_size']}"
        )
    if "patch_size" in ckpt and int(ckpt["patch_size"]) != int(wm_meta["patch_size"]):
        raise ValueError(
            f"Scorer patch_size={ckpt['patch_size']} does not match world-model patch_size={wm_meta['patch_size']}"
        )
    if "use_proprio" in ckpt and bool(ckpt["use_proprio"]) != bool(wm_meta["use_proprio"]):
        raise ValueError(
            f"Scorer use_proprio={ckpt['use_proprio']} does not match world-model use_proprio={wm_meta['use_proprio']}"
        )
    if "max_seq_len" in ckpt and int(ckpt["max_seq_len"]) != int(wm_meta["max_seq_len"]):
        raise ValueError(
            f"Scorer max_seq_len={ckpt['max_seq_len']} does not match world-model max_seq_len={wm_meta['max_seq_len']}"
        )
    if "seq_len" in ckpt and int(ckpt["seq_len"]) != int(wm_meta["max_seq_len"]):
        raise ValueError(
            f"Scorer seq_len={ckpt['seq_len']} does not match world-model max_seq_len={wm_meta['max_seq_len']}"
        )
    if "temporal_stride" in ckpt and int(ckpt["temporal_stride"]) != int(macro_action_repeat):
        raise ValueError(
            f"Scorer temporal_stride={ckpt['temporal_stride']} does not match "
            f"--macro_action_repeat={macro_action_repeat}"
        )
    if "action_block_size" in ckpt and int(ckpt["action_block_size"]) != int(macro_action_repeat):
        raise ValueError(
            f"Scorer action_block_size={ckpt['action_block_size']} does not match "
            f"--macro_action_repeat={macro_action_repeat}"
        )
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    dropout = float(ckpt.get("dropout", 0.0))
    latent_dim = int(wm_meta["latent_dim"])
    exploration_dim = int(ckpt.get("exploration_feature_dim", 128))
    place_embedding_dim = int(ckpt.get("place_embedding_dim", 64))
    place_snippet_len = int(ckpt.get("place_snippet_len", 1))

    if ckpt.get("safety_head") is not None:
        safety = LatentEnergyHead(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        clean_load_state(safety, ckpt["safety_head"])
        safety.eval()
        heads.safety_head = safety

    if ckpt.get("goal_head") is not None:
        goal = GoalEnergyHead(latent_dim=latent_dim, dropout=dropout).to(device)
        clean_load_state(goal, ckpt["goal_head"])
        goal.eval()
        heads.goal_head = goal

    if ckpt.get("exploration") is not None:
        exploration = ExplorationBonus(
            latent_dim=latent_dim,
            feature_dim=exploration_dim,
        ).to(device)
        clean_load_state(exploration, ckpt["exploration"])
        exploration.eval()
        heads.exploration = exploration

    if ckpt.get("place_head") is not None:
        place_head = PlaceSnippetHead(
            latent_dim=latent_dim,
            snippet_len=place_snippet_len,
            hidden_dim=hidden_dim,
            embedding_dim=place_embedding_dim,
            dropout=dropout,
        ).to(device)
        clean_load_state(place_head, ckpt["place_head"])
        place_head.eval()
        heads.place_head = place_head

    heads.goal_weight = float(ckpt.get("goal_weight", 0.0))
    heads.exploration_weight = float(ckpt.get("exploration_weight", 0.0))
    heads.safety_weight = 1.0
    meta = {
        "has_safety_head": heads.safety_head is not None,
        "has_goal_head": heads.goal_head is not None,
        "has_exploration": heads.exploration is not None,
        "has_place_head": heads.place_head is not None,
        "has_progress_head_ckpt": ckpt.get("progress_head") is not None,
        "goal_weight": heads.goal_weight,
        "exploration_weight": heads.exploration_weight,
        "safety_mode": ckpt.get("safety_mode"),
        "seq_len": ckpt.get("seq_len"),
        "temporal_stride": ckpt.get("temporal_stride"),
        "action_block_size": ckpt.get("action_block_size"),
        "window_stride": ckpt.get("window_stride"),
        "use_proprio": ckpt.get("use_proprio"),
        "safety_latent_source": ckpt.get("safety_latent_source"),
        "exploration_latent_source": ckpt.get("exploration_latent_source"),
        "place_latent_source": ckpt.get("place_latent_source"),
        "goal_latent_source": ckpt.get("goal_latent_source"),
        "progress_latent_source": ckpt.get("progress_latent_source"),
        "place_snippet_len": ckpt.get("place_snippet_len"),
        "place_embedding_dim": ckpt.get("place_embedding_dim"),
    }
    return heads, meta

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


def canonical_breadcrumb_proprio(
    q0: torch.Tensor,
    device: torch.device,
    torch_mod,
) -> torch.Tensor:
    """Return a neutral proprio token for externally provided goal images.

    When the world model is trained with proprio enabled, breadcrumb / target
    images still need a proprio input at encode time. For the intended
    deployment setup ("here is a picture of the object, go find it"), there is
    no meaningful robot state attached to that image, so we use a canonical
    standing token:

    - nominal spawn height
    - identity orientation
    - zero linear / angular velocity
    - zero joint offsets / joint velocities
    - zero previous action

    This keeps target-image encoding runtime-valid without requiring simulator
    oracle state.
    """
    dtype = q0.dtype
    batch = 1
    pos_z = torch_mod.full((batch, 1), ROBOT_SPAWN_Z, device=device, dtype=dtype)
    quat = torch_mod.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    vel_b = torch_mod.zeros((batch, 3), device=device, dtype=dtype)
    ang_b = torch_mod.zeros((batch, 3), device=device, dtype=dtype)
    q_rel = torch_mod.zeros((batch, 12), device=device, dtype=dtype)
    dq = torch_mod.zeros((batch, 12), device=device, dtype=dtype)
    prev_action = torch_mod.zeros((batch, 12), device=device, dtype=dtype)
    return torch_mod.cat([pos_z, quat, vel_b, ang_b, q_rel, dq, prev_action], dim=1)

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


def select_score_latent(
    z_raw: torch.Tensor,
    z_proj: torch.Tensor,
    score_space: str,
) -> torch.Tensor:
    """Select the latent representation used for scoring / memory."""
    if score_space == "proj":
        return z_proj
    return z_raw


@torch.no_grad()
def estimate_current_safety_energy(
    z_proj: torch.Tensor,
    heads: PlannerHeads,
) -> float | None:
    if heads.safety_head is None:
        return None
    value = heads.safety_head(z_proj.to(next(heads.safety_head.parameters()).device))
    return float(value.reshape(-1)[0].item())


@torch.no_grad()
def teacher_forced_rollout_latent_pair(
    world_model: LeWorldModel,
    z_start_raw: torch.Tensor,
    command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-step predicted raw/projected latents for an executed command."""
    if command.ndim == 1:
        action_seq = command.unsqueeze(0).unsqueeze(0)
    elif command.ndim == 2:
        action_seq = command.unsqueeze(0)
    else:
        action_seq = command
    action_seq = action_seq.to(z_start_raw.device)
    z_pred_raw = world_model.plan_rollout_raw(z_start_raw, action_seq)
    z_pred_proj = world_model.pred_projector.forward_seq(z_pred_raw)
    return z_pred_raw[:, -1, :].detach(), z_pred_proj[:, -1, :].detach()


@torch.no_grad()
def teacher_forced_rollout_latent(
    world_model: LeWorldModel,
    z_start_raw: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Project one executed command into the rollout latent space."""
    _, z_pred_proj = teacher_forced_rollout_latent_pair(
        world_model,
        z_start_raw,
        command,
    )
    return z_pred_proj


@torch.no_grad()
def visited_rollout_bank_novelty(
    query_z: torch.Tensor,
    visited_bank: list[torch.Tensor] | None,
    k: int = 8,
) -> torch.Tensor:
    """Novelty from rollout-space distance to executed-trajectory memory.

    Uses mean squared distance in projected latent space, averaged over the
    k nearest visited rollout latents. This is a density score rather than a
    per-state RND error, so slight view changes in the same local basin should
    collapse as the bank fills in.
    """
    if not visited_bank:
        return torch.zeros(query_z.shape[0], device=query_z.device, dtype=query_z.dtype)

    bank = torch.stack(
        [z.reshape(-1) for z in visited_bank],
        dim=0,
    ).to(device=query_z.device, dtype=query_z.dtype)
    dists = (query_z.unsqueeze(1) - bank.unsqueeze(0)).square().mean(dim=-1)
    k_eff = min(max(1, int(k)), int(bank.shape[0]))
    return torch.topk(dists, k=k_eff, largest=False, dim=1).values.mean(dim=1)


@torch.no_grad()
def prepare_visited_rollout_snippet_bank(
    visited_bank: list[torch.Tensor] | None,
    tail_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not visited_bank:
        return None
    if len(visited_bank) < int(tail_steps):
        return None
    bank = torch.stack(
        [z.reshape(-1) for z in visited_bank],
        dim=0,
    ).to(device=device, dtype=dtype)
    return bank.unfold(0, int(tail_steps), 1).permute(0, 2, 1).contiguous()


@torch.no_grad()
def visited_rollout_snippet_novelty(
    query_seq: torch.Tensor,
    visited_bank_seq: torch.Tensor | None,
    k: int = 8,
    place_head: PlaceSnippetHead | None = None,
    visited_bank_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Novelty from kNN distance against executed rollout snippets.

    Args:
        query_seq: (B, T, D) rollout tail snippets.
        visited_bank_seq: (N, T, D) executed rollout snippets.
        k: number of nearest snippets to average.

    Returns:
        (B,) mean squared distance to the k nearest executed snippets.
    """
    if visited_bank_seq is None or visited_bank_seq.numel() == 0:
        return torch.zeros(query_seq.shape[0], device=query_seq.device, dtype=query_seq.dtype)

    if visited_bank_emb is not None:
        query_emb = place_head(query_seq) if place_head is not None else query_seq.reshape(query_seq.shape[0], -1)
        dists = (query_emb.unsqueeze(1) - visited_bank_emb.unsqueeze(0)).square().mean(dim=-1)
    elif place_head is not None and int(query_seq.shape[1]) == int(place_head.snippet_len):
        query_emb = place_head(query_seq)
        bank_emb = place_head(visited_bank_seq)
        dists = (query_emb.unsqueeze(1) - bank_emb.unsqueeze(0)).square().mean(dim=-1)
    else:
        dists = (query_seq.unsqueeze(1) - visited_bank_seq.unsqueeze(0)).square().mean(dim=(2, 3))
    k_eff = min(max(1, int(k)), int(visited_bank_seq.shape[0]))
    return torch.topk(dists, k=k_eff, largest=False, dim=1).values.mean(dim=1)


def choose_redundant_prototype(
    bank: list[torch.Tensor],
    hit_counts: list[int],
) -> int:
    """Replace the most redundant prototype, biased against high-hit entries."""
    if len(bank) <= 1:
        return 0

    bank_tensor = torch.stack(bank)
    sim_matrix = bank_tensor @ bank_tensor.T
    sim_matrix.fill_diagonal_(-1.0)
    redundancy = sim_matrix.max(dim=1).values

    best_idx = 0
    best_key = (-float("inf"), -float("inf"))
    for idx, hits in enumerate(hit_counts):
        key = (float(redundancy[idx].item()), -float(hits))
        if key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def update_prototype_bank(
    bank: list[torch.Tensor],
    hit_counts: list[int],
    latent: torch.Tensor,
    n_seen: int,
    max_size: int,
    sim_threshold: float,
) -> PrototypeBankUpdate:
    """Maintain a novelty bank of distinct latent prototypes."""
    n_seen += 1
    if max_size <= 0:
        return PrototypeBankUpdate(n_seen=n_seen, action="off")

    latent = latent.detach().cpu().float()
    latent = F.normalize(latent.unsqueeze(0), p=2, dim=-1).squeeze(0)
    filled_before = len(bank) >= max_size

    if not bank:
        bank.append(latent)
        hit_counts.append(1)
        return PrototypeBankUpdate(
            n_seen=n_seen,
            action="inserted",
            filled_now=(not filled_before and len(bank) >= max_size),
            prototype_idx=0,
        )

    bank_tensor = torch.stack(bank)
    sims = torch.mv(bank_tensor, latent)
    nearest_sim, nearest_idx = sims.max(dim=0)
    nearest_sim_f = float(nearest_sim.item())
    nearest_idx_i = int(nearest_idx.item())

    if nearest_sim_f >= sim_threshold:
        hit_counts[nearest_idx_i] += 1
        return PrototypeBankUpdate(
            n_seen=n_seen,
            action="hit",
            nearest_sim=nearest_sim_f,
            prototype_idx=nearest_idx_i,
        )

    if len(bank) < max_size:
        bank.append(latent)
        hit_counts.append(1)
        return PrototypeBankUpdate(
            n_seen=n_seen,
            action="inserted",
            nearest_sim=nearest_sim_f,
            filled_now=(not filled_before and len(bank) >= max_size),
            prototype_idx=len(bank) - 1,
        )

    replace_idx = choose_redundant_prototype(bank, hit_counts)
    bank[replace_idx] = latent
    hit_counts[replace_idx] = 1
    return PrototypeBankUpdate(
        n_seen=n_seen,
        action="replaced",
        nearest_sim=nearest_sim_f,
        replace_idx=replace_idx,
        prototype_idx=replace_idx,
    )


def apply_prototype_update_steps(
    last_seen_steps: list[int],
    insert_steps: list[int],
    update: PrototypeBankUpdate,
    step: int,
) -> None:
    """Keep per-prototype metadata aligned with bank mutations."""
    idx = update.prototype_idx
    if idx is None:
        return
    if update.action == "inserted":
        if idx == len(last_seen_steps):
            last_seen_steps.append(step)
            insert_steps.append(step)
        elif 0 <= idx < len(last_seen_steps):
            last_seen_steps[idx] = step
            insert_steps[idx] = step
        return
    if 0 <= idx < len(last_seen_steps):
        last_seen_steps[idx] = step
    if update.action == "replaced" and 0 <= idx < len(insert_steps):
        insert_steps[idx] = step


def apply_prototype_update_graph(
    neighbors: list[set[int]],
    update: PrototypeBankUpdate,
) -> None:
    """Keep prototype-transition graph aligned with bank mutations."""
    idx = update.prototype_idx
    if idx is None:
        return
    if update.action == "inserted":
        if idx == len(neighbors):
            neighbors.append(set())
        elif 0 <= idx < len(neighbors):
            stale = list(neighbors[idx])
            for other in stale:
                if 0 <= other < len(neighbors):
                    neighbors[other].discard(idx)
            neighbors[idx].clear()
        return
    if update.action == "replaced" and 0 <= idx < len(neighbors):
        stale = list(neighbors[idx])
        for other in stale:
            if 0 <= other < len(neighbors):
                neighbors[other].discard(idx)
        neighbors[idx].clear()


def add_prototype_graph_edge(
    neighbors: list[set[int]],
    prev_idx: int | None,
    cur_idx: int | None,
) -> None:
    if prev_idx is None or cur_idx is None or prev_idx == cur_idx:
        return
    if prev_idx < 0 or cur_idx < 0:
        return
    if prev_idx >= len(neighbors) or cur_idx >= len(neighbors):
        return
    neighbors[prev_idx].add(cur_idx)
    neighbors[cur_idx].add(prev_idx)


def cosine_similarity_scalar(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = F.normalize(a.detach().cpu().float().unsqueeze(0), p=2, dim=-1)
    b_norm = F.normalize(b.detach().cpu().float().unsqueeze(0), p=2, dim=-1)
    return float(F.cosine_similarity(a_norm, b_norm, dim=-1).item())


def latent_displacement(a: torch.Tensor, b: torch.Tensor) -> float:
    return max(0.0, 1.0 - cosine_similarity_scalar(a, b))


def estimate_dead_reckoning_step(
    prev_xy: np.ndarray,
    yaw_rad: float,
    proprio: torch.Tensor,
    dt_s: float,
) -> np.ndarray:
    """Integrate a short XY odometry step from body-frame velocity."""
    prop_np = proprio[0].detach().cpu().numpy()
    vel_b = np.asarray(prop_np[5:8], dtype=np.float32)
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    vel_world_xy = np.array([
        c * vel_b[0] - s * vel_b[1],
        s * vel_b[0] + c * vel_b[1],
    ], dtype=np.float32)
    return np.asarray(prev_xy, dtype=np.float32) + vel_world_xy * float(dt_s)


def match_keyframe_node(
    nodes: list[KeyframeNode],
    score_latent: torch.Tensor,
    odom_xy: np.ndarray,
    step: int,
    sim_threshold: float,
    match_radius_m: float,
    min_step_gap: int,
) -> int | None:
    if not nodes:
        return None

    best_idx = None
    best_sim = -1.0
    for node in nodes:
        if step - node.last_seen_step < min_step_gap:
            continue
        odom_dist = float(np.linalg.norm(np.asarray(node.odom_xy, dtype=np.float32) - odom_xy))
        if odom_dist > match_radius_m:
            continue
        sim = cosine_similarity_scalar(score_latent, node.score_latent)
        if sim >= sim_threshold and sim > best_sim:
            best_sim = sim
            best_idx = node.idx
    return best_idx


def add_keyframe_node(
    nodes: list[KeyframeNode],
    neighbors: list[set[int]],
    score_latent: torch.Tensor,
    proj_latent: torch.Tensor,
    odom_xy: np.ndarray,
    yaw_rad: float,
    step: int,
) -> int:
    node_idx = len(nodes)
    nodes.append(KeyframeNode(
        idx=node_idx,
        score_latent=score_latent.detach().cpu().float().clone(),
        proj_latent=proj_latent.detach().cpu().float().clone(),
        step=step,
        last_seen_step=step,
        visit_count=1,
        odom_xy=(float(odom_xy[0]), float(odom_xy[1])),
        yaw_rad=float(yaw_rad),
    ))
    neighbors.append(set())
    return node_idx


def touch_keyframe_node(
    node: KeyframeNode,
    odom_xy: np.ndarray,
    yaw_rad: float,
    step: int,
) -> None:
    node.last_seen_step = step
    node.visit_count += 1
    node.odom_xy = (float(odom_xy[0]), float(odom_xy[1]))
    node.yaw_rad = float(yaw_rad)


def bfs_shortest_paths(
    neighbors: list[set[int]],
    start_idx: int,
) -> tuple[dict[int, int], dict[int, int | None]]:
    from collections import deque

    dist: dict[int, int] = {start_idx: 0}
    parent: dict[int, int | None] = {start_idx: None}
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        for nxt in neighbors[idx]:
            if nxt in dist:
                continue
            dist[nxt] = dist[idx] + 1
            parent[nxt] = idx
            queue.append(nxt)
    return dist, parent


def reconstruct_path(
    parent: dict[int, int | None],
    target_idx: int,
) -> list[int]:
    path: list[int] = []
    idx: int | None = target_idx
    while idx is not None:
        path.append(idx)
        idx = parent.get(idx)
    path.reverse()
    return path


def choose_route_path(
    bank: list[torch.Tensor],
    hit_counts: list[int],
    last_seen_steps: list[int],
    insert_steps: list[int],
    neighbors: list[set[int]],
    current_idx: int,
    current_step: int,
    min_age_steps: int,
    frontier_window_steps: int,
    min_hops: int,
    goal_latent: torch.Tensor | None,
    goal_route_improve_margin: float,
) -> tuple[str, int, list[int], float | None] | None:
    """Choose a graph route toward either a better goal basin or a frontier node."""
    if not bank or current_idx < 0 or current_idx >= len(bank):
        return None

    dist, parent = bfs_shortest_paths(neighbors, current_idx)
    reachable = [
        idx for idx, hops in dist.items()
        if idx != current_idx and hops >= max(1, min_hops)
    ]
    if not reachable:
        return None

    current_goal_sim = None
    goal_sims: list[float] | None = None
    if goal_latent is not None:
        goal_sims = [cosine_similarity_scalar(node, goal_latent) for node in bank]
        current_goal_sim = goal_sims[current_idx]
        better_goal_nodes = [
            idx for idx in reachable
            if goal_sims[idx] >= current_goal_sim + goal_route_improve_margin
        ]
        if better_goal_nodes:
            target_idx = max(
                better_goal_nodes,
                key=lambda idx: (
                    goal_sims[idx],
                    -dist[idx],
                    -hit_counts[idx],
                    insert_steps[idx],
                ),
            )
            path = reconstruct_path(parent, target_idx)
            if len(path) >= 2:
                return ("goal", target_idx, path[1:], goal_sims[target_idx])

    newest_insert = max(insert_steps) if insert_steps else current_step
    frontier_cutoff = newest_insert - max(0, frontier_window_steps)
    frontier_nodes = [
        idx for idx in reachable
        if (current_step - int(last_seen_steps[idx])) >= min_age_steps
        and int(insert_steps[idx]) >= frontier_cutoff
    ]
    if not frontier_nodes:
        frontier_nodes = [
            idx for idx in reachable
            if (current_step - int(last_seen_steps[idx])) >= min_age_steps
        ]
    if not frontier_nodes:
        frontier_nodes = reachable
    if not frontier_nodes:
        return None

    target_idx = max(
        frontier_nodes,
        key=lambda idx: (
            -hit_counts[idx],
            -len(neighbors[idx]),
            insert_steps[idx],
            dist[idx],
        ),
    )
    path = reconstruct_path(parent, target_idx)
    if len(path) < 2:
        return None
    target_goal_sim = None if goal_sims is None else goal_sims[target_idx]
    return ("frontier", target_idx, path[1:], target_goal_sim)


def choose_keyframe_route_path(
    nodes: list[KeyframeNode],
    neighbors: list[set[int]],
    current_idx: int,
    current_step: int,
    min_age_steps: int,
    frontier_window_steps: int,
    min_hops: int,
    goal_latent: torch.Tensor | None,
    goal_route_improve_margin: float,
) -> tuple[str, int, list[int], float | None] | None:
    if not nodes or current_idx < 0 or current_idx >= len(nodes):
        return None

    dist, parent = bfs_shortest_paths(neighbors, current_idx)
    reachable = [
        idx for idx, hops in dist.items()
        if idx != current_idx and hops >= max(1, min_hops)
    ]
    if not reachable:
        return None

    goal_sims: dict[int, float] = {}
    if goal_latent is not None:
        current_goal_sim = cosine_similarity_scalar(nodes[current_idx].score_latent, goal_latent)
        better_goal_nodes = []
        for idx in reachable:
            sim = cosine_similarity_scalar(nodes[idx].score_latent, goal_latent)
            goal_sims[idx] = sim
            if sim >= current_goal_sim + goal_route_improve_margin:
                better_goal_nodes.append(idx)
        if better_goal_nodes:
            target_idx = max(
                better_goal_nodes,
                key=lambda idx: (
                    goal_sims[idx],
                    -dist[idx],
                    -nodes[idx].visit_count,
                    nodes[idx].step,
                ),
            )
            path = reconstruct_path(parent, target_idx)
            if len(path) >= 2:
                return ("goal", target_idx, path[1:], goal_sims[target_idx])

    newest_insert = max(node.step for node in nodes)
    frontier_cutoff = newest_insert - max(0, frontier_window_steps)
    frontier_nodes = [
        idx for idx in reachable
        if (current_step - nodes[idx].last_seen_step) >= min_age_steps
        and nodes[idx].step >= frontier_cutoff
    ]
    if not frontier_nodes:
        frontier_nodes = [
            idx for idx in reachable
            if (current_step - nodes[idx].last_seen_step) >= min_age_steps
        ]
    if not frontier_nodes:
        frontier_nodes = reachable
    if not frontier_nodes:
        return None

    target_idx = max(
        frontier_nodes,
        key=lambda idx: (
            -nodes[idx].visit_count,
            -len(neighbors[idx]),
            nodes[idx].step,
            dist[idx],
        ),
    )
    path = reconstruct_path(parent, target_idx)
    if len(path) < 2:
        return None
    return ("frontier", target_idx, path[1:], goal_sims.get(target_idx))


# ---- Breadcrumb encoding ------------------------------------------------- #

@torch.no_grad()
def encode_breadcrumb(
    world_model, render_scene, render_robot, render_act_dofs, cam,
    beacon, view_dist, planning_device, q0, gs, torch_mod,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and projected latents of the goal view."""
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

    proprio_t = None
    if world_model.encoder.use_proprio:
        proprio_t = canonical_breadcrumb_proprio(q0, planning_device, torch_mod)

    z_raw, z_proj = world_model.encode(vis_t, proprio_t)
    return z_raw.squeeze(0).detach(), z_proj.squeeze(0).detach()


# ---- PPO execution ------------------------------------------------------- #

@torch.no_grad()
def execute_command(
    scene, robot, policy, act_dofs, q0, prev_action,
    nominal_cmd, sim_cfg, gs, torch_mod, macro_action_repeat=1,
    command_representation: str = "mean_scaled",
    command_block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    nominal_cmd = nominal_cmd.to(device=gs.device, dtype=torch_mod.float32).flatten()
    last_action = prev_action.detach().clone()
    if command_representation == "active_block":
        block_size = int(command_block_size if command_block_size is not None else max(1, int(macro_action_repeat)))
        if nominal_cmd.numel() != block_size * 3:
            raise ValueError(
                f"Expected active_block command with {block_size * 3} values, got {nominal_cmd.numel()}"
            )
        cmd_seq = nominal_cmd.view(block_size, 3)
    else:
        cmd = nominal_cmd.view(1, 3)
        repeats = max(1, int(macro_action_repeat))
        cmd_seq = cmd.expand(repeats, -1)

    for step_cmd in cmd_seq:
        cmd = step_cmd.view(1, 3)
        proprio, _, _ = collect_proprio(robot, act_dofs, q0, last_action)
        obs_tensor = torch.cat([proprio, cmd], dim=1)
        actions = policy.act_deterministic(obs_tensor)

        q_tgt = q0.unsqueeze(0) + sim_cfg.action_scale * actions
        q_tgt[:, 0:4] = torch_mod.clamp(q_tgt[:, 0:4], -0.8, 0.8)
        q_tgt[:, 4:8] = torch_mod.clamp(q_tgt[:, 4:8], -1.5, 1.5)
        q_tgt[:, 8:12] = torch_mod.clamp(q_tgt[:, 8:12], -2.5, -0.5)

        robot.control_dofs_position(q_tgt, act_dofs)
        for _ in range(sim_cfg.decimation):
            scene.step()
        last_action = actions.detach().clone()

    return nominal_cmd.detach().clone(), last_action


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

def compute_topdown_half_extent(obstacle_layout, path_xy, pad=0.3):
    max_ext = 0.5
    for obs in obstacle_layout.obstacles:
        max_ext = max(max_ext,
                      abs(float(obs.pos[0])) + 0.5 * float(obs.size[0]),
                      abs(float(obs.pos[1])) + 0.5 * float(obs.size[1]))
    for xy in path_xy:
        max_ext = max(max_ext, abs(float(xy[0])), abs(float(xy[1])))
    return max_ext + pad

def draw_topdown_trajectory(out_path, obstacle_layout, beacon_layout, path_xy, breadcrumb_xy):
    size = 900
    half = compute_topdown_half_extent(obstacle_layout, path_xy)

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

def densify_path_xy(path_xy, step_m=0.04):
    if not path_xy:
        return []
    dense = [np.asarray(path_xy[0], dtype=np.float32)]
    for prev_xy, cur_xy in zip(path_xy, path_xy[1:]):
        prev = np.asarray(prev_xy, dtype=np.float32)
        cur = np.asarray(cur_xy, dtype=np.float32)
        delta = cur - prev
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            continue
        n_steps = max(1, int(math.ceil(dist / step_m)))
        for i in range(1, n_steps + 1):
            alpha = float(i) / float(n_steps)
            dense.append(prev + alpha * delta)
    return dense

def make_coverage_tracker(
    obstacle_layout,
    size=900,
    coverage_sigma_m=0.12,
    coverage_radius_m=0.30,
    coverage_threshold=0.20,
):
    half = compute_topdown_half_extent(obstacle_layout, [])
    meters_per_px = (2.0 * half) / float(size - 1)
    area_per_px = meters_per_px * meters_per_px
    scale = (size - 1) / (2.0 * half)

    sigma_px = max(1.0, coverage_sigma_m / meters_per_px)
    radius_px = max(1, int(math.ceil(coverage_radius_m / meters_per_px)))
    grid = np.arange(-radius_px, radius_px + 1, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    dist_sq = xx * xx + yy * yy
    kernel = np.exp(-0.5 * dist_sq / (sigma_px * sigma_px))
    kernel[dist_sq > float(radius_px * radius_px)] = 0.0

    return {
        "size": int(size),
        "half": float(half),
        "scale": float(scale),
        "area_per_px": float(area_per_px),
        "coverage_sigma_m": float(coverage_sigma_m),
        "coverage_radius_m": float(coverage_radius_m),
        "coverage_threshold": float(coverage_threshold),
        "radius_px": int(radius_px),
        "kernel": kernel,
        "coverage": np.zeros((size, size), dtype=np.float32),
        "covered_px": 0,
    }

def stamp_coverage_tracker(tracker, xy):
    size = int(tracker["size"])
    half = float(tracker["half"])
    scale = float(tracker["scale"])
    radius_px = int(tracker["radius_px"])
    kernel = tracker["kernel"]
    threshold = float(tracker["coverage_threshold"])
    coverage = tracker["coverage"]

    cx_f = (float(xy[0]) + half) * scale
    cy_f = (half - float(xy[1])) * scale
    cx = int(round(cx_f))
    cy = int(round(cy_f))
    x0 = max(0, cx - radius_px)
    x1 = min(size, cx + radius_px + 1)
    y0 = max(0, cy - radius_px)
    y1 = min(size, cy + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return
    kx0 = x0 - (cx - radius_px)
    kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (cy - radius_px)
    ky1 = ky0 + (y1 - y0)
    prev = coverage[y0:y1, x0:x1]
    prev_mask = prev >= threshold
    updated = np.maximum(prev, kernel[ky0:ky1, kx0:kx1])
    new_mask = updated >= threshold
    tracker["covered_px"] += int(np.count_nonzero(new_mask & (~prev_mask)))
    coverage[y0:y1, x0:x1] = updated

def update_coverage_tracker(tracker, prev_xy, cur_xy):
    segment = [cur_xy] if prev_xy is None else [prev_xy, cur_xy]
    dense = densify_path_xy(segment)
    if prev_xy is not None and dense:
        dense = dense[1:]
    for xy in dense:
        stamp_coverage_tracker(tracker, xy)

def coverage_tracker_metrics(tracker):
    coverage = tracker["coverage"]
    soft_coverage_area_m2 = float(tracker["covered_px"]) * float(tracker["area_per_px"])
    return {
        "coverage_sigma_m": float(tracker["coverage_sigma_m"]),
        "coverage_radius_m": float(tracker["coverage_radius_m"]),
        "coverage_threshold": float(tracker["coverage_threshold"]),
        "soft_coverage_area_m2": soft_coverage_area_m2,
        "soft_coverage_peak": float(coverage.max()) if coverage.size else 0.0,
        "soft_coverage_mean": float(coverage.mean()) if coverage.size else 0.0,
    }

def render_coverage_tracker(
    out_path,
    tracker,
    obstacle_layout,
    beacon_layout,
    path_xy,
    breadcrumb_xy,
):
    size = int(tracker["size"])
    half = float(tracker["half"])
    scale = float(tracker["scale"])
    coverage = tracker["coverage"]

    def w2c(x, y):
        return (x + half) * scale, (half - y) * scale

    bg = np.full((size, size, 3), (250, 248, 242), dtype=np.float32)
    alpha = np.clip(np.power(coverage, 0.75), 0.0, 1.0) * 0.82
    heat_color = np.zeros_like(bg)
    heat_color[..., 0] = 72.0
    heat_color[..., 1] = 195.0
    heat_color[..., 2] = 176.0
    canvas = bg * (1.0 - alpha[..., None]) + heat_color * alpha[..., None]
    img = Image.fromarray(np.clip(canvas, 0.0, 255.0).astype(np.uint8))
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
        draw.line(pts, fill=(30, 110, 210), width=3)

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

def draw_topdown_coverage(
    out_path,
    obstacle_layout,
    beacon_layout,
    path_xy,
    breadcrumb_xy,
    coverage_sigma_m=0.12,
    coverage_radius_m=0.30,
    coverage_threshold=0.20,
):
    tracker = make_coverage_tracker(
        obstacle_layout,
        size=900,
        coverage_sigma_m=coverage_sigma_m,
        coverage_radius_m=coverage_radius_m,
        coverage_threshold=coverage_threshold,
    )
    prev_xy = None
    for xy in path_xy:
        update_coverage_tracker(tracker, prev_xy, xy)
        prev_xy = xy
    render_coverage_tracker(
        out_path, tracker, obstacle_layout, beacon_layout, path_xy, breadcrumb_xy,
    )
    return coverage_tracker_metrics(tracker)

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
        args.out_dir = os.path.join("inference_runs", f"perception_only_seed_{args.seed:04d}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planning_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    world_model, wm_meta = load_world_model(args.wm_ckpt, planning_device)
    planner_heads, scorer_meta = load_planner_heads(
        args.scorer_ckpt, planning_device, wm_meta, args.macro_action_repeat,
    )
    if args.exploration_weight is not None:
        planner_heads.exploration_weight = float(args.exploration_weight)
    camera_cfg = ego_camera_config_from_args(args)

    print(
        f"Loaded world model: latent_dim={wm_meta['latent_dim']} "
        f"image_size={wm_meta['image_size']} "
        f"cmd_dim={wm_meta['cmd_dim']} "
        f"cmd_repr={wm_meta['command_representation']}"
    )
    if wm_meta["use_proprio"]:
        if not args.allow_mixed_latent_wm:
            raise ValueError(
                "scripts/6_infer_pure_wm.py requires a pure-vision world-model checkpoint, "
                "but the supplied checkpoint was trained with proprio enabled. "
                "Retrain LeWM without --use_proprio, or pass --allow_mixed_latent_wm "
                "only for legacy debugging."
            )
        print(
            "Warning: allowing a mixed latent world model in pure-perception inference. "
            "Exploration and safety will be evaluated in a vision+proprio latent, not "
            "pure perception."
        )
    if wm_meta["command_representation"] == "active_block":
        if int(args.macro_action_repeat) != int(wm_meta["command_block_size"]):
            raise ValueError(
                f"Active-block checkpoint expects --macro_action_repeat={wm_meta['command_block_size']}, "
                f"got {args.macro_action_repeat}."
            )
    if args.scorer_ckpt is not None:
        print(
            "Loaded planner heads: "
            f"safety={scorer_meta.get('has_safety_head', False)} "
            f"goal={scorer_meta.get('has_goal_head', False)} "
            f"exploration={scorer_meta.get('has_exploration', False)} "
            f"place={scorer_meta.get('has_place_head', False)} "
            f"(seq={scorer_meta.get('seq_len')}, "
            f"stride={scorer_meta.get('temporal_stride')}, "
            f"block={scorer_meta.get('action_block_size')}, "
            f"use_proprio={scorer_meta.get('use_proprio')})"
        )
        if (
            scorer_meta.get("safety_latent_source")
            or scorer_meta.get("exploration_latent_source")
            or scorer_meta.get("place_latent_source")
        ):
            print(
                "Scorer latent sources: "
                f"safety={scorer_meta.get('safety_latent_source', 'unknown')} "
                f"exploration={scorer_meta.get('exploration_latent_source', 'unknown')} "
                f"place={scorer_meta.get('place_latent_source', 'unknown')} "
                f"goal={scorer_meta.get('goal_latent_source', 'unknown')}"
            )
        if scorer_meta.get("has_place_head", False) and scorer_meta.get("place_supervision") is not None:
            print(f"Place supervision: {scorer_meta.get('place_supervision')}")
        if scorer_meta.get("has_goal_head", False):
            print("Ignoring goal head in scorer checkpoint; the active planner optimizes safety + novelty only.")
    if args.score_space != "proj":
        print(
            f"Ignoring --score_space={args.score_space!r}; "
            "the active planner always scores projected latents."
        )
    if scorer_meta.get("has_progress_head_ckpt", False):
        print("Ignoring legacy progress head stored in scorer checkpoint.")

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
        if target_beacon is None:
            available = ", ".join(sorted(b.identity for b in beacon_layout.beacons)) or "<none>"
            raise ValueError(
                f"Requested --target_beacon={args.target_beacon!r}, but generated maze "
                f"contains: {available}"
            )
    if target_beacon is None and beacon_layout.beacons:
        target_beacon = sorted(beacon_layout.beacons, key=lambda b: b.identity)[0]

    spawn_yaw = 0.0

    target_claim_xy = beacon_claim_xy(target_beacon, args.wall_thickness) if target_beacon else None
    breadcrumb_xy = [float(target_beacon.pos[0]), float(target_beacon.pos[1])] if target_beacon else None

    print(f"Maze: {args.grid_rows}x{args.grid_cols}, spawn=({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f})")
    if target_beacon:
        print(f"Target: {target_beacon.identity} at ({target_beacon.pos[0]:.2f}, {target_beacon.pos[1]:.2f})")
    objective_name = (
        f"safety+novelty[{args.exploration_bonus_mode}]"
        + ("+terminal_disp" if args.terminal_displacement_weight > 0.0 else "")
        + ("+safety_gate" if args.exploration_safety_gate_threshold is not None else "")
    )
    visited_nn_desc = ""
    if args.exploration_bonus_mode == "visited_nn":
        place_desc = ""
        if scorer_meta.get("has_place_head", False):
            place_desc = (
                f", PlaceSnippet={scorer_meta.get('place_snippet_len', '?')}"
                f", PlaceDim={scorer_meta.get('place_embedding_dim', '?')}"
            )
        visited_nn_desc = (
            f", VisitK={args.visited_nn_k}, "
            f"VisitMargin={args.visited_nn_margin:.3f}, "
            f"VisitTail={args.visited_nn_tail_steps}, "
            f"VisitBank={args.visited_bank_size}, "
            f"RevisitW={args.revisit_penalty_weight:.3f}"
            f"{place_desc}"
        )
    print(f"Planning: H={args.plan_horizon}, N={args.n_candidates}, "
          f"iters={args.cem_iters}, K={args.mpc_execute}, Macro={args.macro_action_repeat}, "
          f"Cmd={wm_meta['command_representation']}:{wm_meta['cmd_dim']}, "
          f"Latent=proj, Objective={objective_name}, "
          f"GoalSucc={args.success_goal_sim_threshold:.2f}/{args.success_goal_raw_sim_threshold:.2f}, "
          f"ExploreW={planner_heads.exploration_weight:.3f}, "
          f"ExploreMode={args.exploration_bonus_mode}, "
          f"DispW={args.terminal_displacement_weight:.3f}, "
          f"GateTau={args.exploration_safety_gate_threshold if args.exploration_safety_gate_threshold is not None else 'off'}, "
          f"GateK={args.exploration_safety_gate_sharpness:.3f}"
          f"{visited_nn_desc}, "
          f"ActionPen={args.action_penalty_weight:.4f}")

    t0 = time.time()
    physics_scene = ego_scene = third_person_scene = None

    ego_frames_hwc: List[np.ndarray] = []
    tp_frames_hwc: List[np.ndarray] = []
    combined_frames: List[np.ndarray] = []
    path_xy: List[List[float]] = []
    oracle_path_xy: List[List[float]] = []
    costs_log: List[float] = []
    cmds_log: List[List[float]] = []
    mode_log: List[str] = []
    goal_similarity_log: List[float | None] = []
    goal_similarity_raw_log: List[float | None] = []
    terminate_reason = "max_steps"
    collision_count = 0
    frame_substitution_count = 0
    first_collision_step: int | None = None
    success_hold_count = 0
    perceptual_goal_detected = False
    perceptual_goal_detected_step: int | None = None
    oracle_goal_reached = False
    min_goal_dist_m = float("inf")
    coverage_tracker = make_coverage_tracker(obstacle_layout)
    last_logged_coverage_area_m2 = 0.0
    goal_activation_step: int | None = None
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

        z_breadcrumb_raw_ref = None
        z_breadcrumb_proj_ref = None
        if target_beacon is not None:
            z_breadcrumb_raw, z_breadcrumb_proj = encode_breadcrumb(
                world_model, ego_scene, ego_robot, ego_act_dofs, ego_cam,
                target_beacon, args.breadcrumb_view_dist,
                planning_device, q0, gs, torch,
            )
            z_breadcrumb_raw_ref = z_breadcrumb_raw.detach().clone()
            z_breadcrumb_proj_ref = z_breadcrumb_proj.detach().clone()
            print(
                f"Goal latents encoded: ||z_raw||={float(z_breadcrumb_raw.norm()):.3f} "
                f"||z_proj||={float(z_breadcrumb_proj.norm()):.3f}"
            )

        reset_robot(physics_robot, physics_act_dofs, q0, spawn_xy, spawn_yaw, gs, torch)
        cmd_low_t, cmd_high_t, init_std_t, min_std_t = build_command_sampling_config(args, wm_meta)

        planner = PureCEMPlanner(
            world_model=world_model,
            horizon=args.plan_horizon,
            n_candidates=args.n_candidates,
            cem_iters=args.cem_iters,
            elite_frac=args.elite_frac,
            cmd_low=cmd_low_t,
            cmd_high=cmd_high_t,
            init_std=init_std_t,
            min_std=min_std_t,
            device=planning_device,
            action_penalty_weight=args.action_penalty_weight,
        )

        prev_action = torch.zeros((1, 12), device=gs.device, dtype=torch.float32)
        last_clean_frame: np.ndarray | None = None
        plan_seq: torch.Tensor | None = None
        plan_step_idx = 0
        plan_metrics_last: dict[str, float] = {}
        visited_rollout_bank: list[torch.Tensor] = []
        plan_audit_records: list[dict[str, Any]] = []
        pending_plan_audit: dict[str, Any] | None = None
        replan_count = 0
        if args.audit_every < 1:
            raise ValueError("--audit_every must be >= 1.")
        if args.audit_topk < 1:
            raise ValueError("--audit_topk must be >= 1.")

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
        start_xy = [float(obs["pos_np"][0]), float(obs["pos_np"][1])]
        path_xy.append(start_xy)
        oracle_path_xy.append(start_xy.copy())
        update_coverage_tracker(coverage_tracker, None, path_xy[-1])

        if target_claim_xy is not None:
            min_goal_dist_m = min(
                min_goal_dist_m,
                float(np.linalg.norm(
                    np.asarray(obs["pos_np"][:2], dtype=np.float32) - target_claim_xy,
                )),
            )

        for step in range(args.steps):
            mode = "explore"
            mode_log.append(mode)

            need_replan = plan_seq is None or plan_step_idx >= args.mpc_execute
            if need_replan:
                audit_this_plan = bool(args.audit_plan and (replan_count % args.audit_every == 0))
                audit_start_cov_area = None
                audit_start_goal_dist = None
                if audit_this_plan:
                    audit_start_cov_area = coverage_tracker_metrics(
                        coverage_tracker,
                    ).get("soft_coverage_area_m2")
                    if target_claim_xy is not None:
                        audit_start_goal_dist = float(np.linalg.norm(
                            np.asarray(obs["pos_np"][:2], dtype=np.float32) - target_claim_xy,
                        ))
                plan_seq, cost, plan_metrics_last, plan_diagnostics = planner.plan(
                    obs["z_raw"],
                    z_start_proj=obs["z_proj"],
                    heads=planner_heads,
                    exploration_bonus_mode=args.exploration_bonus_mode,
                    terminal_displacement_weight=args.terminal_displacement_weight,
                    exploration_safety_gate_threshold=args.exploration_safety_gate_threshold,
                    exploration_safety_gate_sharpness=args.exploration_safety_gate_sharpness,
                    visited_rollout_bank=visited_rollout_bank,
                    visited_rollout_knn_k=args.visited_nn_k,
                    visited_rollout_margin=args.visited_nn_margin,
                    visited_rollout_tail_steps=args.visited_nn_tail_steps,
                    visited_revisit_penalty_weight=args.revisit_penalty_weight,
                    return_diagnostics=audit_this_plan,
                    diagnostics_topk=args.audit_topk,
                )
                plan_step_idx = 0
                costs_log.append(float(cost))
                if audit_this_plan:
                    pending_plan_audit = {
                        "audit_version": 1,
                        "step": int(step),
                        "replan_index": int(replan_count),
                        "mode": mode,
                        "state_before": {
                            "pos_xy": [float(obs["pos_np"][0]), float(obs["pos_np"][1])],
                            "yaw_rad": float(obs["yaw_rad"]),
                            "goal_dist_m": audit_start_goal_dist,
                            "coverage_area_m2": (
                                None if audit_start_cov_area is None else float(audit_start_cov_area)
                            ),
                            "visited_bank_size": int(len(visited_rollout_bank)),
                        },
                        "plan": plan_diagnostics,
                    }
                else:
                    pending_plan_audit = None
                replan_count += 1

            nominal_cmd = plan_seq[plan_step_idx]
            plan_step_idx += 1
            cmd_vals = [float(v) for v in nominal_cmd.cpu().tolist()]
            cmd_display = format_command_for_log(cmd_vals, wm_meta)
            cmds_log.append(cmd_vals)
            executed_rollout_raw = None
            executed_rollout_proj = None
            if (
                pending_plan_audit is not None
                or args.exploration_bonus_mode == "visited_nn"
                or (
                    args.rnd_online_lr > 0.0
                    and planner_heads.exploration is not None
                    and planner_heads.exploration_weight > 0.0
                )
            ):
                executed_rollout_raw, executed_rollout_proj = teacher_forced_rollout_latent_pair(
                    world_model,
                    obs["z_raw"],
                    nominal_cmd.to(planning_device),
                )
            if (
                args.exploration_bonus_mode == "visited_nn"
                and executed_rollout_proj is None
            ):
                raise RuntimeError("visited_nn planning expected executed_rollout_proj to be available.")

            if pending_plan_audit is not None and executed_rollout_raw is not None and executed_rollout_proj is not None:
                pending_plan_audit["predicted_after_first_command"] = {
                    "raw_norm": float(executed_rollout_raw.squeeze(0).norm().item()),
                    "proj_norm": float(executed_rollout_proj.squeeze(0).norm().item()),
                    "goal_similarity_proj": (
                        None
                        if z_breadcrumb_proj_ref is None
                        else float(cosine_similarity_scalar(
                            executed_rollout_proj.squeeze(0),
                            z_breadcrumb_proj_ref,
                        ))
                    ),
                }

            _, actions = execute_command(
                physics_scene, physics_robot, policy,
                physics_act_dofs, q0, prev_action,
                nominal_cmd.to(gs.device), sim_cfg, gs, torch,
                macro_action_repeat=args.macro_action_repeat,
                command_representation=str(wm_meta["command_representation"]),
                command_block_size=int(wm_meta["command_block_size"]),
            )
            prev_action = actions.detach().clone()

            obs = observe(
                physics_robot, physics_act_dofs,
                ego_robot, ego_act_dofs, ego_cam,
                obstacle_layout, camera_cfg,
                world_model, planning_device,
                q0, prev_action, last_clean_frame,
            )
            goal_sim_now = None
            goal_sim_raw_now = None
            if z_breadcrumb_proj_ref is not None:
                goal_sim_now = cosine_similarity_scalar(
                    obs["z_proj"].squeeze(0),
                    z_breadcrumb_proj_ref,
                )
            if z_breadcrumb_raw_ref is not None:
                goal_sim_raw_now = cosine_similarity_scalar(
                    obs["z_raw"].squeeze(0),
                    z_breadcrumb_raw_ref,
                )

            cur_xy = [float(obs["pos_np"][0]), float(obs["pos_np"][1])]
            prev_xy = path_xy[-1]
            path_xy.append(cur_xy)
            update_coverage_tracker(coverage_tracker, prev_xy, cur_xy)
            oracle_cur_xy = cur_xy.copy()
            oracle_path_xy.append(oracle_cur_xy)

            if obs["frame_substituted"]:
                frame_substitution_count += 1
            else:
                last_clean_frame = obs["frame_hwc"].copy()
                if (
                    executed_rollout_proj is not None
                    and args.rnd_online_lr > 0.0
                    and planner_heads.exploration is not None
                    and planner_heads.exploration_weight > 0.0
                    and args.exploration_bonus_mode != "visited_nn"
                ):
                    planner_heads.exploration.online_update(
                        executed_rollout_proj,
                        lr=args.rnd_online_lr,
                    )

            if executed_rollout_proj is not None and args.visited_bank_size > 0:
                visited_rollout_bank.append(executed_rollout_proj.squeeze(0).detach().clone())
                if len(visited_rollout_bank) > args.visited_bank_size:
                    del visited_rollout_bank[0]

            tp_frame = render_third_person_frame(
                physics_robot, physics_act_dofs,
                tp_robot, tp_act_dofs, tp_cam,
                args.chase_dist, args.chase_height, args.side_offset, args.lookahead,
            )
            ego_frames_hwc.append(obs["frame_hwc"])
            tp_frames_hwc.append(tp_frame)
            combined_frames.append(build_side_by_side_frame(obs["frame_hwc"], tp_frame))

            pos_xy_t = torch.from_numpy(
                np.asarray(obs["pos_np"][:2], dtype=np.float32),
            ).unsqueeze(0)
            collided = bool(detect_collisions(
                pos_xy_t, obstacle_layout, margin=sim_cfg.collision_margin,
            )[0].item())
            if collided:
                collision_count += 1
                if first_collision_step is None:
                    first_collision_step = step

            if pending_plan_audit is not None:
                audit_cov_area = coverage_tracker_metrics(
                    coverage_tracker,
                ).get("soft_coverage_area_m2")
                audit_goal_dist = None
                if target_claim_xy is not None:
                    audit_goal_dist = float(np.linalg.norm(
                        np.asarray(cur_xy, dtype=np.float32) - target_claim_xy,
                    ))
                start_state = pending_plan_audit["state_before"]
                start_goal_dist = start_state["goal_dist_m"]
                start_cov_area = start_state["coverage_area_m2"]
                pred_raw = executed_rollout_raw.squeeze(0) if executed_rollout_raw is not None else None
                pred_proj = executed_rollout_proj.squeeze(0) if executed_rollout_proj is not None else None
                act_raw = obs["z_raw"].squeeze(0)
                act_proj = obs["z_proj"].squeeze(0)
                prediction_error = None
                if pred_raw is not None and pred_proj is not None:
                    raw_delta = pred_raw - act_raw
                    proj_delta = pred_proj - act_proj
                    prediction_error = {
                        "raw_mse": float(raw_delta.square().mean().item()),
                        "raw_l2": float(raw_delta.norm().item()),
                        "raw_cosine": float(cosine_similarity_scalar(pred_raw, act_raw)),
                        "proj_mse": float(proj_delta.square().mean().item()),
                        "proj_l2": float(proj_delta.norm().item()),
                        "proj_cosine": float(cosine_similarity_scalar(pred_proj, act_proj)),
                    }
                pending_plan_audit["actual_after_first_command"] = {
                    "pos_xy": [float(cur_xy[0]), float(cur_xy[1])],
                    "yaw_rad": float(obs["yaw_rad"]),
                    "xy_displacement_m": float(math.hypot(
                        cur_xy[0] - start_state["pos_xy"][0],
                        cur_xy[1] - start_state["pos_xy"][1],
                    )),
                    "goal_dist_m": audit_goal_dist,
                    "goal_dist_delta_m": (
                        None
                        if start_goal_dist is None or audit_goal_dist is None
                        else float(audit_goal_dist - start_goal_dist)
                    ),
                    "coverage_area_m2": (
                        None if audit_cov_area is None else float(audit_cov_area)
                    ),
                    "coverage_gain_m2": (
                        None
                        if start_cov_area is None or audit_cov_area is None
                        else float(audit_cov_area - start_cov_area)
                    ),
                    "collision": bool(collided),
                    "collision_count": int(collision_count),
                    "goal_similarity_proj": (
                        None if goal_sim_now is None else float(goal_sim_now)
                    ),
                    "goal_similarity_raw": (
                        None if goal_sim_raw_now is None else float(goal_sim_raw_now)
                    ),
                    "visited_bank_size_after": int(len(visited_rollout_bank)),
                }
                if prediction_error is not None:
                    pending_plan_audit["prediction_error"] = prediction_error
                plan_audit_records.append(pending_plan_audit)
                pending_plan_audit = None

            if float(obs["proprio"][0, 0].item()) < sim_cfg.min_z:
                terminate_reason = "fallen"
                print(f"Step {step:03d} | fallen")
                break

            reached = False
            detected = False
            proj_ok = (
                goal_sim_now is not None
                and goal_sim_now >= args.success_goal_sim_threshold
            )
            raw_ok = (
                goal_sim_raw_now is not None
                and goal_sim_raw_now >= args.success_goal_raw_sim_threshold
            )
            if proj_ok and raw_ok:
                success_hold_count += 1
            else:
                success_hold_count = 0
            goal_similarity_log.append(None if goal_sim_now is None else float(goal_sim_now))
            goal_similarity_raw_log.append(
                None if goal_sim_raw_now is None else float(goal_sim_raw_now)
            )
            if success_hold_count >= args.success_hold_steps:
                detected = True
                if not perceptual_goal_detected:
                    perceptual_goal_detected = True
                    perceptual_goal_detected_step = step

            dist_claim = None
            if target_beacon is not None and target_claim_xy is not None:
                dist_claim = float(np.linalg.norm(
                    np.asarray(oracle_cur_xy, dtype=np.float32) - target_claim_xy,
                ))
                min_goal_dist_m = min(min_goal_dist_m, dist_claim)
                frontness = float(np.dot(
                    np.asarray(oracle_cur_xy, dtype=np.float32) - np.array(target_beacon.pos[:2], dtype=np.float32),
                    np.array(target_beacon.normal[:2], dtype=np.float32),
                ))
                los = has_line_of_sight(
                    np.asarray(oracle_cur_xy, dtype=np.float32),
                    target_claim_xy, obstacle_layout,
                    step_size=0.02, margin=0.01,
                )
                oracle_goal_reached = oracle_goal_reached or (
                    dist_claim <= args.success_range and los and frontness > 0
                )
                reached = oracle_goal_reached

            if step % 20 == 0 or reached or detected:
                cov_area = coverage_tracker_metrics(coverage_tracker)["soft_coverage_area_m2"]
                cov_delta = cov_area - last_logged_coverage_area_m2
                last_logged_coverage_area_m2 = cov_area
                progress_parts = [
                    f"s={plan_metrics_last.get('safety_cost', 0.0):.3f} "
                    f"x={plan_metrics_last.get('exploration_bonus', 0.0):.3f}"
                ]
                if "revisit_penalty" in plan_metrics_last:
                    progress_parts.append(f"r={plan_metrics_last['revisit_penalty']:.3f}")
                if "terminal_displacement_bonus" in plan_metrics_last:
                    progress_parts.append(f"d={plan_metrics_last['terminal_displacement_bonus']:.3f}")
                if "exploration_safety_gate" in plan_metrics_last:
                    progress_parts.append(f"g={plan_metrics_last['exploration_safety_gate']:.3f}")
                progress_str = " ".join(progress_parts)
                goal_str = "n/a" if dist_claim is None else f"{dist_claim:.2f}m"
                if goal_sim_now is None:
                    goal_sim_str = "n/a"
                elif goal_sim_raw_now is None:
                    goal_sim_str = f"{goal_sim_now:.3f}"
                else:
                    goal_sim_str = f"{goal_sim_now:.3f}/{goal_sim_raw_now:.3f}"
                print(
                    f"Step {step:03d} | pos=({cur_xy[0]:+.2f}, {cur_xy[1]:+.2f}) "
                    f"cmd={cmd_display} "
                    f"cost={costs_log[-1]:.3f} goal_sim={goal_sim_str} d_goal={goal_str} "
                    f"cov={cov_area:.2f}m^2 cov+={cov_delta:+.2f} "
                    f"coll={collision_count} "
                    f"mode={mode} {progress_str}"
                )

            if reached:
                terminate_reason = "goal_reached"
                print(f"Step {step:03d} | REACHED {target_beacon.identity if target_beacon else 'goal'}")
                break
            if detected:
                terminate_reason = "goal_detected"
                print(f"Step {step:03d} | DETECTED {target_beacon.identity if target_beacon else 'goal'}")
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
    path_length_m = 0.0
    for prev_xy, cur_xy in zip(path_xy, path_xy[1:]):
        dx = float(cur_xy[0]) - float(prev_xy[0])
        dy = float(cur_xy[1]) - float(prev_xy[1])
        path_length_m += math.hypot(dx, dy)
    min_goal_dist_out = None if math.isinf(min_goal_dist_m) else min_goal_dist_m
    coverage_metrics = coverage_tracker_metrics(coverage_tracker) if path_xy else {}
    if path_xy:
        render_coverage_tracker(
            str(out_dir / "coverage_map.png"),
            coverage_tracker,
            obstacle_layout, beacon_layout, path_xy, breadcrumb_xy,
        )
    soft_coverage_area_m2 = coverage_metrics.get("soft_coverage_area_m2")
    soft_coverage_gain_per_m = None
    if soft_coverage_area_m2 is not None and path_length_m > 1e-6:
        soft_coverage_gain_per_m = soft_coverage_area_m2 / path_length_m
    plan_audit_summary: dict[str, Any] | None = None
    if plan_audit_records:
        pred_error_records = [
            rec["prediction_error"]
            for rec in plan_audit_records
            if "prediction_error" in rec
        ]
        collision_error_records = [
            rec["prediction_error"]
            for rec in plan_audit_records
            if rec.get("actual_after_first_command", {}).get("collision")
            and "prediction_error" in rec
        ]
        free_error_records = [
            rec["prediction_error"]
            for rec in plan_audit_records
            if not rec.get("actual_after_first_command", {}).get("collision")
            and "prediction_error" in rec
        ]

        def _mean_key(records: list[dict[str, float]], key: str) -> float | None:
            if not records:
                return None
            return float(sum(float(rec[key]) for rec in records) / len(records))

        plan_audit_summary = {
            "records": len(plan_audit_records),
            "prediction_error_mean": {
                "raw_mse": _mean_key(pred_error_records, "raw_mse"),
                "raw_l2": _mean_key(pred_error_records, "raw_l2"),
                "raw_cosine": _mean_key(pred_error_records, "raw_cosine"),
                "proj_mse": _mean_key(pred_error_records, "proj_mse"),
                "proj_l2": _mean_key(pred_error_records, "proj_l2"),
                "proj_cosine": _mean_key(pred_error_records, "proj_cosine"),
            },
            "prediction_error_collision": {
                "count": len(collision_error_records),
                "raw_mse": _mean_key(collision_error_records, "raw_mse"),
                "proj_mse": _mean_key(collision_error_records, "proj_mse"),
                "proj_cosine": _mean_key(collision_error_records, "proj_cosine"),
            },
            "prediction_error_noncollision": {
                "count": len(free_error_records),
                "raw_mse": _mean_key(free_error_records, "raw_mse"),
                "proj_mse": _mean_key(free_error_records, "proj_mse"),
                "proj_cosine": _mean_key(free_error_records, "proj_cosine"),
            },
        }

    summary = {
        "approach": "pure_world_model_perception_only",
        "seed": args.seed,
        "grid": f"{args.grid_rows}x{args.grid_cols}",
        "target": target_beacon.identity if target_beacon else None,
        "result": terminate_reason,
        "perceptual_goal_detected": perceptual_goal_detected,
        "perceptual_goal_detected_step": perceptual_goal_detected_step,
        "oracle_goal_reached": oracle_goal_reached,
        "steps": len(cmds_log),
        "low_level_control_steps": len(cmds_log) * (
            int(wm_meta["command_block_size"])
            if wm_meta["command_representation"] == "active_block"
            else max(1, args.macro_action_repeat)
        ),
        "oracle_collisions": collision_count,
        "first_collision_step": first_collision_step,
        "frame_substitutions": frame_substitution_count,
        "min_goal_dist_m": min_goal_dist_out,
        "path_length_m": path_length_m,
        "soft_coverage_area_m2": soft_coverage_area_m2,
        "soft_coverage_gain_per_m": soft_coverage_gain_per_m,
        "elapsed_sec": elapsed,
        "planner": {
            "horizon": args.plan_horizon,
            "candidates": args.n_candidates,
            "cem_iters": args.cem_iters,
            "elite_frac": args.elite_frac,
            "mpc_execute_k": args.mpc_execute,
            "macro_action_repeat": args.macro_action_repeat,
            "latent_space": "proj",
            "objective": (
                f"safety_plus_novelty_{args.exploration_bonus_mode}"
                + ("_plus_terminal_displacement" if args.terminal_displacement_weight > 0.0 else "")
                + ("_plus_safety_gated_novelty" if args.exploration_safety_gate_threshold is not None else "")
            ),
            "world_model_uses_proprio": bool(wm_meta["use_proprio"]),
            "world_model_cmd_dim": int(wm_meta["cmd_dim"]),
            "world_model_command_representation": str(wm_meta["command_representation"]),
            "world_model_command_block_size": int(wm_meta["command_block_size"]),
            "world_model_command_latency": int(wm_meta["command_latency"]),
            "action_penalty_weight": args.action_penalty_weight,
            "success_goal_sim_threshold": args.success_goal_sim_threshold,
            "success_goal_raw_sim_threshold": args.success_goal_raw_sim_threshold,
            "success_hold_steps": args.success_hold_steps,
            "goal_activation_step": goal_activation_step,
            "uses_safety_head": planner_heads.safety_head is not None,
            "uses_goal_head": False,
            "goal_head_present_ckpt": planner_heads.goal_head is not None,
            "uses_exploration": planner_heads.exploration is not None,
            "exploration_weight": planner_heads.exploration_weight,
            "exploration_bonus_mode": args.exploration_bonus_mode,
            "visited_nn_k": args.visited_nn_k,
            "visited_nn_margin": args.visited_nn_margin,
            "visited_nn_tail_steps": args.visited_nn_tail_steps,
            "visited_bank_size": args.visited_bank_size,
            "revisit_penalty_weight": args.revisit_penalty_weight,
            "audit_plan": bool(args.audit_plan),
            "audit_every": args.audit_every,
            "audit_topk": args.audit_topk,
            "terminal_displacement_weight": args.terminal_displacement_weight,
            "exploration_safety_gate_threshold": args.exploration_safety_gate_threshold,
            "exploration_safety_gate_sharpness": args.exploration_safety_gate_sharpness,
            "rnd_online_lr": args.rnd_online_lr,
            "ignored_goal_weight_ckpt": planner_heads.goal_weight,
            "ignored_recent_latent_window": args.recent_latent_window,
            "ignored_revisit_penalty_weight": args.revisit_penalty_weight,
            "ignored_forward_bonus_weight": args.forward_bonus_weight,
            "ignored_forward_bonus_safety_threshold": args.forward_bonus_safety_threshold,
            "ignored_current_safety_brake_threshold": args.current_safety_brake_threshold,
            "ignored_unsafe_macro_action_repeat": args.unsafe_macro_action_repeat,
            "ignored_goal_seek_sim_threshold": args.goal_direct_sim_threshold,
            "ignored_goal_activation_hold_steps": args.goal_activation_hold_steps,
            "ignores_progress_head_ckpt": scorer_meta.get("has_progress_head_ckpt", False),
            "auto_scaled_defaults": getattr(args, "auto_scaled_defaults", {}),
            "planner_heads": scorer_meta,
        },
        "coverage": coverage_metrics,
        "path_xy": path_xy,
        "oracle_path_xy": oracle_path_xy,
        "modes": mode_log,
        "goal_similarity": goal_similarity_log,
        "goal_similarity_raw": goal_similarity_raw_log,
        "commands": cmds_log,
        "costs": costs_log,
        "plan_audit": {
            "enabled": bool(args.audit_plan),
            "records": len(plan_audit_records),
            "file": "plan_audit.jsonl" if plan_audit_records else None,
            "summary": plan_audit_summary,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if plan_audit_records:
        with open(out_dir / "plan_audit.jsonl", "w") as f:
            for record in plan_audit_records:
                f.write(json.dumps(record) + "\n")

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
          f"oracle_collisions={collision_count} | time={elapsed:.1f}s")
    print(f"Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
