#!/usr/bin/env python3
"""Demo: verify that wall clipping is eliminated in the data pipeline.

This script places the robot at various distances and angles relative to a
wall, renders the egocentric camera view, and reports whether the camera
sees through the wall or correctly renders its surface.

Four layers of protection are tested:
  1. Camera placed at the actual front camera mount
  2. Small near-plane so nearby wall surfaces are still rendered
  3. Frustum-aware 9-ray clipping detector (catches off-axis clipping)
  4. Camera retraction pulls the camera back when frustum check detects danger

For cases that would have clipped, the script also renders the retracted
camera position and saves both original and retracted frames for comparison.

Usage:
    python scripts/demo_no_clipping.py [--save_dir clip_test_output]

Output:
    Per-test PNG renders + a summary table showing PASS/CAUGHT/RETRACTED.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image

from lewm.math_utils import yaw_to_quat
from lewm.obstacle_utils import ObstacleSpec, ObstacleLayout
from lewm.camera_utils import (
    add_egocentric_camera_args,
    camera_rotation_matrix,
    camera_safety_metrics,
    depth_buffer_has_clipping,
    ego_camera_config_from_args,
    egocentric_camera_pose,
    quat_to_rotmat_wxyz,
    retract_camera_to_safe,
)


# --------------------------------------------------------------------------- #
# Camera / scene constants (must match 2_visual_renderer.py)
# --------------------------------------------------------------------------- #

IMG_RES = 224
WALL_THICKNESS = 0.20
WALL_HEIGHT = 0.30
URDF_PATH = "assets/mini_pupper/mini_pupper_render.urdf"
Q0 = [0.06, 0.06, -0.06, -0.06, 0.85, 0.85, 0.85, 0.85, -1.75, -1.75, -1.75, -1.75]


# --------------------------------------------------------------------------- #
# Test configurations
# --------------------------------------------------------------------------- #

def base_pose_for_camera_xy(
    desired_cam_xy: tuple[float, float],
    yaw_rad: float,
    base_z: float,
    camera_cfg,
) -> np.ndarray:
    """Solve for base position that places the camera at a desired XY point."""
    quat_np = yaw_to_quat(yaw_rad)
    body_rot = quat_to_rotmat_wxyz(quat_np)
    mount_world = body_rot @ np.asarray(camera_cfg.mount_pos_body, dtype=np.float32)
    return np.array([
        float(desired_cam_xy[0]) - float(mount_world[0]),
        float(desired_cam_xy[1]) - float(mount_world[1]),
        base_z,
    ], dtype=np.float32)


def build_test_cases(camera_cfg):
    """Generate robot poses at various distances/angles from a wall.

    Distances are specified in terms of camera standoff to keep the test
    meaningful regardless of where the camera is mounted on the robot body.
    """
    wall = ObstacleSpec(
        pos=(0.5, 0.0, WALL_HEIGHT / 2.0),
        size=(WALL_THICKNESS, 2.0, WALL_HEIGHT),
        color=(0.85, 0.15, 0.15),
    )
    layout = ObstacleLayout([wall])

    cases = []
    base_z = 0.10
    # Vary camera distance from wall surface
    # Wall front face is at x = 0.5 - WALL_THICKNESS/2 = 0.4
    wall_face_x = 0.5 - WALL_THICKNESS / 2.0

    for standoff in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.00, -0.02]:
        label = f"standoff_{standoff:+.2f}m"
        yaw_rad = 0.0
        cases.append({
            "name": f"{label}_facing_wall",
            "robot_pos": base_pose_for_camera_xy((wall_face_x - standoff, 0.0), yaw_rad, base_z, camera_cfg),
            "robot_yaw": yaw_rad,
            "standoff": standoff,
        })

    # Angled approaches
    for angle_deg in [30, 45, 60]:
        yaw_rad = math.radians(-angle_deg)
        cases.append({
            "name": f"angle_{angle_deg}deg_10cm",
            "robot_pos": base_pose_for_camera_xy((wall_face_x - 0.10, 0.3), yaw_rad, base_z, camera_cfg),
            "robot_yaw": yaw_rad,
            "standoff": 0.10,
        })

    # Parallel to wall (wall-following)
    for standoff in [0.15, 0.05]:
        yaw_rad = math.pi / 2.0
        cases.append({
            "name": f"parallel_{standoff:.2f}m",
            "robot_pos": base_pose_for_camera_xy((wall_face_x - standoff, 0.0), yaw_rad, base_z, camera_cfg),
            "robot_yaw": yaw_rad,  # facing +Y, parallel to wall
            "standoff": standoff,
        })

    return layout, cases


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_test_case(gs, torch, scene, cam, robot, act_dofs, q0, case, layout, camera_cfg):
    """Render a single test case and return frame + safety metrics + retraction info.

    Returns (rgb, safety, cam_pos, retracted_rgb, retract_dist, depth_clip).
    retracted_rgb is None if no retraction was needed/attempted.
    """
    pos = torch.tensor(case["robot_pos"], device=gs.device, dtype=torch.float32).unsqueeze(0)
    quat_np = yaw_to_quat(case["robot_yaw"])
    quat = torch.tensor(quat_np, device=gs.device, dtype=torch.float32).unsqueeze(0)

    robot.set_pos(pos)
    robot.set_quat(quat)
    robot.set_dofs_position(q0.unsqueeze(0), act_dofs)

    cam_pos, cam_lookat, cam_up, cam_forward = egocentric_camera_pose(case["robot_pos"], quat_np, camera_cfg)
    cam_rot = camera_rotation_matrix(quat_np, camera_cfg.pitch_rad)
    safety = camera_safety_metrics(cam_pos, cam_forward, layout, camera_cfg, cam_rot=cam_rot)

    # Render original (possibly clipped) frame
    cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=cam_up)
    render_out = cam.render(rgb=True, depth=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb, dtype=np.uint8)

    # Depth-buffer check
    depth_clip = False
    if len(render_out) > 1 and render_out[1] is not None:
        depth_buf = render_out[1]
        if hasattr(depth_buf, "cpu"):
            depth_buf = depth_buf.cpu().numpy()
        depth_clip = depth_buffer_has_clipping(np.asarray(depth_buf, dtype=np.float32), camera_cfg.near_plane)

    # If unsafe, attempt retraction and render the retracted view
    retracted_rgb = None
    retract_dist = 0.0
    if safety["unsafe"]:
        new_pos, new_lookat, new_up, new_fwd, retract_dist = retract_camera_to_safe(
            cam_pos, cam_forward, cam_up, cam_rot, layout, camera_cfg,
        )
        if retract_dist > 0:
            cam.set_pose(pos=new_pos, lookat=new_lookat, up=new_up)
            ret_out = cam.render(rgb=True, depth=True, force_render=True)
            ret_rgb = ret_out[0]
            if hasattr(ret_rgb, "cpu"):
                ret_rgb = ret_rgb.cpu().numpy()
            retracted_rgb = np.asarray(ret_rgb, dtype=np.uint8)

    return rgb, safety, cam_pos, retracted_rgb, retract_dist, depth_clip


def analyse_frame(rgb: np.ndarray) -> dict:
    """Analyse a rendered frame for signs of wall clipping.

    Clipped frames typically show:
      - Very high mean brightness (seeing the skybox/ground through geometry)
      - Very low variance (uniform color from inside the wall)
      - Unusual color distribution
    """
    img_f = rgb.astype(np.float32)
    mean_brightness = img_f.mean()
    std_brightness = img_f.std()

    # Count very dark pixels (inside-geometry artifact)
    dark_frac = (img_f.mean(axis=-1) < 10).mean()
    # Count very bright pixels (skybox bleed)
    bright_frac = (img_f.mean(axis=-1) > 245).mean()

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Verify wall-clipping elimination")
    parser.add_argument("--save_dir", type=str, default="clip_test_output",
                        help="Directory to save rendered test images.")
    parser.add_argument("--sim_backend", type=str, default="auto")
    add_egocentric_camera_args(parser)
    args = parser.parse_args()
    camera_cfg = ego_camera_config_from_args(args)

    os.makedirs(args.save_dir, exist_ok=True)

    import genesis as gs
    import torch
    from lewm.genesis_utils import init_genesis_once

    init_genesis_once(args.sim_backend)

    layout, cases = build_test_cases(camera_cfg)

    # Build scene
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55)),
    )

    for obs in layout.obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )

    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(IMG_RES, IMG_RES), fov=camera_cfg.fov_deg, near=camera_cfg.near_plane, GUI=False)
    scene.build(n_envs=1)

    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)
    q0 = torch.tensor(Q0, device=gs.device, dtype=torch.float32)

    # Run tests
    print(f"\n{'='*80}")
    print(f"  Wall Clipping Validation — {len(cases)} test configurations")
    print(f"  Wall thickness:     {WALL_THICKNESS:.2f}m")
    print(
        "  Camera mount xyz:  "
        f"({camera_cfg.mount_pos_body[0]:.3f}, {camera_cfg.mount_pos_body[1]:.3f}, {camera_cfg.mount_pos_body[2]:.3f})"
    )
    print(f"  Camera pitch:      {math.degrees(camera_cfg.pitch_rad):.1f}deg")
    print(f"  Camera near_plane: {camera_cfg.near_plane:.3f}m")
    print(f"{'='*80}\n")

    print(
        f"{'Test':<30s} {'Standoff':>10s} {'Clear':>10s} "
        f"{'FrustMin':>10s} {'Retract':>10s} {'DepthClp':>10s} {'Result':>10s}"
    )
    print("-" * 96)

    pass_count = 0
    retract_count = 0
    fail_count = 0

    for case in cases:
        rgb, safety, cam_pos, retracted_rgb, retract_dist, depth_clip = render_test_case(
            gs, torch, scene, cam, robot, act_dofs, q0, case, layout, camera_cfg,
        )
        stats = analyse_frame(rgb)

        # Save original frame
        img_path = os.path.join(args.save_dir, f"{case['name']}.png")
        Image.fromarray(rgb).save(img_path)

        # Save retracted frame if available
        if retracted_rgb is not None:
            ret_path = os.path.join(args.save_dir, f"{case['name']}_retracted.png")
            Image.fromarray(retracted_rgb).save(ret_path)
            ret_stats = analyse_frame(retracted_rgb)

        # Determine result
        if safety["unsafe"] and retracted_rgb is not None:
            # Retraction was applied — check if the retracted frame looks valid
            ret_stats = analyse_frame(retracted_rgb)
            if ret_stats["std_brightness"] < 5.0 and ret_stats["mean_brightness"] < 15.0:
                result = "FAIL"  # retraction didn't fix the visual
                fail_count += 1
            else:
                result = "RETRACTED"
                retract_count += 1
        elif safety["unsafe"]:
            result = "CAUGHT"
            fail_count += 1
        elif depth_clip:
            result = "DEPTH_FAIL"
            fail_count += 1
        elif stats["std_brightness"] < 5.0 and stats["mean_brightness"] < 15.0:
            result = "FAIL"
            fail_count += 1
        else:
            result = "PASS"
            pass_count += 1

        fmin = safety.get("frustum_min_hit", safety["forward_hit"])
        print(
            f"{case['name']:<30s} "
            f"{case['standoff']:>+10.2f} "
            f"{safety['clearance']:>10.3f} "
            f"{fmin:>10.3f} "
            f"{retract_dist:>10.3f} "
            f"{'yes' if depth_clip else 'no':>10s} "
            f"{result:>10s}"
        )

    print("-" * 96)
    print(
        f"\nResults: {pass_count} PASS, {retract_count} RETRACTED, "
        f"{fail_count} CAUGHT/FAIL out of {len(cases)} tests"
    )
    print(f"Images saved to: {os.path.abspath(args.save_dir)}/")

    print(
        f"\nAt standoff >= {camera_cfg.safe_clearance:.2f}m camera clearance, "
        f"the camera should remain outside the near-plane danger zone."
    )
    if retract_count > 0:
        print(
            f"{retract_count} case(s) were saved by camera retraction — "
            f"check the *_retracted.png files for visual quality."
        )

    scene.destroy()
    gs.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
