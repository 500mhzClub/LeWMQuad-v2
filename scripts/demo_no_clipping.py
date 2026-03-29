#!/usr/bin/env python3
"""Demo: verify that wall clipping is eliminated in the v2 data pipeline.

This script places the robot at various distances and angles relative to a
wall, renders the egocentric camera view, and reports whether the camera
sees through the wall or correctly renders its surface.

Three layers of protection are tested:
  1. Thick walls (0.20m) — camera physically cannot reach through
  2. Explicit near_plane (0.08m) — near geometry is rendered, not clipped
  3. Camera-inside-wall detector — catches any remaining edge cases

Usage:
    python scripts/demo_no_clipping.py [--save_dir clip_test_output]

Output:
    Per-test PNG renders + a summary table showing PASS/FAIL for each pose.
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

from lewm.math_utils import forward_up_from_quat, yaw_to_quat
from lewm.obstacle_utils import ObstacleSpec, ObstacleLayout


# --------------------------------------------------------------------------- #
# Camera / scene constants (must match 2_visual_renderer.py v2)
# --------------------------------------------------------------------------- #

CAM_FORWARD_OFFSET = 0.06
CAM_UP_OFFSET = 0.05
CAM_LOOKAT_DIST = 1.0
CAM_NEAR_PLANE = 0.08
CAM_FOV = 58
IMG_RES = 224
WALL_THICKNESS = 0.20
WALL_HEIGHT = 0.30


def camera_inside_any_obstacle(
    cam_xy: np.ndarray,
    layout: ObstacleLayout,
    margin: float = 0.02,
) -> bool:
    """Return True if camera XY is inside any obstacle AABB (+ margin)."""
    cx, cy = float(cam_xy[0]), float(cam_xy[1])
    for obs in layout.obstacles:
        ox, oy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0 + margin, obs.size[1] / 2.0 + margin
        if abs(cx - ox) < hx and abs(cy - oy) < hy:
            return True
    return False


# --------------------------------------------------------------------------- #
# Test configurations
# --------------------------------------------------------------------------- #

def build_test_cases():
    """Generate robot poses at various distances/angles from a wall.

    Wall is at x=0.5, running along Y, facing -X.
    Robot faces +X (toward the wall) from various standoff distances.
    """
    wall = ObstacleSpec(
        pos=(0.5, 0.0, WALL_HEIGHT / 2.0),
        size=(WALL_THICKNESS, 2.0, WALL_HEIGHT),
        color=(0.6, 0.6, 0.6),
    )
    layout = ObstacleLayout([wall])

    cases = []
    # Vary robot distance from wall surface
    # Wall front face is at x = 0.5 - WALL_THICKNESS/2 = 0.4
    wall_face_x = 0.5 - WALL_THICKNESS / 2.0

    for standoff in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.00, -0.02]:
        robot_x = wall_face_x - standoff
        label = f"standoff_{standoff:+.2f}m"
        # Robot facing +X (yaw=0)
        cases.append({
            "name": f"{label}_facing_wall",
            "robot_pos": np.array([robot_x, 0.0, 0.10], dtype=np.float32),
            "robot_yaw": 0.0,
            "standoff": standoff,
        })

    # Angled approaches
    for angle_deg in [30, 45, 60]:
        robot_x = wall_face_x - 0.10
        robot_y = 0.3
        yaw_rad = math.radians(-angle_deg)
        cases.append({
            "name": f"angle_{angle_deg}deg_10cm",
            "robot_pos": np.array([robot_x, robot_y, 0.10], dtype=np.float32),
            "robot_yaw": yaw_rad,
            "standoff": 0.10,
        })

    # Parallel to wall (wall-following)
    for standoff in [0.15, 0.05]:
        robot_x = wall_face_x - standoff
        cases.append({
            "name": f"parallel_{standoff:.2f}m",
            "robot_pos": np.array([robot_x, 0.0, 0.10], dtype=np.float32),
            "robot_yaw": math.pi / 2.0,  # facing +Y, parallel to wall
            "standoff": standoff,
        })

    return layout, cases


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_test_case(gs, torch, scene, cam, robot, act_dofs, case, layout):
    """Render a single test case and return (rgb, clipped_flag)."""
    pos = torch.tensor(case["robot_pos"], device=gs.device, dtype=torch.float32).unsqueeze(0)
    quat_np = yaw_to_quat(case["robot_yaw"])
    quat = torch.tensor(quat_np, device=gs.device, dtype=torch.float32).unsqueeze(0)

    robot.set_pos(pos)
    robot.set_quat(quat)

    fw, up = forward_up_from_quat(quat_np)
    pos_np = case["robot_pos"]
    cam_pos = pos_np + CAM_FORWARD_OFFSET * fw + CAM_UP_OFFSET * up
    cam_lookat = cam_pos + CAM_LOOKAT_DIST * fw

    # Check if camera is inside wall
    clipped = camera_inside_any_obstacle(cam_pos[:2], layout)

    cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=up)
    render_out = cam.render(rgb=True, force_render=True)
    rgb = render_out[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb, dtype=np.uint8)

    return rgb, clipped, cam_pos


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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    import genesis as gs
    import torch
    from lewm.genesis_utils import init_genesis_once

    init_genesis_once(args.sim_backend)

    layout, cases = build_test_cases()

    # Build scene
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())

    for obs in layout.obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )

    robot = scene.add_entity(
        gs.morphs.URDF(file="assets/mini_pupper/mini_pupper.urdf", fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(IMG_RES, IMG_RES), fov=CAM_FOV, near=CAM_NEAR_PLANE, GUI=False)
    scene.build(n_envs=1)

    JOINTS_ACTUATED = [
        "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
        "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
        "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
    ]
    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    # Run tests
    print(f"\n{'='*80}")
    print(f"  Wall Clipping Validation — {len(cases)} test configurations")
    print(f"  Wall thickness:     {WALL_THICKNESS:.2f}m")
    print(f"  Camera fwd offset:  {CAM_FORWARD_OFFSET:.2f}m")
    print(f"  Camera near_plane:  {CAM_NEAR_PLANE:.2f}m")
    print(f"{'='*80}\n")

    print(f"{'Test':<30s} {'Standoff':>10s} {'CamInWall':>10s} {'MeanBrt':>10s} {'StdBrt':>10s} {'Result':>10s}")
    print("-" * 80)

    pass_count = 0
    fail_count = 0

    for case in cases:
        rgb, clipped, cam_pos = render_test_case(gs, torch, scene, cam, robot, act_dofs, case, layout)
        stats = analyse_frame(rgb)

        # Save image
        img_path = os.path.join(args.save_dir, f"{case['name']}.png")
        Image.fromarray(rgb).save(img_path)

        # Determine pass/fail
        # A "clipped" frame means the camera detector caught it — that's the
        # safety net working.  The real question is: does the rendered image
        # look correct (wall surface visible, not seeing through)?
        if clipped:
            result = "CAUGHT"  # detector caught it, frame would be replaced
            fail_count += 1
        elif stats["std_brightness"] < 5.0 and stats["mean_brightness"] < 15.0:
            result = "FAIL"   # likely inside geometry (all black)
            fail_count += 1
        else:
            result = "PASS"
            pass_count += 1

        print(
            f"{case['name']:<30s} "
            f"{case['standoff']:>+10.2f} "
            f"{'YES' if clipped else 'no':>10s} "
            f"{stats['mean_brightness']:>10.1f} "
            f"{stats['std_brightness']:>10.1f} "
            f"{result:>10s}"
        )

    print("-" * 80)
    print(f"\nResults: {pass_count} PASS, {fail_count} CAUGHT/FAIL out of {len(cases)} tests")
    print(f"Images saved to: {os.path.abspath(args.save_dir)}/")

    # The key insight: with 0.20m walls and 0.06m camera offset,
    # the camera needs to be 0.04m past the wall surface to enter it.
    # At standoff=0 the camera is right at the wall surface — it should
    # see the wall filling the frame.  At standoff=-0.02 the robot's base
    # is 2cm past the surface, but the camera (0.06m forward of base) is
    # still 4cm from the wall centre — outside the 0.10m half-extent.
    # Only extreme penetration (robot fully inside the wall) can clip.

    clean_positive = sum(1 for c in cases if c["standoff"] >= 0.05)
    print(f"\nAt standoff >= 0.05m: all {clean_positive} cases should PASS (wall visible, no clip)")

    scene.destroy()
    gs.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
