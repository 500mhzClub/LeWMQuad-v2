#!/usr/bin/env python3
"""Summarize rollout/rendered dataset coverage before training.

Examples:
    python3 scripts/summarize_dataset_coverage.py --raw_dir jepa_raw_data
    python3 scripts/summarize_dataset_coverage.py --rendered_dir jepa_final_dataset
    python3 scripts/summarize_dataset_coverage.py \
        --raw_dir jepa_raw_data \
        --rendered_dir jepa_final_dataset \
        --seq_len 4 --temporal_stride 5 --action_block_size 5 --window_stride 5
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.command_utils import COMMAND_PATTERNS

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


EXTRA_CMD_PATTERN_NAMES = [
    "maze_teacher_beacon",
    "maze_teacher_frontier",
    "maze_teacher_explore",
]
CMD_PATTERN_NAMES = list(COMMAND_PATTERNS.keys()) + EXTRA_CMD_PATTERN_NAMES
MAZE_TEACHER_PATTERN_NAMES = {
    "maze_teacher_beacon",
    "maze_teacher_frontier",
    "maze_teacher_explore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw/rendered dataset coverage.")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Directory containing raw rollout chunks (chunk_*.npz).")
    parser.add_argument("--rendered_dir", type=str, default=None,
                        help="Directory containing rendered HDF5 chunks (*_rgb.h5).")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Optional training seq_len for valid-window estimation.")
    parser.add_argument("--temporal_stride", type=int, default=1,
                        help="Optional training temporal_stride for valid-window estimation.")
    parser.add_argument("--action_block_size", type=int, default=None,
                        help="Optional training action_block_size for valid-window estimation.")
    parser.add_argument("--window_stride", type=int, default=None,
                        help="Optional training window_stride for valid-window estimation.")
    parser.add_argument("--close_range_m", type=float, default=1.0,
                        help="Range threshold for a visible-beacon close-contact proxy.")
    parser.add_argument("--very_close_range_m", type=float, default=0.5,
                        help="Stricter visible-beacon close-contact proxy.")
    parser.add_argument("--top_scenes", type=int, default=10,
                        help="Number of scenes to print in the human-readable summary.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output path for the JSON report.")
    args = parser.parse_args()
    if args.raw_dir is None and args.rendered_dir is None:
        if Path("jepa_raw_data").is_dir():
            args.raw_dir = "jepa_raw_data"
        if h5py is not None and Path("jepa_final_dataset").is_dir():
            args.rendered_dir = "jepa_final_dataset"
    return args


def discover_raw_files(raw_dir: str | None) -> list[Path]:
    if raw_dir is None:
        return []
    return [Path(p) for p in sorted(glob.glob(os.path.join(raw_dir, "chunk_*.npz")))]


def discover_rendered_files(rendered_dir: str | None) -> list[Path]:
    if rendered_dir is None:
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in ("*_rgb.h5", "chunk_*.h5"):
        for path_str in sorted(glob.glob(os.path.join(rendered_dir, pattern))):
            path = Path(path_str)
            if "_tmp_" in path.name or path in seen:
                continue
            out.append(path)
            seen.add(path)
    return out


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = mean(values)
    assert mu is not None
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def summarize_numeric(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(stddev(values)),
        "min": float(min(values)),
        "p50": float(percentile(values, 0.50)),
        "p90": float(percentile(values, 0.90)),
        "max": float(max(values)),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def decode_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_text(value.item())
        return None
    return str(value)


def parse_scene_meta(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        raw_value = raw_value.item()
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    if isinstance(raw_value, dict):
        return raw_value
    return {}


def stable_scene_identity(scene_seed: Any, scene_type: Any, fallback: str) -> tuple[str, int | None, str]:
    seed = None
    if scene_seed is not None:
        try:
            seed = int(to_builtin(scene_seed))
        except (TypeError, ValueError):
            seed = None
    scene_type_text = decode_text(scene_type) or "unknown"
    if seed is not None:
        return f"{scene_type_text}:{seed}", seed, scene_type_text
    return fallback, None, scene_type_text


def command_pattern_name(pattern_idx: int) -> str:
    if 0 <= int(pattern_idx) < len(CMD_PATTERN_NAMES):
        return CMD_PATTERN_NAMES[int(pattern_idx)]
    return f"unknown_{int(pattern_idx)}"


def count_episode_proxies(
    dones: np.ndarray,
    beacon_visible: np.ndarray | None,
    beacon_range: np.ndarray | None,
    close_range_m: float,
    very_close_range_m: float,
) -> dict[str, int]:
    n_envs, steps = int(dones.shape[0]), int(dones.shape[1])
    visible_eps = 0
    close_eps = 0
    very_close_eps = 0
    total_eps = 0

    for env_idx in range(n_envs):
        start = 0
        for step_idx in range(steps):
            is_end = bool(dones[env_idx, step_idx]) or step_idx == steps - 1
            if not is_end:
                continue
            end = step_idx + 1
            total_eps += 1
            if beacon_visible is not None:
                visible_slice = np.asarray(beacon_visible[env_idx, start:end], dtype=bool)
                range_slice = (
                    np.asarray(beacon_range[env_idx, start:end], dtype=np.float32)
                    if beacon_range is not None
                    else None
                )
                if np.any(visible_slice):
                    visible_eps += 1
                if range_slice is not None:
                    finite_close = visible_slice & np.isfinite(range_slice) & (range_slice <= close_range_m)
                    finite_very_close = visible_slice & np.isfinite(range_slice) & (range_slice <= very_close_range_m)
                    if np.any(finite_close):
                        close_eps += 1
                    if np.any(finite_very_close):
                        very_close_eps += 1
            start = end
    return {
        "episodes": int(total_eps),
        "episodes_with_visible_beacon": int(visible_eps),
        "episodes_with_close_visible_beacon": int(close_eps),
        "episodes_with_very_close_visible_beacon": int(very_close_eps),
    }


def estimate_valid_windows(
    dones: np.ndarray,
    collisions: np.ndarray | None,
    seq_len: int,
    temporal_stride: int,
    action_block_size: int,
    window_stride: int,
) -> dict[str, int]:
    n_envs, steps = int(dones.shape[0]), int(dones.shape[1])
    raw_span = (seq_len - 1) * temporal_stride + action_block_size
    total = 0
    no_done = 0
    no_done_no_collision = 0
    if steps < raw_span:
        return {
            "raw_span": int(raw_span),
            "total_windows": 0,
            "windows_no_done": 0,
            "windows_no_done_no_collision": 0,
        }

    for env_idx in range(n_envs):
        for t0 in range(0, steps - raw_span + 1, window_stride):
            t1 = t0 + raw_span
            total += 1
            if np.any(dones[env_idx, t0:t1]):
                continue
            no_done += 1
            if collisions is not None and np.any(collisions[env_idx, t0:t1]):
                continue
            no_done_no_collision += 1

    return {
        "raw_span": int(raw_span),
        "total_windows": int(total),
        "windows_no_done": int(no_done),
        "windows_no_done_no_collision": int(no_done_no_collision),
    }


def build_scene_entry(scene_key: str, scene_seed: int | None, scene_type: str, scene_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_key": scene_key,
        "scene_seed": scene_seed,
        "scene_type": scene_type,
        "scene_label": scene_meta.get("scene_label"),
        "grid_rows": scene_meta.get("grid_rows"),
        "grid_cols": scene_meta.get("grid_cols"),
        "cell_size": scene_meta.get("cell_size"),
        "wall_thickness": scene_meta.get("wall_thickness"),
        "n_obstacles": scene_meta.get("n_obstacles"),
        "n_beacons": scene_meta.get("n_beacons"),
        "n_distractors": scene_meta.get("n_distractors"),
        "files": 0,
        "envs": 0,
        "env_steps": 0,
        "episodes": 0,
        "resets": 0,
        "collisions": 0,
        "visible_steps": 0,
        "close_visible_steps": 0,
        "very_close_visible_steps": 0,
        "episodes_with_visible_beacon": 0,
        "episodes_with_close_visible_beacon": 0,
        "episodes_with_very_close_visible_beacon": 0,
        "maze_teacher_steps": 0,
        "cmd_pattern_counts": Counter(),
    }


def finalize_scene_entry(scene: dict[str, Any]) -> dict[str, Any]:
    env_steps = int(scene["env_steps"])
    episodes = int(scene["episodes"])
    cmd_counts = scene["cmd_pattern_counts"]
    teacher_steps = int(scene["maze_teacher_steps"])
    visible_steps = int(scene["visible_steps"])
    close_visible_steps = int(scene["close_visible_steps"])
    very_close_visible_steps = int(scene["very_close_visible_steps"])
    out = {k: v for k, v in scene.items() if k != "cmd_pattern_counts"}
    out["cmd_pattern_counts"] = dict(sorted(cmd_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    out["maze_teacher_fraction"] = (
        float(teacher_steps / env_steps) if env_steps > 0 else None
    )
    out["visible_beacon_step_fraction"] = (
        float(visible_steps / env_steps) if env_steps > 0 else None
    )
    out["close_visible_beacon_step_fraction"] = (
        float(close_visible_steps / env_steps) if env_steps > 0 else None
    )
    out["very_close_visible_beacon_step_fraction"] = (
        float(very_close_visible_steps / env_steps) if env_steps > 0 else None
    )
    out["episodes_with_visible_beacon_fraction"] = (
        float(scene["episodes_with_visible_beacon"] / episodes) if episodes > 0 else None
    )
    out["episodes_with_close_visible_beacon_fraction"] = (
        float(scene["episodes_with_close_visible_beacon"] / episodes) if episodes > 0 else None
    )
    out["episodes_with_very_close_visible_beacon_fraction"] = (
        float(scene["episodes_with_very_close_visible_beacon"] / episodes) if episodes > 0 else None
    )
    return out


def summarize_source(
    *,
    label: str,
    files: list[Path],
    loader: str,
    close_range_m: float,
    very_close_range_m: float,
    seq_len: int | None,
    temporal_stride: int,
    action_block_size: int | None,
    window_stride: int | None,
) -> dict[str, Any]:
    if loader == "rendered" and h5py is None:
        raise RuntimeError(
            "Rendered dataset summarization requires h5py, but it is not installed.",
        )
    total_env_steps = 0
    total_envs = 0
    total_episodes = 0
    total_resets = 0
    total_collisions = 0
    total_visible_steps = 0
    total_close_visible_steps = 0
    total_very_close_visible_steps = 0
    total_visible_eps = 0
    total_close_visible_eps = 0
    total_very_close_visible_eps = 0
    scene_type_counts: Counter[str] = Counter()
    command_counts: Counter[str] = Counter()
    enclosed_layout_counts: Counter[str] = Counter()
    scene_entries: dict[str, dict[str, Any]] = {}
    per_file_episodes: list[float] = []
    per_file_resets: list[float] = []
    per_file_collisions: list[float] = []
    per_file_teacher_fraction: list[float] = []
    valid_window_accum: Counter[str] = Counter()

    for path in files:
        if loader == "raw":
            with np.load(path, allow_pickle=True) as data:
                cmds = np.asarray(data["cmds"])
                dones = np.asarray(data["dones"], dtype=bool)
                collisions = np.asarray(data["collisions"], dtype=bool) if "collisions" in data else None
                beacon_visible = np.asarray(data["beacon_visible"], dtype=bool) if "beacon_visible" in data else None
                beacon_range = np.asarray(data["beacon_range"], dtype=np.float32) if "beacon_range" in data else None
                cmd_pattern = np.asarray(data["cmd_pattern"], dtype=np.int32) if "cmd_pattern" in data else None
                scene_seed = data["scene_seed"] if "scene_seed" in data else None
                scene_type = data["scene_type"] if "scene_type" in data else None
                scene_meta = parse_scene_meta(data["scene_meta"]) if "scene_meta" in data else {}
        else:
            with h5py.File(path, "r") as h5f:
                cmds = h5f["cmds"]
                dones = np.asarray(h5f["dones"][:], dtype=bool)
                collisions = np.asarray(h5f["collisions"][:], dtype=bool) if "collisions" in h5f else None
                beacon_visible = np.asarray(h5f["beacon_visible"][:], dtype=bool) if "beacon_visible" in h5f else None
                beacon_range = np.asarray(h5f["beacon_range"][:], dtype=np.float32) if "beacon_range" in h5f else None
                cmd_pattern = np.asarray(h5f["cmd_pattern"][:], dtype=np.int32) if "cmd_pattern" in h5f else None
                scene_seed = h5f.attrs.get("scene_seed")
                scene_type = h5f.attrs.get("scene_type")
                scene_meta = parse_scene_meta(h5f.attrs.get("scene_meta"))
                cmds = np.empty((int(cmds.shape[0]), int(cmds.shape[1]), 0), dtype=np.float32)

        n_envs, steps = int(dones.shape[0]), int(dones.shape[1])
        env_steps = int(n_envs * steps)
        resets = int(np.count_nonzero(dones))
        episodes = int(n_envs + resets)
        collisions_count = int(np.count_nonzero(collisions)) if collisions is not None else 0

        scene_key, stable_seed, scene_type_text = stable_scene_identity(
            scene_seed=scene_seed,
            scene_type=scene_type,
            fallback=path.name,
        )
        if scene_key not in scene_entries:
            scene_entries[scene_key] = build_scene_entry(scene_key, stable_seed, scene_type_text, scene_meta)
        scene = scene_entries[scene_key]
        scene["files"] += 1
        scene["envs"] += n_envs
        scene["env_steps"] += env_steps
        scene["episodes"] += episodes
        scene["resets"] += resets
        scene["collisions"] += collisions_count

        scene_type_counts[scene_type_text] += 1
        total_envs += n_envs
        total_env_steps += env_steps
        total_episodes += episodes
        total_resets += resets
        total_collisions += collisions_count
        per_file_episodes.append(float(episodes))
        per_file_resets.append(float(resets))
        per_file_collisions.append(float(collisions_count))

        if scene_type_text == "enclosed":
            rows = scene_meta.get("grid_rows")
            cols = scene_meta.get("grid_cols")
            if rows is not None and cols is not None:
                enclosed_layout_counts[f"{int(rows)}x{int(cols)}"] += 1

        if beacon_visible is not None:
            visible_steps = int(np.count_nonzero(beacon_visible))
            scene["visible_steps"] += visible_steps
            total_visible_steps += visible_steps
        else:
            visible_steps = 0
        if beacon_visible is not None and beacon_range is not None:
            visible_mask = beacon_visible & np.isfinite(beacon_range)
            close_steps = int(np.count_nonzero(visible_mask & (beacon_range <= close_range_m)))
            very_close_steps = int(np.count_nonzero(visible_mask & (beacon_range <= very_close_range_m)))
            scene["close_visible_steps"] += close_steps
            scene["very_close_visible_steps"] += very_close_steps
            total_close_visible_steps += close_steps
            total_very_close_visible_steps += very_close_steps

        proxy_counts = count_episode_proxies(
            dones=dones,
            beacon_visible=beacon_visible,
            beacon_range=beacon_range,
            close_range_m=close_range_m,
            very_close_range_m=very_close_range_m,
        )
        scene["episodes_with_visible_beacon"] += proxy_counts["episodes_with_visible_beacon"]
        scene["episodes_with_close_visible_beacon"] += proxy_counts["episodes_with_close_visible_beacon"]
        scene["episodes_with_very_close_visible_beacon"] += proxy_counts["episodes_with_very_close_visible_beacon"]
        total_visible_eps += proxy_counts["episodes_with_visible_beacon"]
        total_close_visible_eps += proxy_counts["episodes_with_close_visible_beacon"]
        total_very_close_visible_eps += proxy_counts["episodes_with_very_close_visible_beacon"]

        if cmd_pattern is not None:
            unique_ids, counts = np.unique(cmd_pattern, return_counts=True)
            teacher_steps = 0
            for pattern_idx, count in zip(unique_ids.tolist(), counts.tolist()):
                name = command_pattern_name(pattern_idx)
                command_counts[name] += int(count)
                scene["cmd_pattern_counts"][name] += int(count)
                if name in MAZE_TEACHER_PATTERN_NAMES:
                    teacher_steps += int(count)
            scene["maze_teacher_steps"] += teacher_steps
            per_file_teacher_fraction.append(float(teacher_steps / env_steps) if env_steps > 0 else 0.0)

        if seq_len is not None:
            effective_action_block = int(action_block_size if action_block_size is not None else temporal_stride)
            effective_window_stride = int(
                window_stride if window_stride is not None else seq_len * temporal_stride
            )
            window_stats = estimate_valid_windows(
                dones=dones,
                collisions=collisions,
                seq_len=int(seq_len),
                temporal_stride=int(temporal_stride),
                action_block_size=effective_action_block,
                window_stride=effective_window_stride,
            )
            valid_window_accum.update(window_stats)

    scenes = [
        finalize_scene_entry(scene)
        for scene in sorted(
            scene_entries.values(),
            key=lambda item: (-int(item["env_steps"]), str(item["scene_key"])),
        )
    ]
    teacher_steps_total = sum(command_counts.get(name, 0) for name in MAZE_TEACHER_PATTERN_NAMES)

    summary = {
        "label": label,
        "path": None if not files else str(files[0].parent),
        "files": len(files),
        "total_envs": int(total_envs),
        "total_env_steps": int(total_env_steps),
        "total_episodes": int(total_episodes),
        "total_resets": int(total_resets),
        "total_collisions": int(total_collisions),
        "unique_scenes": len(scene_entries),
        "scene_type_file_counts": dict(sorted(scene_type_counts.items())),
        "enclosed_layout_file_counts": dict(sorted(enclosed_layout_counts.items())),
        "command_pattern_step_counts": dict(sorted(command_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "maze_teacher_steps": int(teacher_steps_total),
        "maze_teacher_fraction": (
            float(teacher_steps_total / total_env_steps) if total_env_steps > 0 else None
        ),
        "visible_beacon_steps": int(total_visible_steps),
        "visible_beacon_step_fraction": (
            float(total_visible_steps / total_env_steps) if total_env_steps > 0 else None
        ),
        "close_visible_beacon_steps": int(total_close_visible_steps),
        "close_visible_beacon_step_fraction": (
            float(total_close_visible_steps / total_env_steps) if total_env_steps > 0 else None
        ),
        "very_close_visible_beacon_steps": int(total_very_close_visible_steps),
        "very_close_visible_beacon_step_fraction": (
            float(total_very_close_visible_steps / total_env_steps) if total_env_steps > 0 else None
        ),
        "episodes_with_visible_beacon": int(total_visible_eps),
        "episodes_with_visible_beacon_fraction": (
            float(total_visible_eps / total_episodes) if total_episodes > 0 else None
        ),
        "episodes_with_close_visible_beacon": int(total_close_visible_eps),
        "episodes_with_close_visible_beacon_fraction": (
            float(total_close_visible_eps / total_episodes) if total_episodes > 0 else None
        ),
        "episodes_with_very_close_visible_beacon": int(total_very_close_visible_eps),
        "episodes_with_very_close_visible_beacon_fraction": (
            float(total_very_close_visible_eps / total_episodes) if total_episodes > 0 else None
        ),
        "reset_rate_per_1k_env_steps": (
            float((1000.0 * total_resets) / total_env_steps) if total_env_steps > 0 else None
        ),
        "collision_rate_per_1k_env_steps": (
            float((1000.0 * total_collisions) / total_env_steps) if total_env_steps > 0 else None
        ),
        "per_file_episode_stats": summarize_numeric(per_file_episodes),
        "per_file_reset_stats": summarize_numeric(per_file_resets),
        "per_file_collision_stats": summarize_numeric(per_file_collisions),
        "per_file_maze_teacher_fraction_stats": summarize_numeric(per_file_teacher_fraction),
        "training_window_estimate": (
            {
                "seq_len": int(seq_len),
                "temporal_stride": int(temporal_stride),
                "action_block_size": int(action_block_size if action_block_size is not None else temporal_stride),
                "window_stride": int(window_stride if window_stride is not None else seq_len * temporal_stride),
                **dict(valid_window_accum),
            }
            if seq_len is not None
            else None
        ),
        "scenes": scenes,
    }
    return summary


def chunk_stem(path: Path) -> str:
    if path.name.endswith("_rgb.h5"):
        return path.name[:-len("_rgb.h5")]
    return path.stem


def build_crosscheck(raw_files: list[Path], rendered_files: list[Path], raw_summary: dict[str, Any] | None,
                     rendered_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw_files and not rendered_files:
        return None
    raw_stems = {chunk_stem(path) for path in raw_files}
    rendered_stems = {chunk_stem(path) for path in rendered_files}
    raw_scene_keys = {
        scene["scene_key"] for scene in (raw_summary or {}).get("scenes", [])
    }
    rendered_scene_keys = {
        scene["scene_key"] for scene in (rendered_summary or {}).get("scenes", [])
    }
    return {
        "raw_chunk_files": len(raw_files),
        "rendered_chunk_files": len(rendered_files),
        "matched_chunk_stems": len(raw_stems & rendered_stems),
        "raw_only_chunk_stems": sorted(raw_stems - rendered_stems),
        "rendered_only_chunk_stems": sorted(rendered_stems - raw_stems),
        "matched_scene_keys": len(raw_scene_keys & rendered_scene_keys),
        "raw_only_scene_keys": sorted(raw_scene_keys - rendered_scene_keys),
        "rendered_only_scene_keys": sorted(rendered_scene_keys - raw_scene_keys),
    }


def print_summary_block(summary: dict[str, Any], top_scenes: int) -> None:
    print(f"\n[{summary['label']}]")
    print(
        f"files={summary['files']} | scenes={summary['unique_scenes']} | "
        f"env_steps={summary['total_env_steps']:,} | episodes={summary['total_episodes']:,}"
    )
    print(
        f"resets={summary['total_resets']:,} ({summary['reset_rate_per_1k_env_steps']:.2f}/1k steps) | "
        f"collisions={summary['total_collisions']:,} ({summary['collision_rate_per_1k_env_steps']:.2f}/1k steps)"
    )
    teacher_fraction = summary["maze_teacher_fraction"]
    visible_fraction = summary["visible_beacon_step_fraction"]
    close_fraction = summary["episodes_with_close_visible_beacon_fraction"]
    print(
        f"maze_teacher_fraction={0.0 if teacher_fraction is None else teacher_fraction:.3f} | "
        f"visible_beacon_step_fraction={0.0 if visible_fraction is None else visible_fraction:.3f} | "
        f"episodes_with_close_visible_beacon_fraction={0.0 if close_fraction is None else close_fraction:.3f}"
    )
    if summary["scene_type_file_counts"]:
        scene_types = ", ".join(
            f"{name}={count}" for name, count in summary["scene_type_file_counts"].items()
        )
        print(f"scene types: {scene_types}")
    if summary["enclosed_layout_file_counts"]:
        layouts = ", ".join(
            f"{name}={count}" for name, count in summary["enclosed_layout_file_counts"].items()
        )
        print(f"enclosed layouts: {layouts}")
    if summary["command_pattern_step_counts"]:
        top_patterns = list(summary["command_pattern_step_counts"].items())[:8]
        patterns = ", ".join(f"{name}={count}" for name, count in top_patterns)
        print(f"top command patterns: {patterns}")
    if summary.get("training_window_estimate") is not None:
        win = summary["training_window_estimate"]
        print(
            "training windows: "
            f"total={win['total_windows']:,}, "
            f"no_done={win['windows_no_done']:,}, "
            f"no_done_no_collision={win['windows_no_done_no_collision']:,}"
        )
    scenes = summary.get("scenes", [])[:max(0, int(top_scenes))]
    if scenes:
        print("top scenes:")
        for scene in scenes:
            label = scene["scene_label"] or scene["scene_key"]
            teacher = scene.get("maze_teacher_fraction")
            close_eps = scene.get("episodes_with_close_visible_beacon_fraction")
            print(
                f"  {label}: type={scene['scene_type']} seed={scene['scene_seed']} "
                f"env_steps={scene['env_steps']:,} resets={scene['resets']:,} "
                f"teacher={0.0 if teacher is None else teacher:.3f} "
                f"close_eps={0.0 if close_eps is None else close_eps:.3f}"
            )


def main() -> None:
    args = parse_args()
    raw_files = discover_raw_files(args.raw_dir)
    rendered_files = discover_rendered_files(args.rendered_dir)
    if not raw_files and not rendered_files:
        raise SystemExit("No raw or rendered dataset files found.")

    raw_summary = (
        summarize_source(
            label="raw_rollouts",
            files=raw_files,
            loader="raw",
            close_range_m=float(args.close_range_m),
            very_close_range_m=float(args.very_close_range_m),
            seq_len=args.seq_len,
            temporal_stride=int(args.temporal_stride),
            action_block_size=args.action_block_size,
            window_stride=args.window_stride,
        )
        if raw_files
        else None
    )
    rendered_summary = (
        summarize_source(
            label="rendered_dataset",
            files=rendered_files,
            loader="rendered",
            close_range_m=float(args.close_range_m),
            very_close_range_m=float(args.very_close_range_m),
            seq_len=args.seq_len,
            temporal_stride=int(args.temporal_stride),
            action_block_size=args.action_block_size,
            window_stride=args.window_stride,
        )
        if rendered_files
        else None
    )

    report = {
        "raw": raw_summary,
        "rendered": rendered_summary,
        "crosscheck": build_crosscheck(raw_files, rendered_files, raw_summary, rendered_summary),
        "config": {
            "close_range_m": float(args.close_range_m),
            "very_close_range_m": float(args.very_close_range_m),
            "seq_len": args.seq_len,
            "temporal_stride": int(args.temporal_stride),
            "action_block_size": args.action_block_size,
            "window_stride": args.window_stride,
        },
    }

    print("Dataset coverage report")
    if raw_summary is not None:
        print_summary_block(raw_summary, top_scenes=int(args.top_scenes))
    if rendered_summary is not None:
        print_summary_block(rendered_summary, top_scenes=int(args.top_scenes))
    if report["crosscheck"] is not None:
        cross = report["crosscheck"]
        print("\n[crosscheck]")
        print(
            f"matched chunks={cross['matched_chunk_stems']} | "
            f"raw_only={len(cross['raw_only_chunk_stems'])} | "
            f"rendered_only={len(cross['rendered_only_chunk_stems'])}"
        )
        print(
            f"matched scenes={cross['matched_scene_keys']} | "
            f"raw_only_scenes={len(cross['raw_only_scene_keys'])} | "
            f"rendered_only_scenes={len(cross['rendered_only_scene_keys'])}"
        )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
