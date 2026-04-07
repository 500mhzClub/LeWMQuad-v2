#!/usr/bin/env python3
"""Aggregate `summary.json` files from multiple inference runs.

Usage:
    python3 scripts/7_aggregate_inference_runs.py inference_runs/run_a inference_runs/run_b
    python3 scripts/7_aggregate_inference_runs.py --glob "inference_runs/perception_only_*"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate perception-only inference summaries.")
    p.add_argument("runs", nargs="*", help="Run directories containing summary.json.")
    p.add_argument("--glob", dest="glob_pattern", type=str, default=None,
                   help="Optional glob for run directories.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional output path for the aggregated JSON report.")
    return p.parse_args()


def discover_summary_paths(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = []
    for run in args.runs:
        run_dirs.append(Path(run))
    if args.glob_pattern:
        run_dirs.extend(Path(path) for path in sorted(glob.glob(args.glob_pattern)))

    seen: set[Path] = set()
    summary_paths: list[Path] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and summary_path not in seen:
            summary_paths.append(summary_path)
            seen.add(summary_path)
    return summary_paths


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    summary_paths = discover_summary_paths(args)
    if not summary_paths:
        raise SystemExit("No summary.json files found.")

    summaries = []
    for path in summary_paths:
        with open(path, "r") as f:
            data = json.load(f)
        data["_summary_path"] = str(path)
        summaries.append(data)

    n_runs = len(summaries)
    perceptual_success = sum(bool(s.get("perceptual_goal_detected")) for s in summaries)
    oracle_success = sum(bool(s.get("oracle_goal_reached")) for s in summaries)
    result_counts: dict[str, int] = {}
    for summary in summaries:
        result = str(summary.get("result", "unknown"))
        result_counts[result] = result_counts.get(result, 0) + 1

    coverage_values = [
        float(s["soft_coverage_area_m2"])
        for s in summaries
        if s.get("soft_coverage_area_m2") is not None
    ]
    gain_values = [
        float(s["soft_coverage_gain_per_m"])
        for s in summaries
        if s.get("soft_coverage_gain_per_m") is not None
    ]
    min_goal_dist_values = [
        float(s["min_goal_dist_m"])
        for s in summaries
        if s.get("min_goal_dist_m") is not None and math.isfinite(float(s["min_goal_dist_m"]))
    ]
    collision_values = [int(s.get("oracle_collisions", 0)) for s in summaries]
    path_length_values = [float(s.get("path_length_m", 0.0)) for s in summaries]
    step_values = [int(s.get("steps", 0)) for s in summaries]

    aggregate = {
        "n_runs": n_runs,
        "perceptual_goal_detect_rate": perceptual_success / float(n_runs),
        "oracle_goal_reach_rate": oracle_success / float(n_runs),
        "result_counts": result_counts,
        "mean_soft_coverage_area_m2": mean(coverage_values),
        "mean_soft_coverage_gain_per_m": mean(gain_values),
        "mean_min_goal_dist_m": mean(min_goal_dist_values),
        "mean_oracle_collisions": mean([float(v) for v in collision_values]),
        "mean_path_length_m": mean(path_length_values),
        "mean_steps": mean([float(v) for v in step_values]),
        "runs": summaries,
    }

    report = json.dumps(aggregate, indent=2)
    print(report)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        print(f"\nWrote aggregate report to {out_path}")


if __name__ == "__main__":
    main()
