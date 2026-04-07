#!/usr/bin/env python3
"""Aggregate `plan_audit.jsonl` files from multiple inference runs.

Examples:
    python3 scripts/aggregate_plan_audits.py inference_runs/run_a inference_runs/run_b
    python3 scripts/aggregate_plan_audits.py --glob "inference_runs/perception_only_*"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate planner audit files from inference runs.")
    parser.add_argument("runs", nargs="*", help="Run directories containing plan_audit.jsonl.")
    parser.add_argument("--glob", dest="glob_pattern", type=str, default=None,
                        help="Optional glob for run directories.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output path for the aggregated JSON report.")
    return parser.parse_args()


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


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    assert mx is not None and my is not None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return float(num / (den_x * den_y))


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = [Path(run) for run in args.runs]
    if args.glob_pattern:
        run_dirs.extend(Path(path) for path in sorted(glob.glob(args.glob_pattern)))
    seen: set[Path] = set()
    out: list[Path] = []
    for run_dir in run_dirs:
        if run_dir in seen:
            continue
        if (run_dir / "plan_audit.jsonl").is_file():
            out.append(run_dir)
            seen.add(run_dir)
    return out


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def get_nested(record: dict[str, Any], *keys: str) -> Any:
    cur: Any = record
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def add_pair(pairs: list[tuple[float, float]], x: Any, y: Any) -> None:
    if x is None or y is None:
        return
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return
    if not math.isfinite(xf) or not math.isfinite(yf):
        return
    pairs.append((xf, yf))


def summarize_pairs(pairs: list[tuple[float, float]]) -> dict[str, float | int] | None:
    if not pairs:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    diffs = [x - y for x, y in pairs]
    abs_diffs = [abs(d) for d in diffs]
    return {
        "count": len(pairs),
        "pred_mean": float(mean(xs)),
        "actual_mean": float(mean(ys)),
        "mean_signed_error": float(mean(diffs)),
        "mae": float(mean(abs_diffs)),
        "pearson_r": corr(xs, ys),
    }


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories with plan_audit.jsonl found.")

    result_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    actual_first_displacement: list[float] = []
    actual_first_goal_delta: list[float] = []
    actual_first_coverage_gain: list[float] = []
    actual_first_collision: list[float] = []
    actual_commit_displacement: list[float] = []
    actual_commit_goal_delta: list[float] = []
    actual_commit_coverage_gain: list[float] = []
    actual_commit_collision: list[float] = []
    top1_top2_margin: list[float] = []
    top1_top2_rel_margin: list[float] = []
    topk_cost_std: list[float] = []
    topk_cost_range: list[float] = []
    selected_costs: list[float] = []
    prediction_first: dict[str, list[float]] = {
        "raw_mse": [], "raw_l2": [], "raw_cosine": [],
        "proj_mse": [], "proj_l2": [], "proj_cosine": [],
    }
    prediction_commit: dict[str, list[float]] = {
        "raw_mse": [], "raw_l2": [], "raw_cosine": [],
        "proj_mse": [], "proj_l2": [], "proj_cosine": [],
    }
    first_goal_similarity_pairs: list[tuple[float, float]] = []
    commit_goal_similarity_pairs: list[tuple[float, float]] = []
    commit_displacement_pairs: list[tuple[float, float]] = []
    commit_coverage_pairs: list[tuple[float, float]] = []
    cost_vs_commit_disp: list[tuple[float, float]] = []
    cost_vs_commit_goal_delta: list[tuple[float, float]] = []
    cost_vs_commit_coverage_gain: list[tuple[float, float]] = []
    cost_vs_commit_collision: list[tuple[float, float]] = []
    run_summaries: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        summary = load_summary(run_dir / "summary.json")
        if summary is not None:
            result_counts[str(summary.get("result", "unknown"))] += 1
        audit_path = run_dir / "plan_audit.jsonl"
        n_records = 0
        with audit_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                n_records += 1

                mode_counts[str(record.get("mode", "unknown"))] += 1

                first_actual = record.get("actual_after_first_command", {})
                commit_actual = record.get("actual_after_commitment", {})
                selected_plan = get_nested(record, "plan", "selected_plan") or {}
                topk = get_nested(record, "plan", "final_iteration_topk") or []

                if "xy_displacement_m" in first_actual:
                    actual_first_displacement.append(float(first_actual["xy_displacement_m"]))
                if first_actual.get("goal_dist_delta_m") is not None:
                    actual_first_goal_delta.append(float(first_actual["goal_dist_delta_m"]))
                if first_actual.get("coverage_gain_m2") is not None:
                    actual_first_coverage_gain.append(float(first_actual["coverage_gain_m2"]))
                if "collision" in first_actual:
                    actual_first_collision.append(float(bool(first_actual["collision"])))

                if "xy_displacement_m" in commit_actual:
                    actual_commit_displacement.append(float(commit_actual["xy_displacement_m"]))
                if commit_actual.get("goal_dist_delta_m") is not None:
                    actual_commit_goal_delta.append(float(commit_actual["goal_dist_delta_m"]))
                if commit_actual.get("coverage_gain_m2") is not None:
                    actual_commit_coverage_gain.append(float(commit_actual["coverage_gain_m2"]))
                if "collision" in commit_actual:
                    actual_commit_collision.append(float(bool(commit_actual["collision"])))

                for key, values in prediction_first.items():
                    pred = record.get("prediction_error", {})
                    if key in pred:
                        values.append(float(pred[key]))
                for key, values in prediction_commit.items():
                    pred = record.get("prediction_error_after_commitment", {})
                    if key in pred:
                        values.append(float(pred[key]))

                add_pair(
                    first_goal_similarity_pairs,
                    get_nested(record, "predicted_after_first_command", "goal_similarity_proj"),
                    get_nested(record, "actual_after_first_command", "goal_similarity_proj"),
                )
                add_pair(
                    commit_goal_similarity_pairs,
                    get_nested(record, "predicted_after_commitment", "goal_similarity_proj"),
                    get_nested(record, "actual_after_commitment", "goal_similarity_proj"),
                )
                add_pair(
                    commit_displacement_pairs,
                    get_nested(record, "predicted_after_commitment", "predicted_displacement_m"),
                    get_nested(record, "actual_after_commitment", "xy_displacement_m"),
                )
                add_pair(
                    commit_coverage_pairs,
                    get_nested(record, "predicted_after_commitment", "predicted_coverage_gain_m2"),
                    get_nested(record, "actual_after_commitment", "coverage_gain_m2"),
                )

                if "cost" in selected_plan:
                    selected_cost = float(selected_plan["cost"])
                    selected_costs.append(selected_cost)
                    add_pair(cost_vs_commit_disp, selected_cost, commit_actual.get("xy_displacement_m"))
                    add_pair(cost_vs_commit_goal_delta, selected_cost, commit_actual.get("goal_dist_delta_m"))
                    add_pair(cost_vs_commit_coverage_gain, selected_cost, commit_actual.get("coverage_gain_m2"))
                    add_pair(cost_vs_commit_collision, selected_cost, float(bool(commit_actual.get("collision"))))

                if len(topk) >= 2:
                    c1 = float(topk[0]["cost"])
                    c2 = float(topk[1]["cost"])
                    margin = c2 - c1
                    top1_top2_margin.append(margin)
                    denom = max(abs(c1), 1e-6)
                    top1_top2_rel_margin.append(margin / denom)
                if topk:
                    topk_costs = [float(item["cost"]) for item in topk]
                    topk_cost_std.append(float(stddev(topk_costs)))
                    topk_cost_range.append(float(max(topk_costs) - min(topk_costs)))

        planner = (summary or {}).get("planner", {})
        run_summaries.append({
            "run_dir": str(run_dir),
            "result": None if summary is None else summary.get("result"),
            "seed": None if summary is None else summary.get("seed"),
            "records": n_records,
            "planner_action_space": planner.get("planner_action_space"),
            "memory_router_mode": planner.get("memory_router_mode"),
            "goal_cost_mode": planner.get("goal_cost_mode"),
            "objective": planner.get("objective"),
        })

    ranking_summary = {
        "top1_top2_margin": summarize_numeric(top1_top2_margin),
        "top1_top2_relative_margin": summarize_numeric(top1_top2_rel_margin),
        "topk_cost_std": summarize_numeric(topk_cost_std),
        "topk_cost_range": summarize_numeric(topk_cost_range),
        "relative_near_tie_rate_le_1pct": (
            float(sum(m <= 0.01 for m in top1_top2_rel_margin) / len(top1_top2_rel_margin))
            if top1_top2_rel_margin else None
        ),
        "relative_near_tie_rate_le_5pct": (
            float(sum(m <= 0.05 for m in top1_top2_rel_margin) / len(top1_top2_rel_margin))
            if top1_top2_rel_margin else None
        ),
        "relative_near_tie_rate_le_10pct": (
            float(sum(m <= 0.10 for m in top1_top2_rel_margin) / len(top1_top2_rel_margin))
            if top1_top2_rel_margin else None
        ),
    }

    report = {
        "n_runs": len(run_dirs),
        "result_counts": dict(sorted(result_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "selected_plan_cost": summarize_numeric(selected_costs),
        "actual_after_first_command": {
            "xy_displacement_m": summarize_numeric(actual_first_displacement),
            "goal_dist_delta_m": summarize_numeric(actual_first_goal_delta),
            "coverage_gain_m2": summarize_numeric(actual_first_coverage_gain),
            "collision_rate": mean(actual_first_collision),
        },
        "actual_after_commitment": {
            "xy_displacement_m": summarize_numeric(actual_commit_displacement),
            "goal_dist_delta_m": summarize_numeric(actual_commit_goal_delta),
            "coverage_gain_m2": summarize_numeric(actual_commit_coverage_gain),
            "collision_rate": mean(actual_commit_collision),
        },
        "prediction_error_first_command": {
            key: summarize_numeric(values) for key, values in prediction_first.items()
        },
        "prediction_error_after_commitment": {
            key: summarize_numeric(values) for key, values in prediction_commit.items()
        },
        "goal_similarity_alignment": {
            "first_command": summarize_pairs(first_goal_similarity_pairs),
            "after_commitment": summarize_pairs(commit_goal_similarity_pairs),
        },
        "commitment_metric_alignment": {
            "predicted_displacement_vs_actual_xy_displacement": summarize_pairs(commit_displacement_pairs),
            "predicted_coverage_gain_vs_actual_coverage_gain": summarize_pairs(commit_coverage_pairs),
        },
        "ranking_confidence": ranking_summary,
        "selected_cost_vs_actual_commitment": {
            "xy_displacement_m": summarize_pairs(cost_vs_commit_disp),
            "goal_dist_delta_m": summarize_pairs(cost_vs_commit_goal_delta),
            "coverage_gain_m2": summarize_pairs(cost_vs_commit_coverage_gain),
            "collision": summarize_pairs(cost_vs_commit_collision),
        },
        "runs": run_summaries,
    }

    print("Planner audit aggregate")
    print(
        f"runs={report['n_runs']} | "
        f"records={sum(run['records'] for run in run_summaries)} | "
        f"results={report['result_counts']}"
    )
    first_collision_rate = report["actual_after_first_command"]["collision_rate"]
    commit_collision_rate = report["actual_after_commitment"]["collision_rate"]
    print(
        "collision_rate "
        f"first={0.0 if first_collision_rate is None else first_collision_rate:.3f} | "
        f"commitment={0.0 if commit_collision_rate is None else commit_collision_rate:.3f}"
    )
    rank = report["ranking_confidence"]
    top1_top2 = rank["top1_top2_relative_margin"]
    print(
        "relative top1-top2 margin "
        f"mean={0.0 if top1_top2 is None else top1_top2['mean']:.4f} | "
        f"near_tie<=5%={0.0 if rank['relative_near_tie_rate_le_5pct'] is None else rank['relative_near_tie_rate_le_5pct']:.3f}"
    )
    goal_align = report["goal_similarity_alignment"]["after_commitment"]
    if goal_align is not None:
        print(
            "commitment goal-sim alignment "
            f"mae={goal_align['mae']:.4f} | r={0.0 if goal_align['pearson_r'] is None else goal_align['pearson_r']:.4f}"
        )

    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_json + "\n")
        print(f"\nWrote aggregate report to {out_path}")


if __name__ == "__main__":
    main()
