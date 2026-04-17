#!/usr/bin/env python3
"""Analyze plan_audit.jsonl to test whether the safety head gives CEM any gradient.

Usage:
    python3 scripts/probe_safety_spread.py path/to/plan_audit.jsonl

For each audited replan, prints the spread of safety_cost across the top-K
candidates, plus a summary of command diversity (forward vs backward, yaw
spread) so we can tell whether the candidates were genuinely diverse.

Interpretation:
    safety_std/safety_mean  < 0.05  → head is effectively flat; CEM has no
                                       gradient and will collapse to whatever
                                       the initial/novelty term prefers.
    safety_std/safety_mean  > 0.20  → head is informative; another term is
                                       overriding it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _first_step_cmd(cmd_seq):
    """cmd_seq is (H, cmd_dim*macro_repeat); extract the first macro's mean vx,vy,yaw."""
    first_step = cmd_seq[0]
    n_repeat = len(first_step) // 3
    vx = statistics.mean(first_step[0::3])
    vy = statistics.mean(first_step[1::3])
    yaw = statistics.mean(first_step[2::3])
    return vx, vy, yaw


def analyze(path: Path, max_records: int = 20):
    print(f"Reading {path}")
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Total audited replans: {len(records)}")
    print()
    print(f"{'step':>5} {'pos_x':>6} {'pos_y':>6} "
          f"{'s_mean':>7} {'s_std':>7} {'s_min':>7} {'s_max':>7} "
          f"{'vx_spread':>10} {'yaw_spread':>10} {'best_vx':>8} {'best_yaw':>9}")

    flat_count = 0
    total = 0

    for rec in records[:max_records]:
        topk = rec.get("plan", {}).get("final_iteration_topk", [])
        if not topk:
            continue
        safeties = [c["metrics"].get("safety_cost", 0.0) for c in topk]
        cmds = [_first_step_cmd(c["command_sequence"]) for c in topk]
        vxs = [c[0] for c in cmds]
        yaws = [c[2] for c in cmds]

        s_mean = statistics.mean(safeties)
        s_std = statistics.pstdev(safeties) if len(safeties) > 1 else 0.0
        s_min = min(safeties)
        s_max = max(safeties)

        vx_spread = max(vxs) - min(vxs)
        yaw_spread = max(yaws) - min(yaws)

        best = topk[0]  # rank 1
        best_vx, _, best_yaw = _first_step_cmd(best["command_sequence"])

        pos = rec["state_before"]["pos_xy"]
        step = rec["step"]

        ratio = s_std / max(s_mean, 1e-6)
        flag = " FLAT" if ratio < 0.05 else ""
        if ratio < 0.05:
            flat_count += 1
        total += 1

        print(f"{step:>5d} {pos[0]:>6.2f} {pos[1]:>6.2f} "
              f"{s_mean:>7.4f} {s_std:>7.4f} {s_min:>7.4f} {s_max:>7.4f} "
              f"{vx_spread:>10.3f} {yaw_spread:>10.3f} "
              f"{best_vx:>8.2f} {best_yaw:>9.2f}{flag}")

    print()
    print(f"Flat replans (safety_std/safety_mean < 0.05): {flat_count}/{total}")
    if flat_count / max(total, 1) > 0.5:
        print("→ Safety head is NOT distinguishing candidates. "
              "CEM cannot use it as a cost signal. Retrain or recalibrate.")
    else:
        print("→ Safety head does vary across candidates. Problem is elsewhere.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audit_path", type=str, help="Path to plan_audit.jsonl")
    p.add_argument("--max_records", type=int, default=20)
    args = p.parse_args()
    analyze(Path(args.audit_path), max_records=args.max_records)


if __name__ == "__main__":
    main()
