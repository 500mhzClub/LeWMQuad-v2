#!/usr/bin/env python3
"""Measure StreamingJEPADataset throughput in isolation.

Runs the same dataloader config as the training script but does nothing
with the batches — no model, no GPU transfer. If this benchmark hits a
reasonable batches/s while the training loop does not, the bottleneck is
in the training step (compute or transfer). If this benchmark is slow
too, the bottleneck is in the worker pipeline (h5py reads, Python
per-sample overhead, IPC, etc).

Usage:
    python scripts/bench_dataloader.py \\
        --data_dir /media/andrewknowles/Scratch/jepa_final_full \\
        --batch_size 128 --num_workers 12 --prefetch_factor 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader

from lewm.data import StreamingJEPADataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark dataloader throughput")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--temporal_stride", type=int, default=5)
    p.add_argument("--action_block_size", type=int, default=5)
    p.add_argument("--window_stride", type=int, default=5)
    p.add_argument("--command_representation", type=str, default="active_block")
    p.add_argument("--command_latency", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--warmup_batches", type=int, default=5,
                   help="Batches to discard before timing (dataloader spin-up).")
    p.add_argument("--max_batches", type=int, default=200,
                   help="Stop after this many timed batches.")
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Building StreamingJEPADataset from {args.data_dir} ...", flush=True)
    t_init = time.perf_counter()
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        temporal_stride=args.temporal_stride,
        action_block_size=args.action_block_size,
        command_representation=args.command_representation,
        command_latency=args.command_latency,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=args.num_workers,
        load_labels=False,
    )
    print(
        f"  init took {time.perf_counter() - t_init:.1f}s | "
        f"{len(dataset._all_indices):,} windows across {len(dataset.files)} files",
        flush=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=True,
    )

    print(
        f"Iterating: batch_size={args.batch_size} "
        f"num_workers={args.num_workers} prefetch={args.prefetch_factor}",
        flush=True,
    )

    t_last = None
    t_start = None
    n_timed = 0
    per_batch_times: list[float] = []
    slowest = 0.0
    fastest = float("inf")

    iterator = iter(dataloader)
    total_needed = args.warmup_batches + args.max_batches

    for i in range(total_needed):
        t0 = time.perf_counter()
        batch = next(iterator)
        dt = time.perf_counter() - t0

        if i < args.warmup_batches:
            if i == 0:
                print(f"  warmup batch 0: {dt:.2f}s (cold start)", flush=True)
            continue

        if t_start is None:
            t_start = time.perf_counter() - dt  # count this batch's time

        per_batch_times.append(dt)
        slowest = max(slowest, dt)
        fastest = min(fastest, dt)
        n_timed += 1

        if n_timed % args.print_every == 0:
            elapsed = time.perf_counter() - t_start
            rate = n_timed / elapsed
            avg = sum(per_batch_times) / n_timed
            print(
                f"  batch {n_timed:4d} | "
                f"rate={rate:6.2f} batch/s ({rate * args.batch_size:.0f} samples/s) | "
                f"avg={avg * 1000:6.1f}ms  fastest={fastest * 1000:6.1f}ms  slowest={slowest * 1000:6.1f}ms",
                flush=True,
            )

    if n_timed == 0:
        print("No batches timed — increase --max_batches.", flush=True)
        return

    elapsed = time.perf_counter() - t_start
    rate = n_timed / elapsed
    avg = sum(per_batch_times) / n_timed
    p50 = sorted(per_batch_times)[n_timed // 2]
    p95 = sorted(per_batch_times)[int(n_timed * 0.95)]

    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"  batches timed : {n_timed}", flush=True)
    print(f"  wall time     : {elapsed:.1f}s", flush=True)
    print(f"  throughput    : {rate:.2f} batch/s  ({rate * args.batch_size:.0f} samples/s)", flush=True)
    print(f"  per-batch avg : {avg * 1000:.1f} ms", flush=True)
    print(f"  per-batch p50 : {p50 * 1000:.1f} ms", flush=True)
    print(f"  per-batch p95 : {p95 * 1000:.1f} ms", flush=True)
    print(f"  fastest       : {fastest * 1000:.1f} ms", flush=True)
    print(f"  slowest       : {slowest * 1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
