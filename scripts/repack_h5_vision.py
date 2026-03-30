#!/usr/bin/env python3
"""Repack HDF5 vision datasets from gzip to lzf or uncompressed.

Gzip decompression is too slow for random-access training on 224x224 images.
This script rewrites the 'vision' dataset with a faster compression (or none)
while leaving small datasets (proprio, cmds, dones, labels) as-is.

Usage:
    python scripts/repack_h5_vision.py --data_dir jepa_final_v3 --compression none
    python scripts/repack_h5_vision.py --data_dir jepa_final_v3 --compression lzf
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm


def repack_file(src_path: str, compression: str | None) -> None:
    """Repack a single HDF5 file's vision dataset."""
    tmp_path = src_path + ".repack_tmp"

    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        # Copy vision with new compression
        vis = src["vision"]
        N, T, C, H, W = vis.shape
        chunk_shape = (1, T, C, H, W)

        kw = dict(dtype=vis.dtype, chunks=chunk_shape)
        if compression is not None:
            kw["compression"] = compression
        dst_vis = dst.create_dataset("vision", (N, T, C, H, W), **kw)

        # Copy env-by-env to avoid loading entire dataset into RAM
        for e in tqdm(range(N), desc=f"  envs", leave=False):
            dst_vis[e] = vis[e]

        # Copy everything else as-is (small datasets, keep original compression)
        for key in src:
            if key == "vision":
                continue
            src.copy(key, dst, name=key)

        # Copy attributes
        for attr_name, attr_val in src.attrs.items():
            dst.attrs[attr_name] = attr_val

    # Atomic replace
    os.replace(tmp_path, src_path)


def main():
    parser = argparse.ArgumentParser(description="Repack HDF5 vision compression")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--compression", type=str, default="none",
        choices=["none", "lzf"],
        help="Target compression for vision data. 'none' is fastest reads, "
             "'lzf' is ~10x faster than gzip with ~70%% compression ratio.",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression

    patterns = ("*_rgb.h5", "chunk_*.h5")
    files = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(args.data_dir, pattern))):
            if path not in seen:
                files.append(path)
                seen.add(path)

    if not files:
        print(f"No HDF5 files found in {args.data_dir}")
        sys.exit(1)

    # Check current compression
    with h5py.File(files[0], "r") as f:
        cur = f["vision"].compression
        shape = f["vision"].shape
    print(f"Found {len(files)} files, vision shape={shape}")
    print(f"Current compression: {cur} -> target: {compression or 'none'}")

    if cur == compression:
        print("Already using target compression. Nothing to do.")
        return

    total_start = time.time()
    for i, fpath in enumerate(files):
        sz_before = os.path.getsize(fpath)
        print(f"[{i+1}/{len(files)}] {os.path.basename(fpath)} ({sz_before/1e9:.1f} GB) ...")
        t0 = time.time()
        repack_file(fpath, compression)
        sz_after = os.path.getsize(fpath)
        dt = time.time() - t0
        print(f"  {sz_before/1e9:.1f} GB -> {sz_after/1e9:.1f} GB in {dt:.0f}s")

    print(f"\nDone in {time.time() - total_start:.0f}s total.")


if __name__ == "__main__":
    main()
