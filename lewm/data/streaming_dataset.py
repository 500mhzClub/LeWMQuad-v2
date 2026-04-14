"""Streaming HDF5 dataset for JEPA and energy-head training."""
from __future__ import annotations

import glob
import math
import os
from collections import OrderedDict
from typing import Iterator, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingJEPADataset(IterableDataset):
    """Yields (vision, proprio, cmds, dones, collisions, labels) sequence tuples.

    Streams vision directly from HDF5 and keeps a small worker-local LRU for
    non-vision arrays. Sequence indices are pre-built at init and sharded
    across all workers by index (not by file), so all num_workers are active
    regardless of file count.

    The *labels* dict contains optional supervision signals added by the
    extended data pipeline: clearance, near_miss, traversability,
    beacon_visible, beacon_identity, beacon_bearing, beacon_range, cmd_pattern.
    Missing fields are filled with zeros/defaults.

    Args:
        data_dir: directory containing rendered chunk files (``*_rgb.h5``).
        seq_len: number of timesteps per returned sequence.
        temporal_stride: spacing in raw timesteps between returned observations.
        action_block_size: raw timesteps aggregated into each returned command.
            Defaults to ``temporal_stride`` so each model step spans one stride.
        command_representation: how each action block is represented.
            ``mean_scaled`` returns the legacy mean over logged commands.
            ``mean_active`` returns the mean over reconstructed executed commands.
            ``active_block`` returns the full reconstructed executed block, flattened.
        command_latency: deterministic delay used to reconstruct executed commands
            from the stored nominal command stream.
        window_stride: spacing in raw timesteps between sequence starts.
            Defaults to ``seq_len * temporal_stride`` for non-overlapping windows.
        batch_size: worker-side micro-batch size.
        require_no_done: skip sequences that contain a ``done`` flag.
        require_no_collision: skip sequences that contain a ``collision`` flag.
        load_labels: whether to load extended label fields.
    """

    # Extended label fields and their (dtype, default_value) specs
    LABEL_FIELDS = {
        "clearance":        (np.float32, 999.0),
        "near_miss":        (np.bool_,   False),
        "traversability":   (np.int32,   10),
        "beacon_visible":   (np.bool_,   False),
        "beacon_identity":  (np.int32,   -1),
        "beacon_bearing":   (np.float32, 0.0),
        "beacon_range":     (np.float32, 999.0),
        "cmd_pattern":      (np.int32,   0),
    }
    SMALL_ARRAY_CACHE_SIZE = 2

    def __init__(
        self,
        data_dir: str,
        raw_data_dir: str | None = None,
        seq_len: int = 4,
        temporal_stride: int = 1,
        action_block_size: int | None = None,
        command_representation: str = "mean_scaled",
        command_latency: int = 2,
        window_stride: int | None = None,
        batch_size: int = 256,
        require_no_done: bool = True,
        require_no_collision: bool = True,
        num_workers: int = 1,
        load_labels: bool = True,
        load_pose: bool = False,
        allowed_scene_ids: set[int] | list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.files: List[str] = self._discover_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"No rendered HDF5 chunk files found in {data_dir}")
        self.seq_len = seq_len
        self.temporal_stride = max(1, int(temporal_stride))
        self.action_block_size = max(
            1,
            int(action_block_size if action_block_size is not None else self.temporal_stride),
        )
        self.command_representation = str(command_representation)
        if self.command_representation not in {"mean_scaled", "mean_active", "active_block"}:
            raise ValueError(
                "command_representation must be one of "
                "{'mean_scaled', 'mean_active', 'active_block'}"
            )
        self.command_latency = max(0, int(command_latency))
        self.cmd_dim = (
            3 * self.action_block_size
            if self.command_representation == "active_block"
            else 3
        )
        default_window_stride = self.seq_len * self.temporal_stride
        self.window_stride = max(
            1,
            int(window_stride if window_stride is not None else default_window_stride),
        )
        self.batch_size = batch_size
        self.require_no_done = require_no_done
        self.require_no_collision = require_no_collision
        self._num_workers = max(1, num_workers)
        self.load_labels = load_labels
        self.load_pose = bool(load_pose)
        self.vision_shape, self.proprio_dim = self._inspect_schema()
        self.raw_span = (self.seq_len - 1) * self.temporal_stride + self.action_block_size
        self._scene_ids = self._build_scene_ids()
        self.allowed_scene_ids = (
            None if allowed_scene_ids is None else {int(scene_id) for scene_id in allowed_scene_ids}
        )
        self._raw_files: dict[str, str] = (
            self._resolve_raw_files(raw_data_dir) if self.load_pose else {}
        )
        self._episode_metadata = self._build_episode_metadata()

        # Pre-build index table: list of (file_path, env_idx, t0)
        # Scans dones/collisions at init (small arrays, ~10 MB total).
        self._all_indices: List[Tuple[str, int, int]] = self._precompute_indices()

    def _reconstruct_active_commands(
        self,
        cmd_source: np.ndarray,
        prefix_start: int,
        t0: int,
        raw_end: int,
    ) -> np.ndarray:
        """Reconstruct executed commands from the stored nominal stream.

        The physics rollout applies a deterministic zero-initialized latency
        buffer and stores the nominal commands. This helper recovers the
        latency-buffered commands that the PPO policy actually consumed.
        """
        active = np.zeros((raw_end - t0, 3), dtype=np.float32)
        if self.command_latency <= 0:
            active[:] = cmd_source[(t0 - prefix_start):(raw_end - prefix_start)]
            return active

        src_idx = (
            np.arange(t0, raw_end, dtype=np.int64)
            - self.command_latency
            - prefix_start
        )
        valid = src_idx >= 0
        if np.any(valid):
            active[valid] = cmd_source[src_idx[valid]]
        return active

    def _get_small_arrays(
        self,
        fpath: str,
        h5f: h5py.File,
        cache: OrderedDict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Read non-vision arrays once per file into a small worker-local LRU."""
        arrays = cache.get(fpath)
        if arrays is not None:
            cache.move_to_end(fpath)
            return arrays

        arrays = {
            "proprio": np.asarray(h5f["proprio"][:], dtype=np.float32),
            "cmds": np.asarray(h5f["cmds"][:], dtype=np.float32),
        }
        if "dones" in h5f:
            arrays["dones"] = np.asarray(h5f["dones"][:], dtype=np.bool_)
        if "collisions" in h5f:
            arrays["collisions"] = np.asarray(h5f["collisions"][:], dtype=np.bool_)
        if self.load_labels:
            for field, (dtype, _) in self.LABEL_FIELDS.items():
                if field in h5f:
                    arrays[field] = np.asarray(h5f[field][:], dtype=dtype)

        cache[fpath] = arrays
        while len(cache) > self.SMALL_ARRAY_CACHE_SIZE:
            cache.popitem(last=False)
        return arrays

    @staticmethod
    def _discover_files(data_dir: str) -> List[str]:
        patterns = ("*_rgb.h5", "chunk_*.h5")
        files = []
        seen = set()
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
                if "_tmp_" in os.path.basename(path):
                    continue
                if path not in seen:
                    files.append(path)
                    seen.add(path)
        return files

    @staticmethod
    def _raw_chunk_name(h5_path: str) -> str:
        base = os.path.basename(h5_path)
        if base.endswith("_rgb.h5"):
            return base[: -len("_rgb.h5")] + ".npz"
        if base.endswith(".h5"):
            return base[: -len(".h5")] + ".npz"
        raise ValueError(f"Unsupported rendered chunk filename: {h5_path}")

    def _resolve_raw_files(self, raw_data_dir: str | None) -> dict[str, str]:
        if raw_data_dir is None:
            raise ValueError("load_pose=True requires raw_data_dir to locate raw rollout chunks.")
        mapping: dict[str, str] = {}
        missing: list[str] = []
        for h5_path in self.files:
            raw_name = self._raw_chunk_name(h5_path)
            raw_path = os.path.join(raw_data_dir, raw_name)
            if not os.path.isfile(raw_path):
                missing.append(raw_path)
                continue
            mapping[h5_path] = raw_path
        if missing:
            preview = ", ".join(missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise FileNotFoundError(
                "Could not locate raw rollout chunks for pose supervision. "
                f"Missing: {preview}{suffix}"
            )
        return mapping

    def _inspect_schema(self) -> Tuple[Tuple[int, ...], int]:
        with h5py.File(self.files[0], "r") as h5f:
            vision_shape = tuple(int(x) for x in h5f["vision"].shape[2:])
            proprio_dim = int(h5f["proprio"].shape[-1])

        for fpath in self.files[1:]:
            with h5py.File(fpath, "r") as h5f:
                if tuple(int(x) for x in h5f["vision"].shape[2:]) != vision_shape:
                    raise ValueError(
                        f"Inconsistent vision shape in {fpath}: "
                        f"{tuple(h5f['vision'].shape[2:])} != {vision_shape}"
                    )
                if int(h5f["proprio"].shape[-1]) != proprio_dim:
                    raise ValueError(
                        f"Inconsistent proprio dim in {fpath}: "
                        f"{int(h5f['proprio'].shape[-1])} != {proprio_dim}"
                    )

        return vision_shape, proprio_dim

    def _build_episode_metadata(self) -> dict[str, dict[str, np.ndarray]]:
        """Precompute true reset-separated episode ids and per-episode steps.

        ``dones[t]`` ends the current episode at raw step ``t``; raw step
        ``t + 1`` starts a fresh episode.
        """
        metadata: dict[str, dict[str, np.ndarray]] = {}
        next_episode_id = 0
        for fpath in self.files:
            with h5py.File(fpath, "r") as h5f:
                n_envs, T = h5f["vision"].shape[:2]
                if "dones" in h5f:
                    dones = np.asarray(h5f["dones"][:], dtype=np.bool_)
                else:
                    dones = np.zeros((n_envs, T), dtype=np.bool_)

            episode_ids = np.empty((n_envs, T), dtype=np.int64)
            episode_steps = np.empty((n_envs, T), dtype=np.int64)
            for env_idx in range(n_envs):
                current_episode_id = next_episode_id
                next_episode_id += 1
                step_in_episode = 0
                for raw_step in range(T):
                    episode_ids[env_idx, raw_step] = current_episode_id
                    episode_steps[env_idx, raw_step] = step_in_episode
                    if bool(dones[env_idx, raw_step]):
                        current_episode_id = next_episode_id
                        next_episode_id += 1
                        step_in_episode = 0
                    else:
                        step_in_episode += 1

            metadata[fpath] = {
                "episode_id": episode_ids,
                "episode_step": episode_steps,
            }
        return metadata

    def _build_scene_ids(self) -> dict[str, int]:
        """Build stable scene ids from dataset metadata when available."""
        scene_key_to_id: dict[str, int] = {}
        file_to_scene_id: dict[str, int] = {}
        next_scene_id = 0
        for fpath in self.files:
            scene_key = fpath
            with h5py.File(fpath, "r") as h5f:
                attrs = h5f.attrs
                if "scene_seed" in attrs:
                    scene_seed = int(attrs["scene_seed"])
                    scene_type = attrs.get("scene_type", "scene")
                    if isinstance(scene_type, bytes):
                        scene_type = scene_type.decode("utf-8", errors="ignore")
                    scene_key = f"{scene_type}:{scene_seed}"
            if scene_key not in scene_key_to_id:
                scene_key_to_id[scene_key] = next_scene_id
                next_scene_id += 1
            file_to_scene_id[fpath] = scene_key_to_id[scene_key]
        return file_to_scene_id

    def _precompute_indices(self) -> List[Tuple[str, int, int]]:
        indices = []
        for fpath in self.files:
            scene_id = self._scene_ids[fpath]
            if self.allowed_scene_ids is not None and scene_id not in self.allowed_scene_ids:
                continue
            with h5py.File(fpath, "r") as h5f:
                n_envs, T = h5f["vision"].shape[:2]
                if T < self.raw_span:
                    continue
                dones = (
                    h5f["dones"][:] if ("dones" in h5f and self.require_no_done) else None
                )
                collisions = (
                    h5f["collisions"][:]
                    if ("collisions" in h5f and self.require_no_collision)
                    else None
                )
                for e in range(n_envs):
                    for t0 in range(0, T - self.raw_span + 1, self.window_stride):
                        t1 = t0 + self.raw_span
                        if dones is not None and np.any(dones[e, t0:t1]):
                            continue
                        if collisions is not None and np.any(collisions[e, t0:t1]):
                            continue
                        indices.append((fpath, e, t0))
        return indices

    def __len__(self) -> int:
        # Sum per-worker batch counts (each worker rounds up independently).
        # This mirrors __iter__: worker sharding happens at (file, env) group
        # granularity so workers do not duplicate chunk/cache reads.
        group_sizes = [len(group) for group in self._group_indices(self._all_indices)]
        worker_sizes = [
            sum(group_sizes[w :: self._num_workers])
            for w in range(self._num_workers)
        ]
        return sum(math.ceil(s / self.batch_size) for s in worker_sizes)

    @staticmethod
    def _group_indices(indices):
        """Group indices by (file, env) and sort each group by raw start step."""
        from collections import defaultdict
        groups = defaultdict(list)
        for fpath, e, t0 in indices:
            groups[(fpath, e)].append((fpath, e, t0))
        return [sorted(v, key=lambda x: x[2]) for v in groups.values()]

    @staticmethod
    def _flatten_groups(groups):
        return [idx for group in groups for idx in group]

    def __iter__(self) -> Iterator:
        info = torch.utils.data.get_worker_info()

        # Group by (file, env) so consecutive windows hit the same HDF5 chunk,
        # then shuffle group order for epoch-level randomness.
        rng = np.random.RandomState()
        groups = self._group_indices(self._all_indices)
        rng.shuffle(groups)

        if info is not None:
            groups = groups[info.id :: info.num_workers]
        indices = self._flatten_groups(groups)

        obs_offsets_template = (
            np.arange(self.seq_len, dtype=np.int64) * self.temporal_stride
        )
        # Precomputed gather matrix for per-step reductions over an
        # action_block_size-long window.  Shape: (seq_len, action_block_size).
        block_gather = (
            obs_offsets_template[:, None]
            + np.arange(self.action_block_size, dtype=np.int64)[None, :]
        )

        open_files: dict = {}
        small_array_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        raw_arrays: dict = {}
        try:
            for b0 in range(0, len(indices), self.batch_size):
                batch_idx = indices[b0 : b0 + self.batch_size]
                B = len(batch_idx)
                if B == 0:
                    continue

                vis = np.empty((B, self.seq_len, *self.vision_shape), dtype=np.uint8)
                prop = np.empty((B, self.seq_len, self.proprio_dim), dtype=np.float32)
                cmds = np.empty((B, self.seq_len, self.cmd_dim), dtype=np.float32)
                dones = np.zeros((B, self.seq_len), dtype=np.bool_)
                collisions = np.zeros((B, self.seq_len), dtype=np.bool_)

                label_arrays = {}
                if self.load_labels:
                    for field, (dtype, default) in self.LABEL_FIELDS.items():
                        label_arrays[field] = np.full(
                            (B, self.seq_len), default, dtype=dtype
                        )
                episode_ids = np.empty((B, self.seq_len), dtype=np.int64)
                scene_ids = np.empty((B, self.seq_len), dtype=np.int64)
                obs_steps = np.empty((B, self.seq_len), dtype=np.int64)
                raw_steps = np.empty((B, self.seq_len), dtype=np.int64)
                if self.load_pose:
                    robot_xy = np.empty((B, self.seq_len, 2), dtype=np.float32)
                    robot_yaw = np.empty((B, self.seq_len), dtype=np.float32)

                for i, (fpath, env_idx, t0) in enumerate(batch_idx):
                    if fpath not in open_files:
                        # Vision is chunked (1, T, C, H, W) ~= 150 MB/chunk and
                        # gzip-compressed. The default 1 MB chunk cache means
                        # every per-sample slice re-decompresses the whole
                        # chunk, which is the main source of long dataloader
                        # stalls. A 512 MB rdcc holds the current env's chunk
                        # (plus room for one more across group transitions),
                        # so subsequent windows from the same group hit the
                        # cache instead of re-running gzip.
                        open_files[fpath] = h5py.File(
                            fpath,
                            "r",
                            rdcc_nbytes=512 * 1024 * 1024,
                            rdcc_nslots=1031,
                            rdcc_w0=0.75,
                        )
                    h5f = open_files[fpath]
                    small_arrays = self._get_small_arrays(fpath, h5f, small_array_cache)

                    raw_end = t0 + self.raw_span
                    obs_idx = t0 + obs_offsets_template
                    block_idx = t0 + block_gather

                    episode_meta = self._episode_metadata[fpath]
                    episode_ids[i] = episode_meta["episode_id"][env_idx, obs_idx]
                    scene_ids[i].fill(self._scene_ids[fpath])
                    obs_steps[i] = episode_meta["episode_step"][env_idx, obs_idx]
                    raw_steps[i] = obs_idx

                    vis_slice = h5f["vision"][env_idx, t0:raw_end]
                    vis[i] = vis_slice[obs_offsets_template]
                    prop[i] = small_arrays["proprio"][env_idx, obs_idx]

                    cmds_env = small_arrays["cmds"][env_idx]
                    dones_data = small_arrays.get("dones")
                    collisions_data = small_arrays.get("collisions")

                    if self.load_labels:
                        for field in self.LABEL_FIELDS:
                            label_data = small_arrays.get(field)
                            if label_data is not None:
                                label_arrays[field][i] = label_data[env_idx, obs_idx]

                    if self.load_pose:
                        if fpath not in raw_arrays:
                            raw_npz = np.load(self._raw_files[fpath], allow_pickle=True)
                            base_pos = np.asarray(raw_npz["base_pos"], dtype=np.float32)
                            base_quat = np.asarray(raw_npz["base_quat"], dtype=np.float32)
                            if base_pos.shape[:2] != h5f["vision"].shape[:2]:
                                raise ValueError(
                                    "Raw/HDF5 shape mismatch for pose supervision: "
                                    f"{self._raw_files[fpath]} {base_pos.shape[:2]} vs "
                                    f"{fpath} {h5f['vision'].shape[:2]}"
                                )
                            w = base_quat[..., 0]
                            x = base_quat[..., 1]
                            y = base_quat[..., 2]
                            z = base_quat[..., 3]
                            yaw = np.arctan2(
                                2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z),
                            ).astype(np.float32)
                            raw_arrays[fpath] = {"base_pos": base_pos, "yaw": yaw}
                            raw_npz.close()
                        pose_data = raw_arrays[fpath]
                        robot_xy[i] = pose_data["base_pos"][env_idx, obs_idx, :2]
                        robot_yaw[i] = pose_data["yaw"][env_idx, obs_idx]

                    if self.command_representation == "mean_scaled":
                        cmds[i] = cmds_env[block_idx].mean(axis=1)
                    else:
                        active_prefix_start = max(0, t0 - self.command_latency)
                        active_slice = self._reconstruct_active_commands(
                            cmds_env[active_prefix_start:raw_end],
                            active_prefix_start,
                            t0,
                            raw_end,
                        )
                        gathered = active_slice[block_gather]
                        if self.command_representation == "mean_active":
                            cmds[i] = gathered.mean(axis=1)
                        else:
                            cmds[i] = gathered.reshape(self.seq_len, -1)
                    if dones_data is not None:
                        dones[i] = dones_data[env_idx, block_idx].any(axis=1)
                    if collisions_data is not None:
                        collisions[i] = collisions_data[env_idx, block_idx].any(axis=1)

                # Build label dict of tensors
                labels = {}
                if self.load_labels:
                    for field, arr in label_arrays.items():
                        labels[field] = torch.from_numpy(arr)
                labels["episode_id"] = torch.from_numpy(episode_ids)
                labels["scene_id"] = torch.from_numpy(scene_ids)
                labels["obs_step"] = torch.from_numpy(obs_steps)
                labels["raw_step"] = torch.from_numpy(raw_steps)
                if self.load_pose:
                    labels["robot_xy"] = torch.from_numpy(robot_xy)
                    labels["robot_yaw"] = torch.from_numpy(robot_yaw)

                yield (
                    torch.from_numpy(vis),
                    torch.from_numpy(prop),
                    torch.from_numpy(cmds),
                    torch.from_numpy(dones),
                    torch.from_numpy(collisions),
                    labels,
                )
        finally:
            for f in open_files.values():
                f.close()
