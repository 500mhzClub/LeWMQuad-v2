"""Genesis simulation helpers."""
from __future__ import annotations

import os
import numpy as np
import torch


def resolve_sim_backend(sim_backend_arg: str):
    """Return a (backend, description) tuple for gs.init()."""
    import genesis as gs  # noqa: delay import

    sim_backend_arg = sim_backend_arg.lower().strip()
    env_backend = os.getenv("GS_BACKEND", "").lower().strip()
    gpu_backend = getattr(gs, "gpu", None)
    explicit_backends = {
        "cpu": ("cpu", "CPU requested"),
        "gpu": ("gpu", "GPU requested"),
        "cuda": ("cuda", "CUDA requested"),
        "vulkan": ("vulkan", "Vulkan requested"),
        "metal": ("metal", "Metal requested"),
        "amdgpu": ("amdgpu", "AMDGPU requested"),
        "amd": ("amdgpu", "AMD requested"),
        "hip": ("amdgpu", "HIP requested"),
    }

    if sim_backend_arg == "auto" and env_backend and env_backend != "auto":
        sim_backend_arg = env_backend

    if sim_backend_arg == "auto":
        if getattr(torch.version, "hip", None):
            for attr_name, msg in (
                ("amdgpu", "AUTO (ROCm/AMDGPU preferred)"),
                ("vulkan", "AUTO (Vulkan fallback)"),
                ("gpu", "AUTO (GPU fallback)"),
                ("cuda", "AUTO (CUDA fallback)"),
                ("metal", "AUTO (Metal fallback)"),
            ):
                backend = getattr(gs, attr_name, None)
                if backend is not None:
                    return backend, msg
        if gpu_backend is not None:
            return gpu_backend, "AUTO (GPU preferred)"
        for attr_name, msg in (
            ("cuda", "AUTO (CUDA fallback)"),
            ("vulkan", "AUTO (Vulkan fallback)"),
            ("metal", "AUTO (Metal fallback)"),
        ):
            backend = getattr(gs, attr_name, None)
            if backend is not None:
                return backend, msg
        return gs.cpu, "AUTO (CPU fallback)"

    if sim_backend_arg in explicit_backends:
        attr_name, msg = explicit_backends[sim_backend_arg]
        backend = getattr(gs, attr_name, None)
        if backend is not None:
            return backend, msg
        return gs.cpu, f"{msg}; unavailable, falling back to CPU"

    return gs.cpu, f"Unknown backend '{sim_backend_arg}', falling back to CPU"


def init_genesis_once(sim_backend_arg: str = "auto", logging_level=None) -> None:
    import genesis as gs  # noqa: delay import
    import logging
    backend, msg = resolve_sim_backend(sim_backend_arg)
    level = logging.WARNING if logging_level is None else logging_level
    gs.init(backend=backend, logging_level=level)


def to_genesis_target(x: torch.Tensor) -> torch.Tensor:
    """Move a tensor onto the active Genesis device with a safe fallback."""
    import genesis as gs  # noqa: delay import
    x_det = x.detach()
    try:
        x_gs = x_det.to(device=gs.device, dtype=torch.float32)
        return x_gs if x_gs.is_contiguous() else x_gs.contiguous()
    except Exception:
        # Some Genesis backends are happier with a CPU numpy handoff.
        x_np = x_det.to("cpu").numpy().astype(np.float32, copy=True)
        return torch.tensor(x_np, device=gs.device, dtype=torch.float32)


def to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)
