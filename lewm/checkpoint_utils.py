"""Checkpoint load / save helpers."""
from __future__ import annotations

from typing import Any, Dict

import torch


def clean_state_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Strip ``_orig_mod.`` prefix injected by ``torch.compile()``."""
    return {k.replace("_orig_mod.", ""): v for k, v in d.items()}


def load_jepa_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """Load a CanonicalJEPA checkpoint and return (state_dict, meta)."""
    ckpt = torch.load(path, map_location=device)
    sd = clean_state_dict(ckpt.get("model_state_dict", ckpt))
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return sd, meta


def load_ppo_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """Load a PPO checkpoint and return the model state dict."""
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("model", ckpt)
