"""PPO actor-critic for System 1 blind walking policy."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 50, act_dim: int = 12, hid: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs))
