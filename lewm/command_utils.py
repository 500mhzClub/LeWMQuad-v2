"""Command generation utilities for diverse locomotion behaviour.

Extends the basic Ornstein-Uhlenbeck (OU) exploration commands from
TinyQuadJEPA-v2 with targeted command patterns that produce training data
for situations the old pipeline missed:

- **retreat/reverse**: negative forward velocity, back out of dead-ends
- **stop/hesitation**: near-zero commands near walls (freeze behaviour)
- **near-collision recovery**: sharp turn + reverse after getting close to walls
- **dead-end backout**: detect dead-end, stop, reverse, turn 180°
- **wall-following**: maintain a fixed offset from a wall surface

Each pattern generator returns a command sequence ``(T, 3)`` with
``[vx, vy, yaw_rate]``.  The physics rollout script can interleave these
with OU exploration and PPO policy commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# OU noise (base exploration) — matches TinyQuadJEPA-v2
# --------------------------------------------------------------------------- #

@dataclass
class OUProcess:
    """Vectorised Ornstein-Uhlenbeck process for correlated exploration noise."""
    n_envs: int
    dim: int = 3
    theta: float = 0.15       # mean-reversion rate
    sigma: float = 0.3        # diffusion
    dt: float = 0.02          # physics timestep
    mu: float = 0.0           # long-term mean

    def __post_init__(self):
        self.state = np.zeros((self.n_envs, self.dim), dtype=np.float32)

    def reset(self, env_ids: Optional[np.ndarray] = None):
        if env_ids is None:
            self.state[:] = 0.0
        else:
            self.state[env_ids] = 0.0

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        dx = (self.theta * (self.mu - self.state) * self.dt
              + self.sigma * np.sqrt(self.dt) * rng.randn(self.n_envs, self.dim))
        self.state = (self.state + dx).astype(np.float32)
        return self.state.copy()


# --------------------------------------------------------------------------- #
# Structured command patterns
# --------------------------------------------------------------------------- #

def generate_retreat_sequence(
    rng: np.random.RandomState,
    length: int = 30,
    speed_range: Tuple[float, float] = (0.1, 0.4),
) -> np.ndarray:
    """Pure backward motion — negative forward velocity, small lateral drift.

    Returns:
        (length, 3) array of [vx, vy, yaw_rate] commands.
    """
    cmds = np.zeros((length, 3), dtype=np.float32)
    speed = rng.uniform(speed_range[0], speed_range[1])
    cmds[:, 0] = -speed  # negative forward
    cmds[:, 1] = rng.uniform(-0.05, 0.05, size=length)  # tiny lateral drift
    cmds[:, 2] = rng.uniform(-0.1, 0.1, size=length)    # small yaw
    return cmds


def generate_stop_sequence(
    rng: np.random.RandomState,
    length: int = 20,
    jitter: float = 0.02,
) -> np.ndarray:
    """Near-zero commands (hesitation / stop near obstacles).

    Returns:
        (length, 3) array of near-zero commands with tiny noise.
    """
    cmds = rng.uniform(-jitter, jitter, size=(length, 3)).astype(np.float32)
    return cmds


def generate_recovery_sequence(
    rng: np.random.RandomState,
    length: int = 40,
    reverse_steps: int = 10,
    turn_steps: int = 15,
    forward_steps: int = 15,
    speed: float = 0.25,
) -> np.ndarray:
    """Near-collision recovery: reverse → sharp turn → cautious forward.

    Models the behaviour of backing away from a wall then turning to escape.

    Returns:
        (length, 3) command array.
    """
    total = reverse_steps + turn_steps + forward_steps
    cmds = np.zeros((total, 3), dtype=np.float32)

    # Phase 1: reverse
    cmds[:reverse_steps, 0] = -speed
    cmds[:reverse_steps, 2] = rng.uniform(-0.1, 0.1, size=reverse_steps)

    # Phase 2: sharp turn in place
    turn_dir = rng.choice([-1.0, 1.0])
    turn_rate = rng.uniform(0.5, 1.0)
    cmds[reverse_steps:reverse_steps + turn_steps, 0] = rng.uniform(-0.05, 0.05, size=turn_steps)
    cmds[reverse_steps:reverse_steps + turn_steps, 2] = turn_dir * turn_rate

    # Phase 3: cautious forward
    t0 = reverse_steps + turn_steps
    cmds[t0:t0 + forward_steps, 0] = speed * 0.7
    cmds[t0:t0 + forward_steps, 2] = rng.uniform(-0.15, 0.15, size=forward_steps)

    # Truncate or pad to requested length
    if total >= length:
        return cmds[:length]
    else:
        pad = np.zeros((length - total, 3), dtype=np.float32)
        return np.concatenate([cmds, pad], axis=0)


def generate_dead_end_backout(
    rng: np.random.RandomState,
    length: int = 60,
    approach_steps: int = 15,
    stop_steps: int = 5,
    reverse_steps: int = 15,
    turn_steps: int = 15,
    speed: float = 0.25,
) -> np.ndarray:
    """Dead-end approach → stop → reverse → 180° turn.

    Returns:
        (length, 3) command array.
    """
    total = approach_steps + stop_steps + reverse_steps + turn_steps
    cmds = np.zeros((total, 3), dtype=np.float32)

    # Approach (slow forward into dead end)
    cmds[:approach_steps, 0] = speed * 0.6
    cmds[:approach_steps, 2] = rng.uniform(-0.1, 0.1, size=approach_steps)

    # Stop / hesitate
    t1 = approach_steps
    cmds[t1:t1 + stop_steps] = rng.uniform(-0.02, 0.02, size=(stop_steps, 3))

    # Reverse
    t2 = t1 + stop_steps
    cmds[t2:t2 + reverse_steps, 0] = -speed
    cmds[t2:t2 + reverse_steps, 2] = rng.uniform(-0.1, 0.1, size=reverse_steps)

    # 180° turn
    t3 = t2 + reverse_steps
    turn_dir = rng.choice([-1.0, 1.0])
    cmds[t3:t3 + turn_steps, 2] = turn_dir * rng.uniform(0.8, 1.2)
    cmds[t3:t3 + turn_steps, 0] = rng.uniform(-0.05, 0.05, size=turn_steps)

    if total >= length:
        return cmds[:length]
    else:
        pad = np.zeros((length - total, 3), dtype=np.float32)
        return np.concatenate([cmds, pad], axis=0)


def generate_wall_following(
    rng: np.random.RandomState,
    length: int = 50,
    speed_range: Tuple[float, float] = (0.15, 0.35),
    lateral_bias: float = 0.0,
    yaw_bias: float = 0.0,
) -> np.ndarray:
    """Wall-following behaviour: steady forward with lateral/yaw bias.

    The bias parameters should be set based on which side the wall is on
    (positive lateral_bias → drift right, positive yaw_bias → turn right).

    Returns:
        (length, 3) command array.
    """
    cmds = np.zeros((length, 3), dtype=np.float32)
    speed = rng.uniform(speed_range[0], speed_range[1])
    cmds[:, 0] = speed + rng.uniform(-0.03, 0.03, size=length)
    cmds[:, 1] = lateral_bias + rng.uniform(-0.05, 0.05, size=length)
    cmds[:, 2] = yaw_bias + rng.uniform(-0.1, 0.1, size=length)
    return cmds


def generate_spin_in_place(
    rng: np.random.RandomState,
    length: int = 25,
    rate_range: Tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    """Spin in place — useful for visual coverage and orientation changes.

    Returns:
        (length, 3) command array.
    """
    cmds = np.zeros((length, 3), dtype=np.float32)
    rate = rng.uniform(rate_range[0], rate_range[1]) * rng.choice([-1.0, 1.0])
    cmds[:, 2] = rate
    cmds[:, 0] = rng.uniform(-0.03, 0.03, size=length)
    return cmds


# --------------------------------------------------------------------------- #
# Mixed command sequence builder
# --------------------------------------------------------------------------- #

# Pattern catalogue with relative weights
COMMAND_PATTERNS = {
    "ou_explore":       0.40,   # default OU exploration
    "retreat":          0.10,
    "stop":             0.08,
    "recovery":         0.12,
    "dead_end_backout": 0.10,
    "wall_follow":      0.10,
    "spin":             0.05,
    "forward_burst":    0.05,
}


def generate_forward_burst(
    rng: np.random.RandomState,
    length: int = 20,
    speed_range: Tuple[float, float] = (0.3, 0.6),
) -> np.ndarray:
    """Short burst of fast forward motion."""
    cmds = np.zeros((length, 3), dtype=np.float32)
    speed = rng.uniform(speed_range[0], speed_range[1])
    cmds[:, 0] = speed
    cmds[:, 1] = rng.uniform(-0.05, 0.05, size=length)
    cmds[:, 2] = rng.uniform(-0.15, 0.15, size=length)
    return cmds


def sample_command_pattern(
    rng: np.random.RandomState,
    length: int = 30,
) -> Tuple[str, np.ndarray]:
    """Sample a command pattern from the catalogue and generate commands.

    Returns:
        (pattern_name, commands) where commands is (length, 3).
    """
    names = list(COMMAND_PATTERNS.keys())
    weights = np.array([COMMAND_PATTERNS[n] for n in names])
    weights /= weights.sum()
    pattern = rng.choice(names, p=weights)

    if pattern == "retreat":
        cmds = generate_retreat_sequence(rng, length=length)
    elif pattern == "stop":
        cmds = generate_stop_sequence(rng, length=length)
    elif pattern == "recovery":
        cmds = generate_recovery_sequence(rng, length=length)
    elif pattern == "dead_end_backout":
        cmds = generate_dead_end_backout(rng, length=length)
    elif pattern == "wall_follow":
        lateral = rng.uniform(-0.1, 0.1)
        yaw = rng.uniform(-0.2, 0.2)
        cmds = generate_wall_following(rng, length=length,
                                       lateral_bias=lateral, yaw_bias=yaw)
    elif pattern == "spin":
        cmds = generate_spin_in_place(rng, length=length)
    elif pattern == "forward_burst":
        cmds = generate_forward_burst(rng, length=length)
    else:
        # Default: OU-like random
        cmds = rng.uniform(-0.3, 0.3, size=(length, 3)).astype(np.float32)

    return pattern, cmds


def build_mixed_command_sequence(
    rng: np.random.RandomState,
    total_steps: int = 1000,
    segment_range: Tuple[int, int] = (20, 80),
) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
    """Build a long command sequence by concatenating random patterns.

    Returns:
        (commands, segments) where commands is (total_steps, 3) and segments
        is a list of (start_step, end_step, pattern_name) for labelling.
    """
    cmds_list: List[np.ndarray] = []
    segments: List[Tuple[int, int, str]] = []
    t = 0

    while t < total_steps:
        seg_len = min(rng.randint(segment_range[0], segment_range[1] + 1),
                      total_steps - t)
        name, cmds = sample_command_pattern(rng, length=seg_len)
        cmds_list.append(cmds)
        segments.append((t, t + seg_len, name))
        t += seg_len

    all_cmds = np.concatenate(cmds_list, axis=0)[:total_steps]
    return all_cmds, segments
